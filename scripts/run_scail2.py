#!/usr/bin/env python3
import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


APP_ROOT = Path(os.environ.get("APP_ROOT", "/opt/wan-dance"))
SCAIL2_REPO = Path(os.environ.get("SCAIL2_REPO", "/opt/SCAIL-2"))
WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/workspace/scail2"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(WORKSPACE_DIR / "output")))
DEFAULT_CKPT_DIR = Path(os.environ.get("SCAIL2_CKPT_DIR", str(WORKSPACE_DIR / "models" / "SCAIL-2")))
DEFAULT_SAFETENSORS = Path(
    os.environ.get("SCAIL2_SAFETENSORS", str(WORKSPACE_DIR / "models" / "SCAIL-2.safetensors"))
)
DEFAULT_SAM3 = Path(os.environ.get("SAM3_MODEL", str(WORKSPACE_DIR / "models" / "sam3" / "sam3.pt")))


def run(command, cwd=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SCAIL2_REPO}:{SCAIL2_REPO / 'SCAIL-Pose'}:{env.get('PYTHONPATH', '')}"
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=str(cwd or SCAIL2_REPO), env=env, check=True)


def copy_input(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def require_path(path, label):
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def make_job_dir():
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job_dir = WORKSPACE_DIR / "jobs" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def prepare_auto_masks(args, job_dir):
    sam3_model = Path(args.sam3_model or DEFAULT_SAM3)
    require_path(sam3_model, "SAM3 model")

    condition_dir = job_dir / "conditions"
    ref_path = copy_input(args.image, condition_dir / "ref.png")
    driving_path = copy_input(args.driving_video, condition_dir / "driving.mp4")

    if args.mode == "animation":
        command = [
            sys.executable,
            str(SCAIL2_REPO / "SCAIL-Pose" / "NLFPoseExtract" / "process_animation_aio.py"),
            "--subdir",
            str(condition_dir),
            "--e2e_mode",
            "--max_persons",
            str(args.max_persons),
            "--sam3_model",
            str(sam3_model),
            "--text",
            *args.sam_text,
        ]
        run(command, cwd=SCAIL2_REPO / "SCAIL-Pose")
        return {
            "image": ref_path,
            "mask_image": condition_dir / "ref_mask.jpg",
            "pose": condition_dir / "rendered_v2.mp4",
            "mask_video": condition_dir / "rendered_mask_v2.mp4",
        }

    command = [
        sys.executable,
        str(SCAIL2_REPO / "SCAIL-Pose" / "NLFPoseExtract" / "process_replacement.py"),
        "--subdir",
        str(condition_dir),
        "--sam3_model",
        str(sam3_model),
        "--text",
        *args.sam_text,
    ]
    if args.matchnearest:
        command.append("--matchnearest")
    run(command, cwd=SCAIL2_REPO / "SCAIL-Pose")
    return {
        "image": ref_path,
        "mask_image": condition_dir / "ref_mask.png",
        "pose": condition_dir / "rendered_v2.mp4",
        "mask_video": condition_dir / "replace_mask.mp4",
    }


def prepare_manual_inputs(args):
    require_path(args.mask_image, "reference mask image")
    require_path(args.mask_video, "driving mask video")
    return {
        "image": Path(args.image),
        "mask_image": Path(args.mask_image),
        "pose": Path(args.driving_video),
        "mask_video": Path(args.mask_video),
    }


def generate(args, inputs, output_file):
    require_path(DEFAULT_CKPT_DIR / "Wan2.1_VAE.pth", "SCAIL-2 checkpoint directory")
    require_path(DEFAULT_SAFETENSORS, "converted SCAIL-2 safetensors")
    for label, path in inputs.items():
        require_path(path, label)

    command = [
        sys.executable,
        str(SCAIL2_REPO / "generate.py"),
        "--model",
        "SCAIL-14B",
        "--ckpt_dir",
        str(DEFAULT_CKPT_DIR),
        "--scail_path",
        str(DEFAULT_SAFETENSORS),
        "--scail_config_path",
        str(SCAIL2_REPO / "configs" / "config-14b.json"),
        "--target_w",
        str(args.target_w),
        "--target_h",
        str(args.target_h),
        "--image",
        str(inputs["image"]),
        "--mask_image",
        str(inputs["mask_image"]),
        "--pose",
        str(inputs["pose"]),
        "--mask_video",
        str(inputs["mask_video"]),
        "--prompt",
        args.prompt or "",
        "--save_file",
        str(output_file),
        "--sample_steps",
        str(args.sample_steps),
        "--sample_shift",
        str(args.sample_shift),
        "--sample_guide_scale",
        str(args.sample_guide_scale),
        "--sample_solver",
        args.sample_solver,
        "--segment_len",
        str(args.segment_len),
        "--segment_overlap",
        str(args.segment_overlap),
        "--base_seed",
        str(args.seed),
        "--offload_model",
        str(args.offload_model).lower(),
    ]
    if args.mode == "replacement":
        command.append("--replace_flag")
    if args.lora_path:
        command.extend(["--lora_path", args.lora_path, "--lora_alpha", str(args.lora_alpha)])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    run(command, cwd=SCAIL2_REPO)
    print(f"OUTPUT={output_file}", flush=True)
    return output_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["animation", "replacement"], default="animation")
    parser.add_argument("--image", required=True)
    parser.add_argument("--driving-video", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--mask-image")
    parser.add_argument("--mask-video")
    parser.add_argument("--auto-mask", action="store_true")
    parser.add_argument("--sam3-model")
    parser.add_argument("--sam-text", nargs="+", default=["human", "character"])
    parser.add_argument("--max-persons", type=int, default=2)
    parser.add_argument("--matchnearest", action="store_true")
    parser.add_argument("--target-w", type=int, default=896)
    parser.add_argument("--target-h", type=int, default=512)
    parser.add_argument("--sample-steps", type=int, default=40)
    parser.add_argument("--sample-shift", type=float, default=3.0)
    parser.add_argument("--sample-guide-scale", type=float, default=5.0)
    parser.add_argument("--sample-solver", choices=["unipc", "dpm++"], default="unipc")
    parser.add_argument("--segment-len", type=int, default=81)
    parser.add_argument("--segment-overlap", type=int, default=5)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--offload-model", type=lambda value: str(value).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--lora-path")
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    require_path(args.image, "reference image")
    require_path(args.driving_video, "driving video")
    job_dir = make_job_dir()
    inputs = prepare_auto_masks(args, job_dir) if args.auto_mask else prepare_manual_inputs(args)
    output_file = Path(args.output) if args.output else OUTPUT_DIR / f"{job_dir.name}.mp4"
    generate(args, inputs, output_file)


if __name__ == "__main__":
    main()
