#!/usr/bin/env python3
import json
from pathlib import Path
import re
import py_compile


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "config" / "scail2-runtime.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    required_files = [
        ROOT / "Dockerfile",
        ROOT / "scripts" / "start.sh",
        ROOT / "scripts" / "prepare_models.py",
        ROOT / "scripts" / "patch_scail2_attention.py",
        ROOT / "scripts" / "run_scail2.py",
        ROOT / "scripts" / "app.py",
        ROOT / "scripts" / "job_worker.py",
        ROOT / "README.md",
        ROOT / "RUNPOD_STEPS.md",
        RUNTIME,
    ]
    for path in required_files:
        require(path.is_file(), f"Missing required file: {path.relative_to(ROOT)}")

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    scail = runtime["scail2"]
    require(re.fullmatch(r"[0-9a-f]{40}", scail["code_commit"]) is not None, "Bad SCAIL-2 code commit")
    require(re.fullmatch(r"[0-9a-f]{40}", scail["pose_commit"]) is not None, "Bad SCAIL-Pose commit")
    require(re.fullmatch(r"[0-9a-f]{40}", scail["model_revision"]) is not None, "Bad SCAIL-2 model revision")
    require(re.fullmatch(r"[0-9a-f]{40}", scail["scail_weights_revision"]) is not None, "Bad SCAIL-2 weights revision")
    require(scail["code_commit"] in dockerfile, "Dockerfile is missing pinned SCAIL-2 commit")
    require(scail["pose_commit"] in dockerfile, "Dockerfile is missing pinned SCAIL-Pose commit")
    require("patch_scail2_attention.py" in dockerfile, "Dockerfile must patch SCAIL-2 attention fallback")
    require(len(scail["required_files"]) >= 7, "Expected official SCAIL-2 support files")
    require("model/1/fsdp2_rank_0000_checkpoint.pt" not in json.dumps(scail), "FSDP checkpoint should not be downloaded")
    require(scail["scail_path"].endswith("wan2.1_14B_SCAIL_2_fp16.safetensors"), "Expected direct fp16 safetensors")
    require(int(scail["scail_min_bytes"]) >= 32700000000, "SCAIL-2 safetensors min size is too low")
    for item in scail["required_files"]:
        require(int(item["min_bytes"]) > 0, f"Missing min_bytes: {item['path']}")

    sam3 = runtime["sam3"]
    require(sam3["model_repository"] == "facebook/sam3", "SAM3 repo changed unexpectedly")
    require(sam3.get("gated") is True, "SAM3 gated status must be documented")

    lightx2v = runtime["lightx2v_lora"]
    require(
        lightx2v["model_repository"] == "lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v",
        "Lightx2v LoRA repo changed unexpectedly",
    )
    require(
        lightx2v["lora_path"].endswith("Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"),
        "Unexpected Lightx2v LoRA filename",
    )
    require(int(lightx2v["min_bytes"]) >= 700000000, "Lightx2v LoRA min size is too low")
    require(int(lightx2v["recommended_sample_steps"]) == 8, "Lightx2v LoRA should be configured for 8 steps")

    requirements = (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8")
    require("flash_attn" not in requirements, "flash_attn should remain optional; use the SDPA fallback patch")
    require("torch" not in requirements, "base image should provide torch")
    require("ultralytics==8.4.68" in requirements, "SAM3-capable ultralytics must be pinned")

    patch_script = (ROOT / "scripts" / "patch_scail2_attention.py").read_text(encoding="utf-8")
    require("SCAIL2_RUNPOD_SDPA_FALLBACK" in patch_script, "Attention fallback patch marker is missing")
    require("scaled_dot_product_attention" in patch_script, "Attention fallback patch must use torch SDPA")
    app_script = (ROOT / "scripts" / "app.py").read_text(encoding="utf-8")
    run_script = (ROOT / "scripts" / "run_scail2.py").read_text(encoding="utf-8")
    require("Lightning LoRA" in app_script, "UI must expose the Lightx2v preset")
    require("--lightx2v-lora" in run_script, "CLI must expose the Lightx2v LoRA flag")

    for script in [
        "prepare_models.py",
        "patch_scail2_attention.py",
        "run_scail2.py",
        "app.py",
        "job_worker.py",
        "validate_repo.py",
    ]:
        py_compile.compile(str(ROOT / "scripts" / script), doraise=True)

    all_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix in {".md", ".py", ".json", ".txt", ".sh", ".env", ".example", ""}
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.relative_to(ROOT) != Path("scripts/validate_repo.py")
    )
    forbidden = [
        "Wan21-14B-SCAIL-preview",
        "RenderNLFPoses",
        "PoseDetectionVitPoseToDWPose",
        "ComfyUI-WanVideoWrapper",
        "wan21_scail_pose_dance",
    ]
    for token in forbidden:
        require(token not in all_text, f"Old SCAIL-Preview/ComfyUI token remains: {token}")

    print("Repository validation passed.")
    print(f"SCAIL-2 code: {scail['code_commit']}")
    print(f"SCAIL-2 model revision: {scail['model_revision']}")
    print(f"SCAIL-2 weights revision: {scail['scail_weights_revision']}")
    print(f"SAM3 auto-mask: optional gated model")
    print(f"Lightx2v LoRA: optional 8-step acceleration")


if __name__ == "__main__":
    main()
