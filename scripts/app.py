#!/usr/bin/env python3
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import gradio as gr


APP_ROOT = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parents[1]))
WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/workspace/scail2"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(WORKSPACE_DIR / "output")))
JOBS_DIR = Path(os.environ.get("JOBS_DIR", str(WORKSPACE_DIR / "jobs")))
RECENT_JOB_LIMIT = int(os.environ.get("RECENT_JOB_LIMIT", "20"))

SPEED_PRESETS = {
    "Lightning LoRA": {
        "target_w": 512,
        "target_h": 896,
        "sample_steps": 6,
        "sample_shift": 5.0,
        "guidance": 1.0,
        "segment_len": 81,
        "segment_overlap": 5,
        "lightx2v_lora": True,
        "offload_model": False,
    },
    "Draft": {
        "target_w": 512,
        "target_h": 512,
        "sample_steps": 20,
        "sample_shift": 3.0,
        "guidance": 5.0,
        "segment_len": 49,
        "segment_overlap": 3,
        "lightx2v_lora": False,
        "offload_model": True,
    },
    "Balanced": {
        "target_w": 896,
        "target_h": 512,
        "sample_steps": 30,
        "sample_shift": 3.0,
        "guidance": 5.0,
        "segment_len": 65,
        "segment_overlap": 4,
        "lightx2v_lora": False,
        "offload_model": True,
    },
    "Quality": {
        "target_w": 896,
        "target_h": 512,
        "sample_steps": 40,
        "sample_shift": 3.0,
        "guidance": 5.0,
        "segment_len": 81,
        "segment_overlap": 5,
        "lightx2v_lora": False,
        "offload_model": True,
    },
}

JOB_ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def file_path(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return getattr(value, "name", None)


def utc_now():
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp_path.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_job_id(value):
    job_id = (value or "").strip()
    if not job_id or any(char not in JOB_ID_CHARS for char in job_id):
        return None
    return job_id


def job_dir_for(job_id):
    normalized = normalize_job_id(job_id)
    if not normalized:
        return None
    return JOBS_DIR / normalized


def make_job_dir():
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    (job_dir / "inputs").mkdir(parents=True, exist_ok=True)
    return job_id, job_dir


def copy_upload(src, dst_stem):
    src_path = Path(src)
    if not src_path.is_file():
        raise gr.Error(f"Input file does not exist: {src}")
    suffix = src_path.suffix or ".bin"
    dst = dst_stem.with_suffix(suffix)
    shutil.copy2(src_path, dst)
    return dst


def bool_text(value):
    return str(bool(value)).lower()


def build_status_text(status):
    if not status:
        return "No job selected."
    lines = [
        f"Job: {status.get('job_id', '')}",
        f"Status: {status.get('status', 'unknown')}",
        f"Refreshed: {utc_now()}",
    ]
    if status.get("message"):
        lines.append(f"Message: {status['message']}")
    if status.get("created_at"):
        lines.append(f"Created: {status['created_at']}")
    if status.get("started_at"):
        lines.append(f"Started: {status['started_at']}")
    if status.get("finished_at"):
        lines.append(f"Finished: {status['finished_at']}")
    if status.get("returncode") is not None:
        lines.append(f"Return code: {status['returncode']}")
    if status.get("output"):
        lines.append(f"Output: {status['output']}")
    if status.get("log"):
        lines.append(f"Log: {status['log']}")
    return "\n".join(lines)


def build_status_banner(status):
    if not status:
        return "### No job selected\nSubmit a job or load one from Recent jobs."

    state = status.get("status", "unknown")
    job_id = status.get("job_id", "")
    if state == "queued":
        return f"### QUEUED\nJob `{job_id}` is waiting for the worker to start."
    if state == "running":
        return f"### RUNNING\nJob `{job_id}` is still generating. This page can be refreshed safely."
    if state == "succeeded":
        return f"### COMPLETE\nJob `{job_id}` finished. The video is ready below."
    if state == "failed":
        return f"### FAILED\nJob `{job_id}` failed. Check the log tail below for the error."
    return f"### {state.upper()}\nJob `{job_id}` status was refreshed."


def read_log_tail(log_path, max_bytes=20000):
    path = Path(log_path) if log_path else None
    if not path or not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def load_job_status(job_id):
    job_dir = job_dir_for(job_id)
    if not job_dir:
        return None
    status_path = job_dir / "status.json"
    if not status_path.is_file():
        return None
    return read_json(status_path)


def recent_job_choices():
    if not JOBS_DIR.is_dir():
        return []
    candidates = []
    for path in JOBS_DIR.iterdir():
        if not path.is_dir():
            continue
        status_path = path / "status.json"
        if status_path.is_file():
            candidates.append((status_path.stat().st_mtime, path.name))
    candidates.sort(reverse=True)

    choices = []
    for _, job_id in candidates[:RECENT_JOB_LIMIT]:
        status = load_job_status(job_id) or {}
        label = f"{job_id} | {status.get('status', 'unknown')}"
        choices.append((label, job_id))
    return choices


def refresh_recent_jobs(selected=None):
    choices = recent_job_choices()
    values = {value for _, value in choices}
    value = selected if selected in values else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=value)


def apply_preset(preset):
    values = SPEED_PRESETS.get(preset or "Lightning LoRA", SPEED_PRESETS["Lightning LoRA"])
    return (
        values["target_w"],
        values["target_h"],
        values["sample_steps"],
        values["sample_shift"],
        values["guidance"],
        values["segment_len"],
        values["segment_overlap"],
        values["lightx2v_lora"],
        values["offload_model"],
    )


def detached_popen_kwargs():
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {}


def submit_job(
    mode,
    reference_image,
    driving_video,
    prompt,
    auto_mask,
    reference_mask,
    driving_mask,
    target_w,
    target_h,
    sample_steps,
    sample_shift,
    guidance,
    seed,
    segment_len,
    segment_overlap,
    max_persons,
    sam_text,
    matchnearest,
    offload_model,
    lightx2v_lora,
    lora_alpha,
):
    ref = file_path(reference_image)
    driving = file_path(driving_video)
    if not ref or not driving:
        raise gr.Error("Reference image and driving video are required.")
    if not auto_mask and (not file_path(reference_mask) or not file_path(driving_mask)):
        raise gr.Error("Upload both masks, or enable auto-mask with SAM3.")

    target_w = int(target_w)
    target_h = int(target_h)
    if target_w % 32 != 0 or target_h % 32 != 0:
        raise gr.Error("Target width and height must be multiples of 32.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id, job_dir = make_job_dir()
    input_dir = job_dir / "inputs"
    ref = copy_upload(ref, input_dir / "reference")
    driving = copy_upload(driving, input_dir / "driving")
    output = OUTPUT_DIR / f"{job_id}.mp4"
    log_path = job_dir / "generation.log"

    command = [
        sys.executable,
        str(APP_ROOT / "scripts" / "run_scail2.py"),
        "--mode",
        mode,
        "--image",
        str(ref),
        "--driving-video",
        str(driving),
        "--prompt",
        prompt or "",
        "--target-w",
        str(int(target_w)),
        "--target-h",
        str(int(target_h)),
        "--sample-steps",
        str(int(sample_steps)),
        "--sample-shift",
        str(float(sample_shift)),
        "--sample-guide-scale",
        str(float(guidance)),
        "--seed",
        str(int(seed)),
        "--segment-len",
        str(int(segment_len)),
        "--segment-overlap",
        str(int(segment_overlap)),
        "--max-persons",
        str(int(max_persons)),
        "--sam-text",
        *(sam_text or "human character").split(),
        "--offload-model",
        bool_text(offload_model),
        "--output",
        str(output),
    ]
    if lightx2v_lora:
        command.extend(["--lightx2v-lora", "--lora-alpha", str(float(lora_alpha))])
    if auto_mask:
        command.append("--auto-mask")
    else:
        mask_image = copy_upload(file_path(reference_mask), input_dir / "reference_mask")
        mask_video = copy_upload(file_path(driving_mask), input_dir / "driving_mask")
        command.extend(["--mask-image", str(mask_image)])
        command.extend(["--mask-video", str(mask_video)])
    if matchnearest:
        command.append("--matchnearest")

    status = {
        "job_id": job_id,
        "status": "queued",
        "message": "Waiting for the detached worker to start.",
        "created_at": utc_now(),
        "output": str(output),
        "log": str(log_path),
        "settings": {
            "mode": mode,
            "auto_mask": bool(auto_mask),
            "target_w": target_w,
            "target_h": target_h,
            "sample_steps": int(sample_steps),
            "sample_shift": float(sample_shift),
            "sample_guide_scale": float(guidance),
            "segment_len": int(segment_len),
            "segment_overlap": int(segment_overlap),
            "offload_model": bool(offload_model),
            "lightx2v_lora": bool(lightx2v_lora),
            "lora_alpha": float(lora_alpha),
        },
    }
    write_json(job_dir / "status.json", status)
    write_json(
        job_dir / "command.json",
        {
            "command": command,
            "cwd": str(APP_ROOT),
            "log": str(log_path),
            "output": str(output),
        },
    )

    worker_command = [
        sys.executable,
        str(APP_ROOT / "scripts" / "job_worker.py"),
        "--job-dir",
        str(job_dir),
    ]
    subprocess.Popen(
        worker_command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        **detached_popen_kwargs(),
    )

    return (
        job_id,
        build_status_banner(status),
        build_status_text(status),
        None,
        None,
        f"Job started. Refresh this job ID to follow progress:\n{job_id}",
        refresh_recent_jobs(job_id),
    )


def refresh_job(job_id):
    status = load_job_status(job_id)
    if not status:
        return build_status_banner(None), "No job selected.", None, None, ""
    output = status.get("output")
    output_video = output if output and Path(output).is_file() else None
    output_file = output_video
    return (
        build_status_banner(status),
        build_status_text(status),
        output_video,
        output_file,
        read_log_tail(status.get("log")),
    )


def load_selected_job(selected_job):
    banner, status, output_video, output_file, log_tail = refresh_job(selected_job)
    return selected_job or "", banner, status, output_video, output_file, log_tail


with gr.Blocks(title="SCAIL-2 Wan Dance") as demo:
    gr.Markdown(
        "# SCAIL-2 Wan Dance\n"
        "End-to-end Wan2.1 14B character animation/replacement. "
        "Generation runs as a persistent server-side job. "
        "The default preset uses fp8 SCAIL-2 weights plus the LightX2V 6-step LoRA profile."
    )
    with gr.Row():
        with gr.Column():
            mode = gr.Dropdown(["animation", "replacement"], value="animation", label="Mode")
            reference_image = gr.Image(type="filepath", label="Reference image")
            driving_video = gr.Video(label="Driving video")
            prompt = gr.Textbox(
                label="Prompt",
                value="A character is dancing with natural full-body motion, stable identity, detailed clothing, high quality.",
                lines=4,
            )
            auto_mask = gr.Checkbox(value=True, label="Auto-mask with SAM3")
            reference_mask = gr.Image(type="filepath", label="Reference mask image (manual mode)")
            driving_mask = gr.Video(label="Driving mask video (manual mode)")
        with gr.Column():
            speed_preset = gr.Dropdown(
                list(SPEED_PRESETS.keys()),
                value="Lightning LoRA",
                label="Speed preset",
            )
            target_w = gr.Number(value=512, precision=0, label="Target width")
            target_h = gr.Number(value=896, precision=0, label="Target height")
            sample_steps = gr.Number(value=6, precision=0, label="Sampling steps")
            sample_shift = gr.Number(value=5.0, label="Sample shift")
            guidance = gr.Number(value=1.0, label="Guidance scale")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1=random)")
            segment_len = gr.Number(value=81, precision=0, label="Segment length")
            segment_overlap = gr.Number(value=5, precision=0, label="Segment overlap")
            max_persons = gr.Number(value=2, precision=0, label="Auto-mask max persons")
            sam_text = gr.Textbox(value="human character", label="SAM3 text prompts")
            matchnearest = gr.Checkbox(value=False, label="Replacement: match nearest driving actor")
            offload_model = gr.Checkbox(value=False, label="CPU offload (lower VRAM, slower, higher RAM)")
            lightx2v_lora = gr.Checkbox(value=True, label="LightX2V 6-step LoRA")
            lora_alpha = gr.Number(value=1.0, label="LoRA alpha")
            button = gr.Button("Generate")
    with gr.Row():
        job_id = gr.Textbox(label="Job ID")
        recent_jobs = gr.Dropdown(choices=recent_job_choices(), label="Recent jobs")
    with gr.Row():
        refresh_button = gr.Button("Refresh")
        load_button = gr.Button("Load selected job")
        refresh_jobs_button = gr.Button("Refresh job list")
    status_banner = gr.Markdown(value=build_status_banner(None))
    status = gr.Textbox(label="Status", lines=8)
    output_video = gr.Video(label="Output")
    output_file = gr.File(label="Download output")
    log = gr.Textbox(label="Log tail", lines=12)

    speed_preset.change(
        apply_preset,
        inputs=[speed_preset],
        outputs=[
            target_w,
            target_h,
            sample_steps,
            sample_shift,
            guidance,
            segment_len,
            segment_overlap,
            lightx2v_lora,
            offload_model,
        ],
    )
    button.click(
        submit_job,
        inputs=[
            mode,
            reference_image,
            driving_video,
            prompt,
            auto_mask,
            reference_mask,
            driving_mask,
            target_w,
            target_h,
            sample_steps,
            sample_shift,
            guidance,
            seed,
            segment_len,
            segment_overlap,
            max_persons,
            sam_text,
            matchnearest,
            offload_model,
            lightx2v_lora,
            lora_alpha,
        ],
        outputs=[job_id, status_banner, status, output_video, output_file, log, recent_jobs],
    )
    refresh_button.click(
        refresh_job,
        inputs=[job_id],
        outputs=[status_banner, status, output_video, output_file, log],
    )
    load_button.click(
        load_selected_job,
        inputs=[recent_jobs],
        outputs=[job_id, status_banner, status, output_video, output_file, log],
    )
    refresh_jobs_button.click(
        refresh_recent_jobs,
        inputs=[recent_jobs],
        outputs=[recent_jobs],
    )
    timer = gr.Timer(value=10)
    timer.tick(
        refresh_job,
        inputs=[job_id],
        outputs=[status_banner, status, output_video, output_file, log],
        show_progress=False,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=os.environ.get("HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "8188")),
        show_error=True,
    )
