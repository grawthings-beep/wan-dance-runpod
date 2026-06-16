#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys
import uuid

import gradio as gr


APP_ROOT = Path(os.environ.get("APP_ROOT", "/opt/wan-dance"))
WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/workspace/scail2"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(WORKSPACE_DIR / "output")))


def file_path(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return getattr(value, "name", None)


def generate_video(
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
):
    ref = file_path(reference_image)
    driving = file_path(driving_video)
    if not ref or not driving:
        raise gr.Error("Reference image and driving video are required.")
    if not auto_mask and (not file_path(reference_mask) or not file_path(driving_mask)):
        raise gr.Error("Upload both masks, or enable auto-mask with SAM3.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"scail2_{uuid.uuid4().hex[:10]}.mp4"
    command = [
        sys.executable,
        str(APP_ROOT / "scripts" / "run_scail2.py"),
        "--mode",
        mode,
        "--image",
        ref,
        "--driving-video",
        driving,
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
        "--output",
        str(output),
    ]
    if auto_mask:
        command.append("--auto-mask")
    else:
        command.extend(["--mask-image", file_path(reference_mask)])
        command.extend(["--mask-video", file_path(driving_mask)])
    if matchnearest:
        command.append("--matchnearest")

    proc = subprocess.run(command, text=True, capture_output=True)
    log = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise gr.Error(log[-4000:] or "SCAIL-2 generation failed.")
    return str(output), log[-4000:]


with gr.Blocks(title="SCAIL-2 Wan Dance") as demo:
    gr.Markdown(
        "# SCAIL-2 Wan Dance\n"
        "End-to-end Wan2.1 14B character animation/replacement. "
        "No skeleton render is required. Masks are still important; use SAM3 auto-mask "
        "or upload prepared SCAIL-2 masks."
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
            target_w = gr.Number(value=896, precision=0, label="Target width")
            target_h = gr.Number(value=512, precision=0, label="Target height")
            sample_steps = gr.Number(value=40, precision=0, label="Sampling steps")
            sample_shift = gr.Number(value=3.0, label="Sample shift")
            guidance = gr.Number(value=5.0, label="Guidance scale")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1=random)")
            segment_len = gr.Number(value=81, precision=0, label="Segment length")
            segment_overlap = gr.Number(value=5, precision=0, label="Segment overlap")
            max_persons = gr.Number(value=2, precision=0, label="Auto-mask max persons")
            sam_text = gr.Textbox(value="human character", label="SAM3 text prompts")
            matchnearest = gr.Checkbox(value=False, label="Replacement: match nearest driving actor")
            button = gr.Button("Generate")
    output_video = gr.Video(label="Output")
    log = gr.Textbox(label="Log tail", lines=12)
    button.click(
        generate_video,
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
        ],
        outputs=[output_video, log],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=os.environ.get("HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "8188")),
        show_error=True,
    )
