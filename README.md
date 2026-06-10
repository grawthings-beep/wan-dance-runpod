# Wan SCAIL Dance on RunPod

RunPod-ready ComfyUI image for reference-driven dance video generation with:

- Wan 2.1 SCAIL 14B FP8 scaled
- SCAIL-Pose / NLF 3D pose extraction
- ViTPose + YOLO whole-body pose preprocessing
- Lightx2v step-distillation LoRA
- A bundled ComfyUI workflow

The image pins the RunPod base image by digest and pins all required custom
nodes. Model files are downloaded once to the RunPod network volume and reused
on later starts. The Docker build also runs ComfyUI's custom-node import smoke
test in CPU mode before publishing the image.

## Hardware

- Recommended GPU: 48 GB VRAM
- Practical minimum: 24 GB VRAM with block swapping and at least 64 GB system RAM
- Network volume: 50 GB minimum, 80 GB recommended

The included workflow defaults to 480 x 1216, 65 generated frames, and 6
sampling steps. The video loader reads up to 81 source frames and selects every
second frame. Start lower while testing.

## Quick Start

1. Build the image through the included GitHub Actions workflow.
2. Create a RunPod Pod from `ghcr.io/YOUR_GITHUB_NAME/wan-dance-runpod:latest`.
3. Attach a network volume at `/workspace`.
4. Expose HTTP port `8188`.
5. Wait for the first parallel model download to finish.
6. Open ComfyUI and load `wan21_scail_pose_dance`.
7. Select a reference character image in `LoadImage`.
8. Select a dance video in `VHS_LoadVideo`.
9. Edit the positive and negative prompts, then queue the workflow.

Outputs are saved under `/workspace/comfyui/output`.
Startup, download, and ComfyUI logs are appended to
`/workspace/logs/wan-dance-startup.log`.

## Environment

See `runpod-template.env.example`. `HF_TOKEN` is optional for public files but
recommended to reduce Hugging Face rate-limit failures.

Set `DOWNLOAD_MODELS=0` only after all model files are already present.
Set `FORCE_PINNED_NODES=0` if you intentionally want to manage the five custom
node folders yourself.

Downloads use up to four files in parallel and sixteen aria2 connections per
file by default, matching the LTX template's fast-download settings. Tune
`DOWNLOAD_JOBS`, `ARIA2_CONNECTIONS`, and `ARIA2_SPLITS` if the provider
rate-limits the Pod. Interrupted `.part` files are retained and resumed on the
next start.

`COMFYUI_ARGS` is written into the base image's persistent
`comfyui_args.txt` on every start. The default is `--reserve-vram 3`.

## Managed Models

The model manifest is `config/scail-models.json`. The Hugging Face URLs are
pinned to repository revisions and each file is checked against its expected
size. You can replace URLs or add files without rebuilding the downloader.

## Sources

- SCAIL: https://github.com/zai-org/SCAIL
- SCAIL model: https://huggingface.co/zai-org/SCAIL-Preview
- WanVideoWrapper: https://github.com/kijai/ComfyUI-WanVideoWrapper
- SCAIL-Pose: https://github.com/kijai/ComfyUI-SCAIL-Pose

See `THIRD_PARTY.md` for component and model licensing notes.
