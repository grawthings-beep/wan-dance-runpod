# SCAIL-2 Wan Dance on RunPod

RunPod-ready ComfyUI image for SCAIL-2 end-to-end character animation and
replacement.

This repo now defaults back to ComfyUI. The container ships a pinned current
ComfyUI build, VideoHelperSuite, a trimmed Scail2-infinity workflow, and a
local `WanSCAILInfinity` custom node.

## What Changed

- SCAIL-2 14B, based on Wan2.1 I2V 14B, runs through ComfyUI's native SCAIL
  nodes instead of the direct Python wrapper.
- No skeleton render is required for normal animation. The driving video itself
  is passed into the workflow.
- Masks are still required by the model. The workflow uses ComfyUI SAM3.1
  tracking plus `SCAIL2ColoredMask`.
- Long driving videos use the included Scail2-infinity node:
  81-frame windows, 5-frame overlap, stitched and trimmed automatically.
- The RunPod workflow is trimmed to the core path: no Pusa LoRA, RIFE, RTX
  upscaler, EasyUse post-processing, or KJ SageAttention dependency.
- The default workflow uses Comfy-Org fp8-scaled SCAIL-2 weights, the LightX2V
  rank64 step-distill LoRA, 6 sampling steps, CFG 1.0, and SD3 shift 5.0.

## Hardware

- Recommended GPU: 48 GB VRAM or more for the Lightning LoRA first-pass path.
- 24 GB VRAM is experimental. Use Draft settings, short videos, and CPU
  offload; out-of-memory failures are still possible.
- 48 GB GPUs should start with CPU offload disabled for Lightning LoRA. CPU
  offload reduces VRAM pressure but can exhaust RunPod container RAM on pods
  with roughly 46 GB system memory.
- 80 GB-class GPUs are recommended when speed matters.
- Network volume: 70 GB minimum, 100 GB recommended.
- Container disk: 30 GB or more.
- First use downloads roughly 28 GB before SAM3.1. The image uses the already
  converted `Comfy-Org/SCAIL-2` fp8-scaled safetensors file plus a LightX2V
  step-distill LoRA, so there is no startup-time checkpoint conversion. The
  ComfyUI Wan text encoder, VAE, CLIP vision, and SAM3.1 checkpoint add several
  more GB.
  Keep extra volume space for job inputs, logs, and generated videos.

## Runtime Behavior

- ComfyUI starts on port `8188`.
- The bundled workflow is copied to:
  `/workspace/ComfyUI/user/default/workflows/scail2-comfyui-infinity-runpod.json`
- Put `reference.png` and `driving.mp4` in `/workspace/ComfyUI/input`, or change
  the Load Image / Load Video nodes in the workflow.
- Finished videos are saved under `/workspace/ComfyUI/output/SCAIL/`.

## Speed Controls

Generation time is dominated by video length, resolution, sampling steps,
segment length, and CPU/GPU offload.

- Use the bundled workflow as the first-pass profile:
  512 x 896, 6 steps, CFG 1.0, Euler/simple, SD3 shift 5.0, LightX2V rank64
  LoRA, 81-frame windows, 5-frame overlap.
- Keep the driving video short for tests. Runtime scales with the number of
  frames.
- Reducing sampling steps is usually close to linear: 20 steps is much faster
  than 40, with a quality tradeoff.
- Lower resolution reduces memory and runtime sharply.
- For the default Lightning LoRA profile, treat 48 GB VRAM as the practical
  target. 24 GB GPUs are not enough for SCAIL-2 14B, and 32 GB GPUs are still
  likely to struggle.
- SAM3.1 tracking adds preprocessing time. Manual masks avoid that step.
- Lightning LoRA uses
  `lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v` and passes
  `--sample_steps 6 --sample_shift 5 --sample_guide_scale 1 --lora_alpha 1`.
  Treat it as the default cost-control mode before spending time on Quality.

## Quick Start

1. Build the image through the included GitHub Actions workflow.
2. Create a RunPod Pod from `ghcr.io/YOUR_GITHUB_NAME/wan-dance-runpod:latest`.
3. Attach a network volume at `/workspace`.
4. Expose HTTP port `8188`.
5. Set `HF_TOKEN` to a Hugging Face token.
6. Open ComfyUI on port `8188`.
7. In ComfyUI, open `scail2-comfyui-infinity-runpod.json`.
8. Upload or place `reference.png` and `driving.mp4` in
   `/workspace/ComfyUI/input`.
9. Queue the workflow. Start short before spending time on long clips.

Outputs are saved under `/workspace/ComfyUI/output/SCAIL`.
Startup logs are appended to `/workspace/logs/wan-dance-startup.log`.

## Masks

SCAIL-2 is end-to-end with respect to motion: the driving video can be the
original dance/replacement video. It is not mask-free. The bundled workflow
tracks the reference and driving person with SAM3.1, then uses
`SCAIL2ColoredMask` to build the colored SCAIL masks.

## Environment

See `runpod-template.env.example`.

- `START_COMFYUI=1` starts ComfyUI on `PORT`, default `8188`.
- `DOWNLOAD_MODELS=1` downloads ComfyUI SCAIL-2/Wan/SAM3.1 model files.
- `REFRESH_RUNTIME_CONFIG=1` refreshes `/workspace/config/scail2-runtime.json`
  from the image on startup. Keep this on when changing model defaults.
- `PREPARE_MODELS_BACKGROUND=1` starts the UI immediately while model files
  download in the background.
- `DOWNLOAD_SAM31=1` pre-downloads the ComfyUI SAM3.1 checkpoint. This may be
  gated, so the `HF_TOKEN` account must have accepted the license.
- `DOWNLOAD_LIGHTX2V_LORA=1` pre-downloads the LightX2V 6-step LoRA. Leave this
  enabled for the bundled workflow.
- `PREPARE_LOCK_STALE_SECONDS=43200` controls when an abandoned model-download
  lock is removed. The lock uses a directory instead of `flock` so it works on
  RunPod `/workspace` volumes that reject POSIX file locks.
- `START_COMFYUI=0 START_GRADIO=1` starts the old direct Gradio wrapper for
  debugging only.
- `DOWNLOAD_MODELS=0` is useful after the network volume is already prepared.

## Sources

- SCAIL-2 code: https://github.com/zai-org/SCAIL-2
- SCAIL-2 model: https://huggingface.co/zai-org/SCAIL-2
- SCAIL-2 project page: https://teal024.github.io/SCAIL-2/
- SCAIL-Pose masks: https://github.com/zai-org/SCAIL-Pose
- ComfyUI: https://github.com/Comfy-Org/ComfyUI
- ComfyUI VideoHelperSuite: https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
- ComfyUI Wan repackaged models:
  https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged
- ComfyUI SAM3.1 checkpoint: https://huggingface.co/Comfy-Org/sam3.1
- LightX2V 6-step LoRA:
  https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v

See `THIRD_PARTY.md` for component and model licensing notes.
