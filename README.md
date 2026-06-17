# SCAIL-2 Wan Dance on RunPod

RunPod-ready image for SCAIL-2 end-to-end character animation and replacement.

This repo now uses the official `zai-org/SCAIL-2` `wan-scail2` inference code
directly instead of the older ComfyUI SCAIL-Preview pose workflow.

## What Changed

- SCAIL-2 14B, based on Wan2.1 I2V 14B.
- No skeleton render is required for normal animation. The driving video itself
  is passed to SCAIL-2.
- Masks are still required by the model. Auto-mask uses SCAIL-Pose in
  `--e2e_mode`, which skips NLF/DWPose skeleton extraction and uses SAM3 masks.
- ComfyUI custom nodes and workflow JSON were removed. The container exposes a
  small Gradio UI and a CLI wrapper.

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
- First use downloads roughly 46 GB. The largest file is the already converted
  `Comfy-Org/SCAIL-2` fp16 safetensors file, which is validated as at least
  32.7 GB. The SCAIL-2 support files add several more GB, and optional SAM3
  mask weights add at least 1 GB. The optional Lightx2v 8-step LoRA adds about
  739 MB and is enabled by default because it is the practical first-pass mode.
  Keep extra volume space for job inputs, logs, and generated videos.
- The image patches SCAIL-2 attention to fall back to PyTorch SDPA when
  external `flash-attn` wheels are unavailable. This avoids a hard
  `FLASH_ATTN_2_AVAILABLE` assertion on Blackwell/CUDA 12.8 pods, though
  native flash-attn can still be faster when present.

## Runtime Behavior

- The Gradio UI now submits persistent server-side jobs instead of running the
  full generation inside the browser request.
- You can leave or refresh the page while a job is running. Reopen the UI, load
  the job from Recent jobs, or paste the Job ID to view status, logs, and output.
- The UI shows QUEUED, RUNNING, COMPLETE, or FAILED above the raw status text.
- Completed jobs expose both a video preview and a Download output file link.
- Job metadata and logs are stored under `/workspace/scail2/jobs/<job_id>/`.
- Finished videos are saved under `/workspace/scail2/output/`.

## Speed Controls

Generation time is dominated by video length, resolution, sampling steps,
segment length, and CPU/GPU offload.

- Use the UI Speed preset:
  - Lightning LoRA: 896 x 512, 8 steps, Lightx2v LoRA, fast first-pass mode.
  - Draft: 512 x 512, 20 steps, shorter segments.
  - Balanced: 896 x 512, 30 steps, medium segments.
  - Quality: 896 x 512, 40 steps, longer segments.
- Keep the driving video short for tests. Runtime scales with the number of
  frames.
- Reducing sampling steps is usually close to linear: 20 steps is much faster
  than 40, with a quality tradeoff.
- Lower resolution reduces memory and runtime sharply.
- CPU offload lowers VRAM pressure but uses more system RAM and is slower. On
  48 GB A40 pods, start with Lightning LoRA and CPU offload disabled. Enable
  offload only if you hit CUDA VRAM errors and the pod has enough system RAM.
- Auto-mask with SAM3 adds preprocessing time. Manual masks avoid that step.
- Lightning LoRA uses
  `lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v` and passes
  `--sample_steps 8 --sample_shift 1 --sample_guide_scale 1 --lora_alpha 1`.
  Treat it as the default cost-control mode before spending time on Quality.

## Quick Start

1. Build the image through the included GitHub Actions workflow.
2. Create a RunPod Pod from `ghcr.io/YOUR_GITHUB_NAME/wan-dance-runpod:latest`.
3. Attach a network volume at `/workspace`.
4. Expose HTTP port `8188`.
5. Set `HF_TOKEN` to a Hugging Face token.
6. Open the Gradio UI on port `8188`.
7. Upload a reference image and a driving video.
8. Choose Lightning LoRA for the first pass, then Draft, Balanced, or Quality
   if the result is promising.
9. Enable auto-mask if your token has access to `facebook/sam3`, or upload
   prepared SCAIL-2 mask files manually.

Outputs are saved under `/workspace/scail2/output`.
Startup logs are appended to `/workspace/logs/wan-dance-startup.log`.
Per-generation logs are saved under `/workspace/scail2/jobs/<job_id>/`.

## Masks

SCAIL-2 is end-to-end with respect to motion: `--pose` can be the original
driving video. It is not mask-free.

For best quality, use SAM3 auto-mask:

```bash
python /opt/wan-dance/scripts/run_scail2.py \
  --auto-mask \
  --mode animation \
  --image /workspace/scail2/input/ref.png \
  --driving-video /workspace/scail2/input/driving.mp4 \
  --prompt "A character is dancing with natural full-body motion." \
  --output /workspace/scail2/output/out.mp4
```

Manual masks also work:

```bash
python /opt/wan-dance/scripts/run_scail2.py \
  --mode animation \
  --image ref.png \
  --mask-image ref_mask.jpg \
  --driving-video driving.mp4 \
  --mask-video rendered_mask_v2.mp4 \
  --prompt "The character is dancing." \
  --output out.mp4
```

For replacement, add `--mode replacement`; with auto-mask, add `--matchnearest`
when the driving video contains more than one actor and the nearest matching
track should be selected.

## Environment

See `runpod-template.env.example`.

- `DOWNLOAD_MODELS=1` downloads SCAIL-2 support files and direct fp16 weights.
- `PREPARE_MODELS_BACKGROUND=1` starts the UI immediately while model files
  download in the background.
- `ENSURE_MODELS=1` makes generation wait for missing model files instead of
  failing immediately.
- `DOWNLOAD_SAM3=1` pre-downloads `facebook/sam3` for auto-mask. If this is
  `0`, the first auto-mask generation downloads it on demand. This model is
  gated, so the `HF_TOKEN` account must have accepted the license.
- `DOWNLOAD_LIGHTX2V_LORA=1` pre-downloads the Lightx2v 8-step LoRA. Leave this
  enabled unless you only want the original 30-40 step SCAIL-2 path.
- `START_GRADIO=1` starts the web UI on `PORT`, default `8188`.
- `DOWNLOAD_MODELS=0` is useful after the network volume is already prepared.

## Sources

- SCAIL-2 code: https://github.com/zai-org/SCAIL-2
- SCAIL-2 model: https://huggingface.co/zai-org/SCAIL-2
- SCAIL-2 project page: https://teal024.github.io/SCAIL-2/
- SCAIL-Pose masks: https://github.com/zai-org/SCAIL-Pose
- Lightx2v 8-step LoRA:
  https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v

See `THIRD_PARTY.md` for component and model licensing notes.
