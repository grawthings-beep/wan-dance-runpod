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

- Recommended GPU: 48 GB VRAM or more.
- 24 GB may be tight and should be treated as experimental.
- Network volume: 70 GB minimum, 100 GB recommended.
- First use downloads roughly 45 GB. The image uses the already converted
  `Comfy-Org/SCAIL-2` fp16 safetensors file, so there is no startup-time
  checkpoint conversion.
- The image patches SCAIL-2 attention to fall back to PyTorch SDPA when
  external `flash-attn` wheels are unavailable. This avoids a hard
  `FLASH_ATTN_2_AVAILABLE` assertion on Blackwell/CUDA 12.8 pods, though
  native flash-attn can still be faster when present.

## Quick Start

1. Build the image through the included GitHub Actions workflow.
2. Create a RunPod Pod from `ghcr.io/YOUR_GITHUB_NAME/wan-dance-runpod:latest`.
3. Attach a network volume at `/workspace`.
4. Expose HTTP port `8188`.
5. Set `HF_TOKEN` to a Hugging Face token.
6. Open the Gradio UI on port `8188`.
8. Upload a reference image and a driving video.
9. Enable auto-mask if your token has access to `facebook/sam3`, or upload
   prepared SCAIL-2 mask files manually.

Outputs are saved under `/workspace/scail2/output`.
Logs are appended to `/workspace/logs/wan-dance-startup.log`.

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
- `START_GRADIO=1` starts the web UI on `PORT`, default `8188`.
- `DOWNLOAD_MODELS=0` is useful after the network volume is already prepared.

## Sources

- SCAIL-2 code: https://github.com/zai-org/SCAIL-2
- SCAIL-2 model: https://huggingface.co/zai-org/SCAIL-2
- SCAIL-2 project page: https://teal024.github.io/SCAIL-2/
- SCAIL-Pose masks: https://github.com/zai-org/SCAIL-Pose

See `THIRD_PARTY.md` for component and model licensing notes.
