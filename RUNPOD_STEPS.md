# RunPod Setup

## 1. Build the container

Push this repository to GitHub. GitHub Actions publishes:

```text
ghcr.io/YOUR_GITHUB_NAME/wan-dance-runpod:latest
```

Make the GHCR package public, or configure RunPod registry credentials.

## 2. Create the Pod

- Container image: the GHCR image above
- Container disk: 30 GB or more
- Network volume: 100 GB recommended
- Volume mount path: `/workspace`
- Expose HTTP port: `8188`
- GPU: 48 GB VRAM recommended; 24 GB is experimental, and 80 GB-class GPUs are
  better when speed matters

Copy the values from `runpod-template.env.example` into the template variables.

## 3. First start

ComfyUI starts immediately on port `8188`. In the background, the container
downloads the ComfyUI model set:

- `Comfy-Org/SCAIL-2` fp8-scaled SCAIL diffusion model
- `Comfy-Org/Wan_2.1_ComfyUI_repackaged` text encoder, VAE, and CLIP vision
- LightX2V rank64 LoRA
- `Comfy-Org/sam3.1` SAM3.1 checkpoint when `DOWNLOAD_SAM31=1`

Watch `/workspace/logs/wan-dance-startup.log`. If model download is still
running, wait before queueing the workflow.

## 4. Use the UI

Open the Pod HTTP endpoint on port `8188`.

In ComfyUI, open:

```text
scail2-comfyui-infinity-runpod.json
```

Place or upload these input files:

```text
/workspace/ComfyUI/input/reference.png
/workspace/ComfyUI/input/driving.mp4
```

The bundled workflow is already set to 512 x 896, 6 sampling steps, CFG 1.0,
Euler/simple, SD3 shift 5.0, LightX2V rank64 LoRA, and 81-frame windows with a
5-frame overlap.

For the first test, keep the driving video short. If the short test is bad,
avoid spending GPU time on a longer clip with the same input pair.

## 5. Persistent files

```text
/workspace/scail2/models
/workspace/ComfyUI/models
/workspace/ComfyUI/input
/workspace/ComfyUI/output
/workspace/ComfyUI/user/default/workflows
/workspace/logs/wan-dance-startup.log
```

Deleting the Pod does not remove these files when a network volume is attached.
