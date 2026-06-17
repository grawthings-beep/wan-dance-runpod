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

The UI starts immediately. In the background, the container downloads the
official SCAIL-2 support files plus the already-converted Comfy-Org fp8-scaled
safetensors file and the LightX2V 6-step LoRA. Watch
`/workspace/logs/wan-dance-startup.log`.

Set `DOWNLOAD_SAM3=1` only if the `HF_TOKEN` account has access to
`facebook/sam3` and you want to prefetch it. Without prefetch, the first
auto-mask generation downloads SAM3 on demand. The UI still works in manual-mask
mode without SAM3.

## 4. Use the UI

Open the Pod HTTP endpoint on port `8188`.

For a first test, use 512 x 896 or 896 x 512 and keep the driving video short.
The default UI profile uses the fast workflow-style settings: 512 x 896,
6 sampling steps, guidance 1.0, shift 5.0, LightX2V fast LoRA, and an
81-frame SCAIL-2 segment length at 16 fps.

For longer driving videos, SCAIL-2 is patched to use Scail2-infinity-style
auto-windowing: 81-frame windows, 5-frame overlap, repeated final-frame padding
for the last short window, and exact output trimming.

The container applies a build-time SCAIL-2 attention patch so generation can
fall back to PyTorch SDPA when `flash-attn` is not available. If you still hit
memory errors, reduce sampling steps, resolution, or segment length before
increasing video duration. CPU offload lowers VRAM use but is slower and can
raise system-RAM use enough to trigger RunPod container OOM. Use it only when
CUDA VRAM is the limiting factor and the pod has enough system RAM.

If Lightning LoRA does not produce a usable preview on a short clip, avoid
spending GPU time on longer Balanced or Quality jobs for the same input pair.

Generation runs as a persistent job. The UI returns a Job ID immediately, and
the actual SCAIL-2 process continues on the server if the browser tab is closed
or refreshed. Reopen the UI and select the job from Recent jobs, or paste the
Job ID to check status, logs, and output.

## 5. Persistent files

```text
/workspace/scail2/models
/workspace/scail2/input
/workspace/scail2/output
/workspace/scail2/jobs
/workspace/logs/wan-dance-startup.log
```

Deleting the Pod does not remove these files when a network volume is attached.
