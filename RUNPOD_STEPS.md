# RunPod Setup

## 1. Build the container

Push this repository to GitHub. GitHub Actions publishes:

```text
ghcr.io/YOUR_GITHUB_NAME/wan-dance-runpod:latest
```

Make the GHCR package public, or configure RunPod registry credentials.

## 2. Create the Pod

- Container image: the GHCR image above
- Container disk: 20 GB or more
- Network volume: 80 GB recommended
- Volume mount path: `/workspace`
- Expose HTTP port: `8188`
- GPU: 48 GB recommended; 24 GB is the practical lower bound

Copy the values from `runpod-template.env.example` into the template variables.

## 3. First start

The first start downloads roughly 32 GB. Four files download in parallel by
default. Watch the Pod logs for `DOWNLOAD`, `READY`, and `SKIP existing`
messages. ComfyUI starts after required files pass exact-size or checksum
validation. Interrupted `.part` files resume on the next start.

If Hugging Face rate-limits the Pod, reduce `DOWNLOAD_JOBS` from `4` to `2`, or
reduce `ARIA2_CONNECTIONS` and `ARIA2_SPLITS` from `16` to `8`.

## 4. Use the workflow

Open ComfyUI and select:

```text
Workflows > Browse > wan21_scail_pose_dance
```

Change these three inputs:

1. `LoadImage`: the character or subject reference
2. `VHS_LoadVideo`: the dance/motion reference
3. `WanVideoTextEncodeCached`: positive and negative prompts

For a first test, use 33-49 frames and 448 x 768. Increase frame count and
resolution after the workflow succeeds.

## 5. Persistent files

```text
/workspace/comfyui/models
/workspace/comfyui/input
/workspace/comfyui/output
/workspace/comfyui/user/default/workflows
/workspace/logs/wan-dance-startup.log
```

Deleting the Pod does not remove these files when a network volume is attached.
