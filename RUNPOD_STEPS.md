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
- Network volume: 120 GB recommended
- Volume mount path: `/workspace`
- Expose HTTP port: `8188`
- GPU: 48 GB recommended

Copy the values from `runpod-template.env.example` into the template variables.

## 3. First start

The first start downloads the official SCAIL-2 checkpoint and converts it to
`/workspace/scail2/models/SCAIL-2.safetensors`. This is large and can take a
while. Watch `/workspace/logs/wan-dance-startup.log`.

Set `DOWNLOAD_SAM3=1` only if the `HF_TOKEN` account has access to
`facebook/sam3`. Without SAM3, the UI still works in manual-mask mode.

## 4. Use the UI

Open the Pod HTTP endpoint on port `8188`.

For a first test, use 512 x 896 or 896 x 512 and keep the driving video short.
The default SCAIL-2 segment length is 81 frames at 16 fps.

## 5. Persistent files

```text
/workspace/scail2/models
/workspace/scail2/input
/workspace/scail2/output
/workspace/scail2/jobs
/workspace/logs/wan-dance-startup.log
```

Deleting the Pod does not remove these files when a network volume is attached.
