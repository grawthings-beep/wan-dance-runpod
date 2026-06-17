# comfyui-scail2-infinity

This custom node adds `SCAIL-2 Infinity (auto window)` to ComfyUI.

It wraps the native ComfyUI SCAIL-2 conditioning path and automatically repeats
fixed SCAIL-2 windows:

- 81 frames per window
- 5 previous frames as clean overlap
- 76 new frames per window after the first
- final stitched output trimmed to the driving video length

The node reuses ComfyUI's own `WanSCAILToVideo`, `common_ksampler`, and VAE
decode logic. It does not reimplement the SCAIL model itself.

The bundled RunPod workflow is:

```text
workflows/scail2-comfyui-infinity-runpod.json
```
