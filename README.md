# SCAIL-2 Quality Dance on RunPod

Animate one reference character from a dance video in ComfyUI. The bundle now
includes a quality comparison path alongside the original six-step fast workflow.

## Choose a workflow

| Workflow | Purpose | Sampling |
| --- | --- | --- |
| `scail2-preflight-runpod.json` | **Start here after updating:** verify generation, tiled decode and MP4 output; not a quality test | 9 frames, 2 steps, about 0.25 MP |
| `scail2-quality-runpod.json` | Quality comparison: one reference, DPO detail LoRA | 40 steps, CFG 5, UniPC/simple, shift 3 |
| `scail2-quality-multiref-runpod.json` | Experimental: add one face/back/costume view of the same character | Same quality settings |
| `scail2-comfyui-infinity-runpod.json` | Original fast workflow, retained unchanged | LightX2V rank64, 6 steps, CFG 1, Euler/simple, shift 5 |

**日本語の操作・比較手順: [QUALITY_JA.md](docs/QUALITY_JA.md)**

Quality workflows use the existing pinned FP8-scaled SCAIL-2 model. They do not
load LightX2V. The official DPO LoRA starts at strength 1; set its strength to 0
for an otherwise identical comparison. DPO targets details such as hands, lips
and eyes, and is not a guaranteed identity lock.

The positive prompt is no longer empty. Edit it to describe the actual
character's face, hair, costume, accessories and background. The generic default
does not infer those details from the image. The note in the graph includes an
example, which must be adapted rather than copied as a different identity.

Quality workflows start at **24 fps and 121 frames** (about five seconds,
including a window boundary). For a full clip, set both Load Video's
`frame_load_cap` and Infinity's `max_frames` to 0. Input resampling and output
encoding use the same FPS control, and the original audio is connected.

**Resolution is image-driven:** the reference is scaled to approximately
0.9 megapixels and aligned to multiples of 32; its width/height are connected
to the generator. The unconnected-looking 512×896 widget values are not the
effective resolution when those inputs are linked.

## Extra reference views

The experimental graph takes the main reference plus one additional image of
the **same character in the same outfit**, such as a face close-up or back view.
Replace the second Load Image's default `reference.png` with a real extra view.

Each view is segmented independently. Both SAM tracks select object 0 after
sorting by area, binding that person to blue. Inspect the previews and change
the SAM subject descriptions if another person is selected. The main reference
has a white mask background; extra references have black backgrounds so unrelated
scenery does not become target-scene evidence.

The bundled `WanSCAILReferenceBatch` fits the extra image and mask to the main
image's canvas without cropping, keeps the main image first, and rejects missing
or mismatched identity masks. The existing Infinity node passes the complete
reference batch to the native SCAIL conditioning on every window.

Multi-reference SCAIL-2 is experimental: the upstream authors explicitly warn
that more references can degrade quality. Compare the two quality graphs with
the same prompt, source trim, seed, resolution and DPO strength before adopting it.

## Hardware and validation scope

- The existing fast path targets 48 GB VRAM or more. 24 GB remains experimental.
- For the first quality/multi-reference GPU comparison, use an 80 GB-class Pod
  if available. This is a conservative test target, **not a measured minimum**.
- A user run on RTX 6000 Ada 48 GB took about 87 minutes for one 40-step window,
  then exposed a tiled-decode argument bug. The fix routes decoding through the
  core VAEDecodeTiled node with explicit overlaps. GPU completion and output
  quality of the corrected run still need validation; start with preflight.
- Keep the base image's CUDA/PyTorch stack. ComfyUI, VideoHelperSuite and the
  legacy SCAIL code pins remain unchanged.
- Network volume: 100 GB recommended at `/workspace`; container disk: 30 GB+.
- Quality adds approximately **1.23 GB** for the DPO LoRA. Its revision and
  SHA-256 are pinned; it is downloaded on the Pod, not baked into the image.

CI validates workflow generation, graph wiring, model availability in the
manifest, offline download behavior and workflow backups. The image build runs
a **CPU smoke test against the actual pinned ComfyUI**, including reference
geometry, mask semantics, multi-reference conditioning, the actual 3D tiler
regression and recovery-latent roundtrip. This does not run
the 14B model or validate GPU-generated videos.

## Quick start

1. Build through GitHub Actions and use the image's immutable commit tag:
   `ghcr.io/grawthings-beep/wan-dance-runpod:<40-character-commit>`.
2. Mount the volume at `/workspace`, expose HTTP port `8188`, and apply
   [runpod-template.env.example](runpod-template.env.example).
3. Provide `HF_TOKEN` through a RunPod Secret when required for model access.
4. Open ComfyUI and wait for `ComfyUI model files ready` in
   `/workspace/logs/wan-dance-startup.log` before queueing.
5. Open `scail2-preflight-runpod.json`, upload the reference and dance video,
   and verify that the nine-frame test saves an MP4. Then open the quality graph,
   describe the actual character in the prompt, and run the quality comparison.

Outputs go to `/workspace/ComfyUI/output/SCAIL/`. Each profile has a distinct
filename prefix. Full setup: [RUNPOD_STEPS.md](RUNPOD_STEPS.md).

Startup installs all bundled workflows. If an existing bundled filename differs,
it first copies that file into a timestamped `workflow-backups` directory next
to the workflows directory. Files saved under your own names are untouched.
An already-open browser canvas must be reopened to use the updated graph.

If decoding fails, the affected window's sampled latent is saved to
`/workspace/ComfyUI/output/SCAIL/recovery/` before the original error is raised.
The `.latent` file can be loaded and decoded separately; it is not an automatic
resume of all windows. Older failed jobs did not save this recovery file.

## Runtime behavior

- ComfyUI code: `/opt/ComfyUI`; persistent inputs, outputs and user workflows:
  `/workspace/ComfyUI`.
- `START_COMFYUI=1` serves on `PORT` (8188).
- `REFRESH_RUNTIME_CONFIG=1` refreshes the bundled download manifest.
- `DOWNLOAD_MODELS=1` downloads the base model, DPO and support assets.
- `DOWNLOAD_SAM31=1` downloads the segmentation checkpoint.
- `DOWNLOAD_LIGHTX2V_LORA=1` is retained for the fast/debug path; the ComfyUI
  bundle prepares both quality and fast workflow assets.
- `DOWNLOAD_MODELS=0` requires cached assets, including DPO; model preparation
  must not fetch a missing LightX2V LoRA in offline mode.
- `PREPARE_MODELS_BACKGROUND=1` starts the UI before model preparation finishes.
- `PREPARE_LOCK_STALE_SECONDS=43200` recovers abandoned directory locks on
  volumes without working POSIX file locks.
- The old direct Gradio wrapper remains a debugging path via
  `START_COMFYUI=0 START_GRADIO=1`; the new quality workflows are ComfyUI-only.

If the proxy returns HTTP 403 while `curl http://127.0.0.1:8188/` works inside
the Pod, check the exposed HTTP port and use the RunPod service link:
`https://POD_ID-8188.proxy.runpod.net`.

## Local checks

```bash
python scripts/build_quality_workflows.py --check
python scripts/validate_repo.py
python -m unittest discover -s tests -v
bash -n scripts/start.sh
# Inside the built image (CPU; does not generate a video):
python3.12 /opt/wan-dance/scripts/container_smoke.py
```

## Sources

- [SCAIL-2 code and multi-reference limitations](https://github.com/zai-org/SCAIL-2)
- [Pinned ComfyUI SCAIL-2 conditioning](https://github.com/Comfy-Org/ComfyUI/blob/ca3dbe206c2fea84f2af4371ca13e9f2bfeb82e5/comfy_extras/nodes_scail.py)
- [Pinned DPO LoRA](https://huggingface.co/Comfy-Org/SCAIL-2/blob/86e1d4f5062f9c518d2b4b66e7aa3cc5110b38ec/loras/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors)
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI)
- [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [SAM3.1 checkpoint](https://huggingface.co/Comfy-Org/sam3.1)
- [LightX2V fast LoRA](https://huggingface.co/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v)

See [THIRD_PARTY.md](THIRD_PARTY.md) for component/model terms.
