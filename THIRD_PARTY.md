# Third-Party Components

This repository builds an image containing pinned revisions of:

- `Comfy-Org/ComfyUI` at `ca3dbe206c2fea84f2af4371ca13e9f2bfeb82e5`
- `Kosinkadink/ComfyUI-VideoHelperSuite` at
  `4ee72c065db22c9d96c2427954dc69e7b908444b`
- `zai-org/SCAIL-2` at `f998bcc29127ae9b177711ee8f39d65ccd73cca1`
- `zai-org/SCAIL-Pose` at `519c7f54cb972e7f92684213b7ef6c3e05a8f3b2`
- `Comfy-Org/Wan_2.1_ComfyUI_repackaged` at
  `06e001fc51048fb03433a6fb25334de7836704a5`
- `Comfy-Org/SCAIL-2` at `86e1d4f5062f9c518d2b4b66e7aa3cc5110b38ec`
- `Comfy-Org/sam3.1` at `ba901fbc9701054c359ed5240c4d76f83a178108`
- `lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v`
  fast LoRA at `fef288b326f4fed6d2983b9800c35363da31fcfe`

Their own licenses remain applicable. The default runtime path is now ComfyUI
with the bundled Scail2-infinity workflow.

Model weights are not included in this Git repository. They are downloaded
from the sources listed in `config/scail2-runtime.json` and remain subject to
their respective licenses and terms.

SAM3/SAM3.1 models may be gated. Auto-mask requires a Hugging Face account that
has accepted the relevant model license.

`lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v` is optional and
is used by the bundled 6-step ComfyUI workflow.
