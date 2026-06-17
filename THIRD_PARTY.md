# Third-Party Components

This repository builds an image containing pinned revisions of:

- `zai-org/SCAIL-2` at `f998bcc29127ae9b177711ee8f39d65ccd73cca1`
- `zai-org/SCAIL-Pose` at `519c7f54cb972e7f92684213b7ef6c3e05a8f3b2`
- `lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v`
  fast LoRA at `fef288b326f4fed6d2983b9800c35363da31fcfe`

Their own licenses remain applicable. The runtime wrapper follows the official
SCAIL-2 command-line inference path rather than a ComfyUI graph.

Model weights are not included in this Git repository. They are downloaded
from the sources listed in `config/scail2-runtime.json` and remain subject to
their respective licenses and terms.

`facebook/sam3` is optional and gated. Auto-mask requires a Hugging Face account
that has accepted that model's license.
