#!/usr/bin/env python3
import sys
from pathlib import Path


PATCH_MARKER = "SCAIL2_RUNPOD_LOW_MEMORY_MODEL_LOADING"


IMPORT_TARGET = "from safetensors.torch import load_file\n"
IMPORT_REPLACEMENT = "from safetensors import safe_open\nfrom safetensors.torch import load_file\n"


HELPER_TARGET = "class SCAIL2Pipeline:\n"
HELPER_REPLACEMENT = '''def _load_safetensors_into_model(model, filename):
    model_state = model.state_dict()
    missing = set(model_state.keys())
    unexpected = []

    with safe_open(filename, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key not in model_state:
                unexpected.append(key)
                continue
            tensor = handle.get_tensor(key)
            target = model_state[key]
            if tuple(tensor.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"Shape mismatch for {key}: checkpoint {tuple(tensor.shape)} "
                    f"!= model {tuple(target.shape)}"
                )
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            missing.discard(key)
            del tensor

    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)[:20]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:20]}")
        raise RuntimeError("SCAIL-2 checkpoint keys do not match model: " + "; ".join(details))


class SCAIL2Pipeline:
'''


LOAD_TARGET = '''        logging.info(f"Creating WanSCAILModel from {scail_safetensors_path}")
        self.model = SCAIL2Model.from_config(scail_config_path)
        state_dict = load_file(scail_safetensors_path)
        self.model.load_state_dict(state_dict)
'''

LOAD_REPLACEMENT = '''        logging.info(f"Creating WanSCAILModel from {scail_safetensors_path}")
        # SCAIL2_RUNPOD_LOW_MEMORY_MODEL_LOADING:
        # Build the model directly in the configured inference dtype, then stream
        # safetensors into it one tensor at a time. This avoids holding the full
        # fp8/fp16 state_dict plus fp32 model weights in CPU RAM at once.
        self.model = SCAIL2Model.from_config(scail_config_path).to(dtype=self.param_dtype)
        _load_safetensors_into_model(self.model, scail_safetensors_path)
'''


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/opt/SCAIL-2")
    scail_path = repo / "wan" / "scail.py"
    if not scail_path.is_file():
        raise FileNotFoundError(f"Missing SCAIL-2 pipeline: {scail_path}")

    current = scail_path.read_text(encoding="utf-8")
    if PATCH_MARKER in current:
        print(f"SCAIL-2 low-memory model-loading patch already applied: {scail_path}")
        return

    patched = current
    for old, new in [
        (IMPORT_TARGET, IMPORT_REPLACEMENT),
        (HELPER_TARGET, HELPER_REPLACEMENT),
        (LOAD_TARGET, LOAD_REPLACEMENT),
    ]:
        if old not in patched:
            raise RuntimeError(f"Unexpected scail.py contents; missing patch target: {old[:80]!r}")
        patched = patched.replace(old, new, 1)

    scail_path.write_text(patched, encoding="utf-8")
    print(f"Applied SCAIL-2 low-memory model-loading patch: {scail_path}")


if __name__ == "__main__":
    main()
