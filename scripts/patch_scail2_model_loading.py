#!/usr/bin/env python3
import sys
from pathlib import Path


PATCH_MARKER = "SCAIL2_RUNPOD_GPU_AWARE_FP8_V2_MODEL_LOADING"
OLD_PATCH_MARKERS = (
    "SCAIL2_RUNPOD_GPU_AWARE_MODEL_LOADING",
    "SCAIL2_RUNPOD_GPU_AWARE_FP8_MODEL_LOADING",
)


IMPORT_TARGET = "from safetensors.torch import load_file\n"
IMPORT_REPLACEMENT = "from safetensors import safe_open\nfrom safetensors.torch import load_file\n"


HELPER_TARGET = "class SCAIL2Pipeline:\n"
HELPER_REPLACEMENT = '''def _scale_tensor_for_target(tensor, scale, target, key):
    value = tensor.to(device=target.device, dtype=target.dtype)
    scale = scale.to(device=target.device, dtype=target.dtype)
    if scale.ndim == 0 or scale.numel() == 1:
        return value.mul_(scale.reshape(()))

    if scale.ndim == 3:
        if value.ndim != 2 or scale.shape[0] != value.shape[0] or scale.shape[2] != 1:
            raise RuntimeError(
                f"Unsupported block-wise fp8 scale shape for {key}: "
                f"weight={tuple(value.shape)} scale={tuple(scale.shape)}"
            )
        reshaped = value.reshape(scale.shape[0], scale.shape[1], -1)
        if reshaped.shape[:2] != scale.shape[:2]:
            raise RuntimeError(
                f"Cannot apply block-wise fp8 scale for {key}: "
                f"weight={tuple(value.shape)} scale={tuple(scale.shape)}"
            )
        return reshaped.mul_(scale).reshape_as(value)

    raise RuntimeError(
        f"Unsupported fp8 scale shape for {key}: weight={tuple(value.shape)} "
        f"scale={tuple(scale.shape)}"
    )


def _load_safetensors_into_model(model, filename):
    model_state = model.state_dict()
    missing = set(model_state.keys())
    unexpected = []
    scaled_weights = 0

    with safe_open(filename, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        scale_keys = {
            key[:-len(".scale_weight")]: key
            for key in keys
            if key.endswith(".scale_weight")
        }

        for key in keys:
            if key == "scaled_fp8":
                continue

            if key.endswith(".scale_weight"):
                base_key = key[:-len(".scale_weight")] + ".weight"
                if base_key in model_state:
                    continue
                unexpected.append(key)
                continue

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

            scale_key = None
            if key.endswith(".weight"):
                scale_key = scale_keys.get(key[:-len(".weight")])
            if scale_key is not None:
                value = _scale_tensor_for_target(tensor, handle.get_tensor(scale_key), target, key)
                scaled_weights += 1
            else:
                value = tensor.to(device=target.device, dtype=target.dtype)
            target.copy_(value)
            missing.discard(key)
            del value
            del tensor

    if scaled_weights:
        logging.info(f"Applied {scaled_weights} fp8 scale_weight tensors while loading SCAIL-2.")

    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)[:20]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:20]}")
        raise RuntimeError("SCAIL-2 checkpoint keys do not match model: " + "; ".join(details))


def _build_scail_model_from_config(config_path, device, dtype):
    previous_dtype = torch.get_default_dtype()
    dtype_changed = False
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        torch.set_default_dtype(dtype)
        dtype_changed = True

    try:
        with torch.device(device):
            model = SCAIL2Model.from_config(config_path)
    finally:
        if dtype_changed:
            torch.set_default_dtype(previous_dtype)

    if not dtype_changed:
        model = model.to(dtype=dtype)
    return model


class SCAIL2Pipeline:
'''


LOAD_TARGET = '''        logging.info(f"Creating WanSCAILModel from {scail_safetensors_path}")
        self.model = SCAIL2Model.from_config(scail_config_path)
        state_dict = load_file(scail_safetensors_path)
        self.model.load_state_dict(state_dict)
'''

LOAD_REPLACEMENT = '''        logging.info(f"Creating WanSCAILModel from {scail_safetensors_path}")
        # SCAIL2_RUNPOD_GPU_AWARE_FP8_V2_MODEL_LOADING:
        # Build the model directly in the configured inference dtype, then stream
        # safetensors into it one tensor at a time. When CPU offload is disabled,
        # load directly onto the GPU to avoid keeping the 14B transformer in
        # RunPod system RAM during startup.
        model_device = torch.device("cpu") if init_on_cpu else self.device
        logging.info(f"Loading WanSCAILModel weights on {model_device} (init_on_cpu={init_on_cpu}).")
        self.model = _build_scail_model_from_config(scail_config_path, model_device, self.param_dtype)
        _load_safetensors_into_model(self.model, scail_safetensors_path)
'''


GENERATE_TARGET = '''        t5_cpu=args.t5_cpu,
        lora_path=args.lora_path,
'''

GENERATE_REPLACEMENT = '''        t5_cpu=args.t5_cpu,
        init_on_cpu=args.offload_model,
        lora_path=args.lora_path,
'''


LOAD_BLOCK_START = '''        logging.info(f"Creating WanSCAILModel from {scail_safetensors_path}")
'''

LOAD_BLOCK_END = '''        if self.lora_path is not None:
'''


def replace_existing_helper(text):
    start = text.find("def _scale_tensor_for_target(tensor, scale, target, key):\n")
    if start == -1:
        start = text.find("def _load_safetensors_into_model(model, filename):\n")
    if start == -1:
        raise RuntimeError("Unexpected patched scail.py contents; missing helper start.")
    end = text.find(HELPER_TARGET, start)
    if end == -1:
        raise RuntimeError("Unexpected patched scail.py contents; missing SCAIL2Pipeline class.")
    return text[:start] + HELPER_REPLACEMENT + text[end + len(HELPER_TARGET):]


def replace_model_load_block(text):
    start = text.find(LOAD_BLOCK_START)
    if start == -1:
        raise RuntimeError("Unexpected scail.py contents; missing model load block start.")
    end = text.find(LOAD_BLOCK_END, start)
    if end == -1:
        raise RuntimeError("Unexpected scail.py contents; missing model load block end.")
    return text[:start] + LOAD_REPLACEMENT + text[end:]


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/opt/SCAIL-2")
    scail_path = repo / "wan" / "scail.py"
    if not scail_path.is_file():
        raise FileNotFoundError(f"Missing SCAIL-2 pipeline: {scail_path}")

    current = scail_path.read_text(encoding="utf-8")
    scail_changed = False
    if PATCH_MARKER not in current:
        patched = current
        if "from safetensors import safe_open\n" not in patched:
            if IMPORT_TARGET not in patched:
                raise RuntimeError(f"Unexpected scail.py contents; missing patch target: {IMPORT_TARGET!r}")
            patched = patched.replace(IMPORT_TARGET, IMPORT_REPLACEMENT, 1)

        if any(marker in patched for marker in OLD_PATCH_MARKERS):
            patched = replace_existing_helper(patched)
        else:
            if HELPER_TARGET not in patched:
                raise RuntimeError(f"Unexpected scail.py contents; missing patch target: {HELPER_TARGET!r}")
            patched = patched.replace(HELPER_TARGET, HELPER_REPLACEMENT, 1)

        patched = replace_model_load_block(patched)

        scail_path.write_text(patched, encoding="utf-8")
        scail_changed = True

    generate_path = repo / "generate.py"
    if not generate_path.is_file():
        raise FileNotFoundError(f"Missing SCAIL-2 generate.py: {generate_path}")
    generate = generate_path.read_text(encoding="utf-8")
    generate_changed = False
    if GENERATE_REPLACEMENT not in generate:
        if GENERATE_TARGET not in generate:
            raise RuntimeError("Unexpected generate.py contents; missing SCAIL2Pipeline init target.")
        generate = generate.replace(GENERATE_TARGET, GENERATE_REPLACEMENT, 1)
        generate_path.write_text(generate, encoding="utf-8")
        generate_changed = True

    if scail_changed or generate_changed:
        print(f"Applied SCAIL-2 GPU-aware model-loading patch: {scail_path}")
    else:
        print(f"SCAIL-2 GPU-aware model-loading patch already applied: {scail_path}")


if __name__ == "__main__":
    main()
