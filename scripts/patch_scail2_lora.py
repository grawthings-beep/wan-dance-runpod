#!/usr/bin/env python3
import sys
from pathlib import Path


PATCH_MARKER = "SCAIL2_RUNPOD_INPLACE_LORA_FUSION"


PATCHED_LORA = r'''import torch


def _strip_diffusion_prefix(key: str) -> str:
    return key[len("diffusion_model."):] if key.startswith("diffusion_model.") else key


def _add_delta_in_place(model_state, key, delta, alpha: float):
    key = _strip_diffusion_prefix(key)
    candidates = [key]
    if not key.endswith((".weight", ".bias")):
        candidates.append(key + ".weight")

    for candidate in candidates:
        if candidate in model_state:
            target = model_state[candidate]
            target.add_(delta.to(device=target.device, dtype=target.dtype), alpha=alpha)
            return True
    return False


def fuse_lora_with_diff_b(
    model: torch.nn.Module,
    lora_state_dict: dict[str, torch.Tensor],
    alpha: float = 1.0,
):
    # SCAIL2_RUNPOD_INPLACE_LORA_FUSION:
    # Apply LoRA and direct .diff tensors in-place. The upstream implementation
    # replaced entries in a full state_dict and reloaded it, which can double CPU
    # RAM during fast-LoRA fusion on a 14B model.
    model_state = model.state_dict()
    lora_keys = [k for k in lora_state_dict.keys() if k.endswith(".lora_down.weight")]

    with torch.no_grad():
        for lora_key in lora_keys:
            prefix = lora_key[:-len(".lora_down.weight")]

            lora_down_key = lora_key
            lora_up_key = prefix + ".lora_up.weight"
            lora_diff_b_key = prefix + ".diff_b"

            if lora_up_key not in lora_state_dict:
                print(f"[Warning] {lora_up_key} not in LoRA model")
                continue

            weight_key = _strip_diffusion_prefix(prefix + ".weight")
            bias_key = _strip_diffusion_prefix(prefix + ".bias")
            if weight_key not in model_state:
                print(f"[Skip] {weight_key} not in model")
                continue

            W = model_state[weight_key]
            W_down = lora_state_dict[lora_down_key]
            W_up = lora_state_dict[lora_up_key]
            delta_W = torch.matmul(W_up, W_down)
            W.add_(delta_W.to(W.device, W.dtype), alpha=alpha)
            del delta_W

            if bias_key in model_state and lora_diff_b_key in lora_state_dict:
                bias = model_state[bias_key]
                bias.add_(lora_state_dict[lora_diff_b_key].to(bias.device, bias.dtype), alpha=alpha)

        diff_keys = [k for k in lora_state_dict.keys() if k.endswith(".diff")]
        for diff_key in diff_keys:
            base_key = diff_key[:-len(".diff")]
            if not _add_delta_in_place(model_state, base_key, lora_state_dict[diff_key], alpha):
                print(f"[Skip] {_strip_diffusion_prefix(base_key)}(.weight) not in model")
'''


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/opt/SCAIL-2")
    lora_path = repo / "wan" / "utils" / "lora.py"
    if not lora_path.is_file():
        raise FileNotFoundError(f"Missing SCAIL-2 LoRA helper: {lora_path}")

    current = lora_path.read_text(encoding="utf-8")
    if PATCH_MARKER in current:
        print(f"SCAIL-2 in-place LoRA patch already applied: {lora_path}")
        return
    if "def fuse_lora_with_diff_b" not in current:
        raise RuntimeError("Unexpected lora.py contents; refusing to patch.")

    lora_path.write_text(PATCHED_LORA, encoding="utf-8")
    print(f"Applied SCAIL-2 in-place LoRA patch: {lora_path}")


if __name__ == "__main__":
    main()
