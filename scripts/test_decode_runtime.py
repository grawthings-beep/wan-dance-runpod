"""Regression checks using the pinned ComfyUI 3D tiler, without model weights."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file
import comfy.sd
import comfy.utils
import comfy.model_management as mm
import folder_paths


def check_decode_runtime(infinity_module):
    class IdentityVideoVAE(comfy.sd.VAE):
        def __init__(self):
            self.first_stage_model = SimpleNamespace(decode=lambda samples: samples)
            self.vae_dtype = torch.float32
            self.device = torch.device("cpu")
            self.output_device = torch.device("cpu")
            self.extra_1d_channel = None
            self.disable_offload = False
            self.patcher = None
            self.memory_used_decode = lambda shape, dtype: 0
            self.upscale_ratio = (1, 1, 1)
            self.upscale_index_formula = None
            self.output_channels = 3
            self.process_output = lambda pixels: pixels

        def temporal_compression_decode(self):
            return 4

        def spacial_compression_decode(self):
            return 8

        def vae_output_dtype(self):
            return torch.float32

    vae = IdentityVideoVAE()
    # Span multiple tiles in time and both spatial dimensions. Only the neural
    # decoder and GPU model loading are replaced; core tile arithmetic is real.
    samples = torch.ones((1, 3, 21, 72, 72))
    with patch.object(mm, "load_models_gpu"):
        try:
            vae.decode_tiled(samples)
        except TypeError as error:
            assert "NoneType" in str(error), error
        else:
            raise AssertionError("Expected the original no-overlap call to reproduce the reported failure")
        images = infinity_module._decode(vae, samples, True)
    assert images.shape == (21, 72, 72, 3), images.shape
    assert torch.isfinite(images).all() and torch.allclose(images, torch.ones_like(images))
    print("Decode regression passed: original failure reproduced; core 3D tiling succeeds with explicit overlaps.")

    with TemporaryDirectory() as directory:
        with patch.object(folder_paths, "get_output_directory", return_value=directory), \
                patch.object(infinity_module, "_decode", side_effect=RuntimeError("decode probe")):
            try:
                infinity_module._decode_and_preserve(vae, samples, True, 123, 0)
            except RuntimeError as error:
                assert str(error) == "decode probe"
            else:
                raise AssertionError("A failed decode must still fail the job")
        files = list(Path(directory).rglob("*.latent"))
        assert len(files) == 1, files
        recovered = load_file(str(files[0]), device="cpu")
        assert "latent_format_version_0" in recovered
        assert torch.equal(recovered["latent_tensor"], samples)
    print("Recovery regression passed: failed decode preserves a readable, unchanged ComfyUI latent.")
