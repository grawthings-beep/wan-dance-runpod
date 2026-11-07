#!/usr/bin/env python3
"""Exercise the real pinned ComfyUI imports and reference conditioning on CPU.

No model weights or GPU are used; this does not certify generated video quality.
"""
import asyncio
import importlib
import json
import os
from pathlib import Path
import sys

APP = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("COMFYUI_ROOT", "/opt/ComfyUI"))
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
# Match main.py: load the top-level utils package before nodes prepends the
# comfy/ directory (which also contains a utils.py module) to sys.path.
import utils.install_util
import torch
import nodes


async def main():
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    custom = APP / "custom_nodes/comfyui-scail2-infinity"
    if not await nodes.load_custom_node(str(custom)):
        raise RuntimeError("The bundled SCAIL-2 nodes failed to load")
    batch = nodes.NODE_CLASS_MAPPINGS["WanSCAILReferenceBatch"]
    infinity = nodes.NODE_CLASS_MAPPINGS["WanSCAILInfinity"]
    assert "uni_pc" in importlib.import_module("comfy.samplers").KSampler.SAMPLERS
    for path in (APP / "workflows").glob("*.json"):
        graph = json.loads(path.read_text(encoding="utf-8"))
        for node in graph["nodes"]:
            if node["type"] == "Note" or node["type"].startswith("VHS_"):
                continue  # VHS graph is unchanged; its requirements are installed by Docker.
            cls = nodes.NODE_CLASS_MAPPINGS[node["type"]]
            schema = cls.INPUT_TYPES()
            input_names = set(schema.get("required", {})) | set(schema.get("optional", {}))
            for slot in node.get("inputs", []):
                assert slot["name"] in input_names, (path.name, node["type"], slot["name"])

    # A wide extra view fits inside a portrait frame: no crop, matching geometry.
    primary = torch.full((1, 64, 32, 3), 0.25)
    primary_mask = torch.ones_like(primary)
    primary_mask[:, 8:56, 8:24] = torch.tensor([0.0, 0.0, 1.0])
    extra = torch.ones((1, 16, 64, 3))
    extra_mask = torch.ones_like(extra)
    extra_mask[:, 2:14, 8:56] = torch.tensor([0.0, 0.0, 1.0])
    refs, masks = batch.execute(primary, primary_mask, extra, extra_mask).args
    assert refs.shape == masks.shape == (2, 64, 32, 3)
    assert torch.equal(refs[:1], primary) and torch.equal(masks[:1], primary_mask)
    assert torch.all(refs[1, 28:36] == 1)  # all of the wide image remains visible
    assert torch.all(refs[1, :28] == 0) and torch.all(refs[1, 36:] == 0)
    assert not masks[1, ..., :2].any()  # no white/background evidence in the extra mask
    assert set(masks[1].unique().tolist()) == {0.0, 1.0}
    try:
        batch.execute(primary, primary_mask, extra, torch.ones_like(extra))
    except ValueError:
        pass
    else:
        raise AssertionError("Missing subject mask must be rejected")

    # Call the actual core conditioning with a tiny fake VAE. Verify separate
    # image encodes and matching masks survive through the core interface.
    class TinyVAE:
        def __init__(self):
            self.calls = []

        def encode(self, images):
            self.calls.append(images.shape[0])
            return torch.zeros((1, 16, (images.shape[0] - 1) // 4 + 1,
                                images.shape[1] // 8, images.shape[2] // 8))

    vae = TinyVAE()
    scail = importlib.import_module("comfy_extras.nodes_scail")
    positive = [[torch.zeros(1, 1, 1), {}]]
    result = scail.WanSCAILToVideo.execute(
        positive=positive, negative=positive, vae=vae, width=32, height=64,
        length=5, batch_size=1, pose_strength=1.0, pose_start=0.0, pose_end=1.0,
        video_frame_offset=0, previous_frame_count=5, reference_image=refs,
        reference_image_mask=masks,
    ).args
    assert vae.calls == [1, 1], vae.calls
    metadata = result[0][0][1]
    assert len(metadata["reference_latents"]) == 2
    assert metadata["ref_mask_28ch"].shape[1] == 4
    assert "reference_image" in infinity.INPUT_TYPES()["optional"]
    print("CPU container smoke passed: real ComfyUI schemas, reference geometry, masks and conditioning.")
    print("GPU video generation and quality validation: NOT RUN")


if __name__ == "__main__":
    asyncio.run(main())
