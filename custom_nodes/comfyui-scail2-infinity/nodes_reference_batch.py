"""Pair one character's reference views with their SCAIL identity masks."""

import torch
import torch.nn.functional as F
from comfy_api.latest import io


def append_reference_view(reference_images, reference_masks, extra_image, extra_mask):
    for label, value in (("reference images", reference_images),
                         ("reference masks", reference_masks),
                         ("extra image", extra_image), ("extra mask", extra_mask)):
        if value.ndim != 4 or value.shape[-1] < 3 or value.shape[0] < 1:
            raise ValueError(f"Invalid {label}: expected nonempty BHWC RGB images.")
    if reference_images.shape[:3] != reference_masks.shape[:3]:
        raise ValueError("Each reference image needs a matching mask at the same size.")
    if extra_image.shape[0] != 1 or extra_mask.shape[0] != 1:
        raise ValueError("Add one extra reference view at a time.")
    if extra_image.shape[:3] != extra_mask.shape[:3]:
        raise ValueError("The extra reference and its mask must have the same dimensions.")

    # These workflows select object 0 in both SAM tracks: blue is one identity.
    # Refuse missing/wrong masks instead of silently binding a different person.
    for label, mask in (("primary", reference_masks), ("extra", extra_mask)):
        rgb = mask[..., :3]
        blue = (rgb[..., 0] < 0.1) & (rgb[..., 1] < 0.1) & (rgb[..., 2] > 0.9)
        gray = (rgb.amax(dim=-1) - rgb.amin(dim=-1)) < 0.1
        if not blue.flatten(1).any(dim=1).all().item() or not (blue | gray).all().item():
            raise ValueError(f"The {label} mask must contain one blue character. "
                             "Check the SAM subject description and select object 0.")

    target_h, target_w = reference_images.shape[1:3]
    source_h, source_w = extra_image.shape[1:3]
    scale = min(target_h / source_h, target_w / source_w)
    height = max(1, min(target_h, round(source_h * scale)))
    width = max(1, min(target_w, round(source_w * scale)))
    image = extra_image[..., :3].to(reference_images).movedim(-1, 1)
    mask = extra_mask[..., :3].to(reference_masks).movedim(-1, 1)
    image = F.interpolate(image, size=(height, width), mode="bicubic", align_corners=False).clamp(0, 1)
    mask = F.interpolate(mask, size=(height, width), mode="nearest-exact")
    # Additional views supply character details, never their unrelated background.
    blue = (mask[:, 0:1] < 0.1) & (mask[:, 1:2] < 0.1) & (mask[:, 2:3] > 0.9)
    mask = torch.cat((torch.zeros_like(blue), torch.zeros_like(blue), blue), dim=1).to(mask.dtype)
    left, top = (target_w - width) // 2, (target_h - height) // 2
    padding = (left, target_w - width - left, top, target_h - height - top)
    image = F.pad(image, padding).movedim(1, -1)
    mask = F.pad(mask, padding).movedim(1, -1)
    return (torch.cat((reference_images[..., :3], image), dim=0),
            torch.cat((reference_masks[..., :3], mask), dim=0))


class WanSCAILReferenceBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanSCAILReferenceBatch",
            display_name="SCAIL-2 Add Reference View (one character)",
            category="video/scail2",
            description="Append a face/back/detail view of the SAME character. "
                        "Keeps the main reference first; fits extra views without cropping. "
                        "Experimental: compare against the single-reference workflow.",
            inputs=[io.Image.Input("reference_images"), io.Image.Input("reference_masks"),
                    io.Image.Input("extra_image"), io.Image.Input("extra_mask")],
            outputs=[io.Image.Output("reference_images"), io.Image.Output("reference_masks")],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, reference_images, reference_masks, extra_image, extra_mask):
        return io.NodeOutput(*append_reference_view(reference_images, reference_masks, extra_image, extra_mask))
