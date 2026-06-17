"""SCAIL-2 Infinity: a single all-in-one node that loops the SCAIL-2 81-frame
chunk -> sample -> decode -> re-anchor process internally to produce arbitrarily
long video, bounded by the driving pose video.

It reuses ComfyUI core verbatim:
  - comfy_extras.nodes_scail.WanSCAILToVideo.execute()  -> builds the per-chunk
    conditioning + empty latent + adjusted frame offset (pure function of inputs).
  - nodes.common_ksampler()                             -> samples each chunk.
  - VAE.decode / VAE.decode_tiled                        -> pixels per chunk.

SCAIL-2 was trained at 81-frame chunks with a 5-frame overlap (76-frame step), so
each chunk after the first is anchored on the last 5 decoded frames of the previous
chunk and we drop those 5 overlap frames when stitching.

Memory: peak VRAM equals a single 81-frame run. The window never widens past
window_length, the model stays resident across chunks, and every decoded chunk /
kept latent is moved to CPU immediately. soft_empty_cache() is called each chunk.
"""

from typing_extensions import override

import torch

import nodes
import comfy.samplers
import comfy.model_management as mm
from comfy_api.latest import ComfyExtension, io

# Reuse the core SCAIL conditioning node's logic verbatim (no duplication).
from comfy_extras.nodes_scail import WanSCAILToVideo
from nodes import common_ksampler


def _decode(vae, latent_samples, tiled: bool):
    """Decode a video latent to (T, H, W, C), mirroring VAEDecode's 5D reshape."""
    images = vae.decode_tiled(latent_samples) if tiled else vae.decode(latent_samples)
    if len(images.shape) == 5:  # (B, T, H, W, C) -> combine batch+time
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


class WanSCAILInfinity(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanSCAILInfinity",
            display_name="SCAIL-2 Infinity (auto window)",
            category="video/scail2",
            description=(
                "All-in-one SCAIL-2 long-video node. Internally loops 81-frame chunks "
                "(with a 5-frame overlap) over the driving pose video, stitching them into "
                "one continuous video. Replaces the WanSCAILToVideo + KSampler + VAEDecode "
                "+ manual chunk-wiring graph."
            ),
            inputs=[
                # --- core conditioning / model / vae ---
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                # --- geometry ---
                io.Int.Input("width", default=512, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=896, min=32, max=nodes.MAX_RESOLUTION, step=32),
                # --- sampler ---
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=6.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                # --- loop / windowing ---
                io.Int.Input("window_length", default=81, min=5, max=nodes.MAX_RESOLUTION, step=4,
                             tooltip="Frames per chunk. SCAIL-2 was trained at 81; changing this is not recommended."),
                io.Int.Input("previous_frame_count", default=5, min=1, max=nodes.MAX_RESOLUTION, step=4,
                             tooltip="Overlap frames anchored from the previous chunk. SCAIL-2 trained at 5."),
                io.Int.Input("max_frames", default=0, min=0, max=nodes.MAX_RESOLUTION, step=1,
                             tooltip="Hard cap on total output frames. 0 = run until the driving pose video is exhausted."),
                io.Boolean.Input("decode_tiled", default=False,
                                 tooltip="Use tiled VAE decode to bound decode VRAM at high resolution."),
                io.Boolean.Input("vary_seed_per_window", default=False,
                                 tooltip="Add the window index to the seed each chunk. Off = same seed every chunk (anchored continuity)."),
                # --- SCAIL-2 passthrough (all optional) ---
                io.Image.Input("pose_video", optional=True,
                               tooltip="Driving pose video. Its length determines how many frames are generated. Downscaled to half resolution internally."),
                io.Image.Input("pose_video_mask", optional=True,
                               tooltip="SCAIL-2 colored per-identity SAM3 mask video, same resolution as pose_video."),
                io.Boolean.Input("replacement_mode", default=False, optional=True,
                                 tooltip="False = Animation Mode (black-bg mask). True = Replacement Mode (white-bg mask)."),
                io.Float.Input("pose_strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("pose_start", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("pose_end", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Image.Input("reference_image", optional=True,
                               tooltip="Reference character image. For multiple references composite them on a single image."),
                io.Image.Input("reference_image_mask", optional=True,
                               tooltip="SCAIL-2 colored reference mask, same resolution as reference_image."),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True,
                                          tooltip="CLIP vision features for conditioning."),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="Full stitched video."),
                io.Latent.Output(display_name="latent", tooltip="Concatenated, overlap-free latent of the full video."),
                io.Int.Output(display_name="total_frames"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, positive, negative, model, vae, width, height, seed, steps, cfg,
                sampler_name, scheduler, denoise, window_length, previous_frame_count,
                max_frames, decode_tiled, vary_seed_per_window,
                pose_strength=1.0, pose_start=0.0, pose_end=1.0, replacement_mode=False,
                pose_video=None, pose_video_mask=None, reference_image=None,
                reference_image_mask=None, clip_vision_output=None) -> io.NodeOutput:

        lat_drop = ((previous_frame_count - 1) // 4) + 1   # overlap in latent frames
        step = max(1, window_length - previous_frame_count)  # new frames produced per chunk after the first
        pose_len = pose_video.shape[0] if pose_video is not None else None

        # How many output frames we ultimately want.
        if pose_len is not None:
            target = pose_len if max_frames <= 0 else min(pose_len, max_frames)
        else:
            target = window_length if max_frames <= 0 else max_frames

        # Expected number of fixed-size windows (chunk 0 covers `window_length`, each
        # subsequent chunk adds `step` new frames).
        if pose_len is None:
            total_windows = 1
        else:
            extra = max(0, target - window_length)
            total_windows = 1 + (extra + step - 1) // step

        print(f"[SCAIL2-Infinity] target={target} frames | window={window_length} "
              f"overlap={previous_frame_count} step={step} -> {total_windows} window(s)", flush=True)

        stitched_imgs = []   # list of (T, H, W, C) on CPU
        stitched_lat = []    # list of (B, C, T_lat, H, W) on CPU
        prev_anchor = None   # last `previous_frame_count` decoded frames of previous chunk (CPU)
        offset = 0           # output-timeline end of the previous chunk == start passed to the node
        window_index = 0

        while offset < target:
            first = window_index == 0
            anchor = 0 if first else previous_frame_count
            eff = max(0, offset - anchor)  # where this chunk begins on the pose/output timeline
            pose_hi = min(eff + window_length, pose_len) if pose_len is not None else eff + window_length

            print(f"[SCAIL2-Infinity] window {window_index + 1}/{total_windows}: "
                  f"output frames [{offset}..{eff + window_length}) | "
                  f"pose frames [{eff}..{pose_hi}) | anchored on {anchor} previous frame(s)", flush=True)

            # --- 1) build per-chunk conditioning + latent + adjusted offset (core logic) ---
            prev_off = offset
            pos, neg, latent, offset = WanSCAILToVideo.execute(
                positive=positive, negative=negative, vae=vae,
                width=width, height=height, length=window_length, batch_size=1,
                pose_strength=pose_strength, pose_start=pose_start, pose_end=pose_end,
                video_frame_offset=offset, previous_frame_count=previous_frame_count,
                replacement_mode=replacement_mode, reference_image=reference_image,
                clip_vision_output=clip_vision_output, pose_video=pose_video,
                pose_video_mask=pose_video_mask, reference_image_mask=reference_image_mask,
                previous_frames=prev_anchor,
            ).args

            # --- 2) sample this chunk ---
            chunk_seed = seed + (window_index if vary_seed_per_window else 0)
            print(f"[SCAIL2-Infinity]   sampling (seed={chunk_seed}, {steps} steps)...", flush=True)
            sampled = common_ksampler(model, chunk_seed, steps, cfg, sampler_name, scheduler,
                                      pos, neg, latent, denoise=denoise)[0]
            chunk_latent = sampled["samples"]

            # --- 3) decode to pixels ---
            imgs = _decode(vae, chunk_latent, decode_tiled)

            # --- 4) stitch: drop the overlap frames on every chunk after the first ---
            new_imgs = imgs[anchor:]
            stitched_imgs.append(new_imgs.cpu())
            stitched_lat.append(chunk_latent[:, :, (0 if first else lat_drop):].cpu())
            total_so_far = sum(t.shape[0] for t in stitched_imgs)
            print(f"[SCAIL2-Infinity]   decoded {imgs.shape[0]} frames, kept {new_imgs.shape[0]} new "
                  f"(stitched total: {total_so_far})", flush=True)

            # --- 5) next anchor (only the tail the next chunk needs) + 6) free VRAM ---
            prev_anchor = imgs[-previous_frame_count:].cpu()
            del sampled, latent, chunk_latent, imgs, new_imgs, pos, neg
            mm.soft_empty_cache()

            window_index += 1
            if pose_len is None:
                break  # no driving video -> single window only
            if offset <= prev_off:  # safety: should never happen with fixed windows
                print("[SCAIL2-Infinity] offset failed to advance; stopping to avoid an infinite loop.", flush=True)
                break

        result_imgs = torch.cat(stitched_imgs, dim=0)
        result_lat = torch.cat(stitched_lat, dim=2)

        # Trim to the exact requested length (the last window usually overshoots).
        if result_imgs.shape[0] > target:
            result_imgs = result_imgs[:target]
            result_lat = result_lat[:, :, :((target - 1) // 4) + 1]

        print(f"[SCAIL2-Infinity] done: {result_imgs.shape[0]} frames over {window_index} window(s).", flush=True)
        return io.NodeOutput(result_imgs, {"samples": result_lat}, result_imgs.shape[0])


class SCAILInfinityExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [WanSCAILInfinity]


async def comfy_entrypoint() -> SCAILInfinityExtension:
    return SCAILInfinityExtension()
