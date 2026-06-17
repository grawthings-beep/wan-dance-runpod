#!/usr/bin/env python3
import sys
from pathlib import Path


PATCH_MARKER = "SCAIL2_RUNPOD_INFINITY_WINDOWING"


OLD_SEGMENT_BLOCK = '''        def build_segments(total_frames):
            if total_frames <= segment_len:
                keep = ((total_frames - 1) // self.vae_stride[0]) * self.vae_stride[0] + 1
                return [(0, keep)]
            segments = []
            start = 0
            stride = segment_len - segment_overlap
            while start < total_frames:
                end = start + segment_len
                if end > total_frames:
                    break
                segments.append((start, end))
                start += stride
            return segments

        segments = build_segments(num_frames)
        if len(segments) == 0:
            raise ValueError(
                f"No valid segment was produced for {num_frames} frames. "
                f"Use a longer driving video or reduce segment_len.")
        if len(segments) > 1:
'''


NEW_SEGMENT_BLOCK = '''        # SCAIL2_RUNPOD_INFINITY_WINDOWING:
        # Keep sampling fixed-size windows until the stitched output covers the
        # driving video, then trim the final overshoot. This mirrors the
        # Scail2-infinity auto-window behavior and avoids dropping short tails.
        def slice_with_tail_pad(tensor, start, end, dim):
            wanted = end - start
            total = tensor.shape[dim]
            if wanted <= 0:
                raise ValueError("Segment end must be greater than segment start.")
            if total <= 0:
                raise ValueError("Cannot slice an empty driving tensor.")

            pieces = []
            real_start = min(start, total)
            real_end = min(end, total)
            if real_start < real_end:
                pieces.append(tensor.narrow(dim, real_start, real_end - real_start))

            current = sum(piece.shape[dim] for piece in pieces)
            pad = wanted - current
            if pad > 0:
                repeat_shape = [1] * tensor.dim()
                repeat_shape[dim] = pad
                pieces.append(tensor.narrow(dim, total - 1, 1).repeat(*repeat_shape))

            return torch.cat(pieces, dim=dim)

        def build_segments(total_frames):
            if total_frames <= 0:
                raise ValueError("Driving video must contain at least one frame.")
            segments = []
            start = 0
            stride = segment_len - segment_overlap
            produced_frames = 0
            while produced_frames < total_frames:
                segments.append((start, start + segment_len))
                produced_frames = segment_len + (len(segments) - 1) * stride
                start += stride
            return segments

        segments = build_segments(num_frames)
        if len(segments) > 1:
'''


REPLACEMENTS = [
    (
        OLD_SEGMENT_BLOCK,
        NEW_SEGMENT_BLOCK,
    ),
    (
        "                pose_segment = pose_video[seg_start:seg_end]",
        "                pose_segment = slice_with_tail_pad(pose_video, seg_start, seg_end, 0)",
    ),
    (
        "                driving_mask_segment = driving_mask_video[:, seg_start:seg_end]",
        "                driving_mask_segment = slice_with_tail_pad(driving_mask_video, seg_start, seg_end, 1)",
    ),
    (
        '''        if self.rank == 0:
            return torch.cat(output_segments, dim=1).to(self.device)
        return None
''',
        '''        if self.rank == 0:
            output_video = torch.cat(output_segments, dim=1)
            if output_video.shape[1] > num_frames:
                logging.info(
                    f"Trimming stitched output from {output_video.shape[1]} to {num_frames} frames.")
                output_video = output_video[:, :num_frames].contiguous()
            return output_video.to(self.device)
        return None
''',
    ),
]


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/opt/SCAIL-2")
    scail_path = repo / "wan" / "scail.py"
    if not scail_path.is_file():
        raise FileNotFoundError(f"Missing SCAIL-2 pipeline: {scail_path}")

    current = scail_path.read_text(encoding="utf-8")
    if PATCH_MARKER in current:
        print(f"SCAIL-2 infinity windowing patch already applied: {scail_path}")
        return

    patched = current
    for old, new in REPLACEMENTS:
        if old not in patched:
            raise RuntimeError(f"Unexpected scail.py contents; missing patch target: {old[:80]!r}")
        patched = patched.replace(old, new, 1)

    scail_path.write_text(patched, encoding="utf-8")
    print(f"Applied SCAIL-2 infinity windowing patch: {scail_path}")


if __name__ == "__main__":
    main()
