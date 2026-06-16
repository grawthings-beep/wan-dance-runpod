#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "scail2-runtime.json"


def expand_path(value):
    return Path(os.path.expandvars(value)).expanduser()


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_is_large_enough(path, min_bytes):
    return path.is_file() and path.stat().st_size >= int(min_bytes)


def require_file(path, min_bytes, label):
    if not file_is_large_enough(path, min_bytes):
        size = path.stat().st_size if path.exists() else 0
        raise RuntimeError(
            f"{label} is missing or too small: {path} ({size} bytes)"
        )


def snapshot_download(repo_id, revision, local_dir, allow_patterns):
    from huggingface_hub import snapshot_download as hf_snapshot_download

    token = os.environ.get("HF_TOKEN", "").strip() or None
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id}@{revision} -> {local_dir}", flush=True)
    hf_snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        token=token,
    )


def ensure_scail2(config, skip_download=False, force_convert=False):
    scail = config["scail2"]
    checkpoint_dir = expand_path(
        os.environ.get("SCAIL2_CKPT_DIR", scail["checkpoint_dir"])
    )
    converted_path = expand_path(
        os.environ.get("SCAIL2_SAFETENSORS", scail["converted_path"])
    )

    if not skip_download:
        snapshot_download(
            scail["model_repository"],
            os.environ.get("SCAIL2_MODEL_REVISION", scail["model_revision"]),
            checkpoint_dir,
            scail["allow_patterns"],
        )

    for item in scail["required_files"]:
        require_file(
            checkpoint_dir / item["path"],
            item["min_bytes"],
            f"SCAIL-2 file {item['path']}",
        )

    if (
        not force_convert
        and file_is_large_enough(converted_path, scail["converted_min_bytes"])
    ):
        print(f"SKIP converted checkpoint: {converted_path}", flush=True)
        return converted_path

    converted_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = converted_path.with_suffix(converted_path.suffix + ".part")
    temp_path.unlink(missing_ok=True)

    scail2_repo = expand_path(os.environ.get("SCAIL2_REPO", "/opt/SCAIL-2"))
    command = [
        sys.executable,
        str(scail2_repo / "convert.py"),
        "--scail-dir",
        str(checkpoint_dir),
        "--save-path",
        str(temp_path),
    ]
    print("Converting SCAIL-2 FSDP checkpoint to safetensors.", flush=True)
    subprocess.run(command, cwd=str(scail2_repo), check=True)
    require_file(temp_path, scail["converted_min_bytes"], "converted SCAIL-2 checkpoint")
    temp_path.replace(converted_path)
    print(f"Converted checkpoint ready: {converted_path}", flush=True)
    return converted_path


def ensure_sam3(config, required=False):
    sam3 = config["sam3"]
    model_path = expand_path(os.environ.get("SAM3_MODEL", sam3["model_path"]))
    if file_is_large_enough(model_path, sam3["min_bytes"]):
        print(f"SKIP SAM3 checkpoint: {model_path}", flush=True)
        return model_path

    try:
        snapshot_download(
            sam3["model_repository"],
            os.environ.get("SAM3_MODEL_REVISION", sam3["model_revision"]),
            expand_path(sam3["checkpoint_dir"]),
            sam3["allow_patterns"],
        )
        require_file(model_path, sam3["min_bytes"], "SAM3 checkpoint")
        return model_path
    except Exception as exc:
        message = (
            "SAM3 download failed. facebook/sam3 is gated; set HF_TOKEN to an "
            "account that has accepted the model license, or upload masks "
            "manually instead of using auto-mask."
        )
        if required:
            raise RuntimeError(message) from exc
        print(f"WARNING: {message}\n{exc}", file=sys.stderr, flush=True)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--download-sam3", action="store_true")
    parser.add_argument("--require-sam3", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_scail2(
        config,
        skip_download=args.skip_download,
        force_convert=args.force_convert,
    )
    if args.download_sam3 or args.require_sam3:
        ensure_sam3(config, required=args.require_sam3)


if __name__ == "__main__":
    main()
