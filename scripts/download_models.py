#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def headers_for(entry):
    headers = {"User-Agent": "wan-dance-runpod"}
    token = os.environ.get("HF_TOKEN", "").strip()
    if token and "huggingface.co" in entry["url"]:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def download_with_curl(url, output, headers):
    temp = output.with_suffix(output.suffix + ".part")
    command = [
        "curl",
        "-fL",
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "--retry-all-errors",
        "-C",
        "-",
    ]
    for name, value in headers.items():
        command.extend(["-H", f"{name}: {value}"])
    command.extend(["-o", str(temp), url])
    subprocess.run(command, check=True)
    temp.replace(output)


def download_with_urllib(url, output, headers):
    temp = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    temp.replace(output)


def is_valid_existing(output, entry):
    if not output.exists():
        return False
    if output.stat().st_size < int(entry.get("min_bytes", 1)):
        return False
    expected = entry.get("sha256", "").lower()
    return not expected or sha256_file(output) == expected


def download(entry, root):
    output = root / entry["path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    if is_valid_existing(output, entry):
        print(f"SKIP existing: {entry['name']}")
        return

    if output.exists():
        output.unlink()
    print(f"DOWNLOAD: {entry['name']} -> {output}")
    headers = headers_for(entry)
    if shutil.which("curl"):
        download_with_curl(entry["url"], output, headers)
    else:
        download_with_urllib(entry["url"], output, headers)

    if not is_valid_existing(output, entry):
        output.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file failed validation: {entry['name']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    root = Path(args.root)
    for entry in manifest["models"]:
        download(entry, root)


if __name__ == "__main__":
    main()

