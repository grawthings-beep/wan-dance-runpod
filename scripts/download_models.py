#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
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
    unresolved = "{{" in token or "}}" in token or "${" in token
    if token and not unresolved and "huggingface.co" in entry["url"]:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def resolve_download_url(url, headers, timeout=90):
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.geturl()


def download_with_aria2(url, output, headers, connections, splits):
    temp = output.with_suffix(output.suffix + ".part")
    final_url = resolve_download_url(url, headers)
    command = [
        "aria2c",
        "-x",
        str(connections),
        "-s",
        str(splits),
        "-k",
        "1M",
        "--min-split-size=1M",
        "--continue=true",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--file-allocation=none",
        "--max-tries=8",
        "--retry-wait=3",
        "--connect-timeout=30",
        "--timeout=120",
        "--summary-interval=10",
        "--console-log-level=warn",
        "--user-agent",
        headers["User-Agent"],
        "-d",
        str(temp.parent),
        "-o",
        temp.name,
    ]
    if final_url == url:
        for name, value in headers.items():
            if name.lower() != "user-agent":
                command.extend(["--header", f"{name}: {value}"])
    command.append(final_url)
    subprocess.run(command, check=True)
    temp.replace(output)


def download_with_curl(url, output, headers):
    temp = output.with_suffix(output.suffix + ".part")
    command = [
        "curl",
        "-fL",
        "--retry",
        "8",
        "--retry-delay",
        "3",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
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
    request_headers = dict(headers)
    mode = "wb"
    if temp.exists() and temp.stat().st_size:
        request_headers["Range"] = f"bytes={temp.stat().st_size}-"
        mode = "ab"

    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        if mode == "ab" and response.status != 206:
            mode = "wb"
        with temp.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    temp.replace(output)


def validation_error(output, entry):
    if not output.exists():
        return "file is missing"

    size = output.stat().st_size
    expected_size = int(entry.get("size_bytes", 0))
    if expected_size and size != expected_size:
        return f"size is {size} bytes, expected {expected_size}"

    minimum_size = int(entry.get("min_bytes", 1))
    if size < minimum_size:
        return f"size is {size} bytes, minimum is {minimum_size}"

    expected_sha = entry.get("sha256", "").lower()
    if expected_sha:
        actual_sha = sha256_file(output)
        if actual_sha != expected_sha:
            return f"sha256 is {actual_sha}, expected {expected_sha}"
    return None


def download(entry, root, use_aria2, connections, splits):
    output = root / entry["path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".part")

    error = validation_error(output, entry)
    if error is None:
        print(f"SKIP existing: {entry['name']}")
        return
    if output.exists():
        print(f"INVALID existing: {entry['name']} ({error})", file=sys.stderr)
        output.unlink()

    temp_error = validation_error(temp, entry)
    if temp_error is None:
        temp.replace(output)
        print(f"RECOVER completed partial: {entry['name']}")
        return
    expected_size = int(entry.get("size_bytes", 0))
    if temp.exists() and expected_size and temp.stat().st_size > expected_size:
        print(f"INVALID partial: {entry['name']} ({temp_error})", file=sys.stderr)
        temp.unlink()

    print(f"DOWNLOAD: {entry['name']} -> {output}")
    headers = headers_for(entry)
    if use_aria2 and shutil.which("aria2c"):
        download_with_aria2(entry["url"], output, headers, connections, splits)
    elif shutil.which("curl"):
        download_with_curl(entry["url"], output, headers)
    else:
        download_with_urllib(entry["url"], output, headers)

    error = validation_error(output, entry)
    if error is not None:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded file failed validation: {error}")
    print(f"READY: {entry['name']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--no-aria2", action="store_true")
    parser.add_argument(
        "--connections",
        type=int,
        default=int(os.environ.get("ARIA2_CONNECTIONS", "16")),
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=int(os.environ.get("ARIA2_SPLITS", "16")),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("DOWNLOAD_JOBS", "4")),
    )
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    root = Path(args.root)
    entries = [entry for entry in manifest["models"] if entry.get("enabled", True)]
    jobs = max(1, min(args.jobs, len(entries)))

    if jobs == 1:
        for entry in entries:
            download(
                entry,
                root,
                not args.no_aria2,
                max(1, args.connections),
                max(1, args.splits),
            )
        return

    print(
        f"Downloading {len(entries)} models with {jobs} parallel jobs "
        f"(up to {args.connections} aria2 connections per file)"
    )
    errors = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                download,
                entry,
                root,
                not args.no_aria2,
                max(1, args.connections),
                max(1, args.splits),
            ): entry["name"]
            for entry in entries
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors.append(name)
                print(f"ERROR: {name}: {exc}", file=sys.stderr)

    if errors:
        raise SystemExit(f"failed to download required models: {', '.join(errors)}")


if __name__ == "__main__":
    main()
