#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def utc_now():
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp_path.replace(path)


def update_status(status_path, **updates):
    status = read_json(status_path)
    status.update(updates)
    write_json(status_path, status)
    return status


def run_job(job_dir):
    job_dir = Path(job_dir)
    command_path = job_dir / "command.json"
    status_path = job_dir / "status.json"
    payload = read_json(command_path)
    log_path = Path(payload["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    update_status(
        status_path,
        status="running",
        message="Generation process is running.",
        started_at=utc_now(),
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.update(payload.get("env", {}))

    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"[job-worker] started {utc_now()}\n")
        log.write("+ " + " ".join(str(part) for part in payload["command"]) + "\n")
        process = subprocess.run(
            payload["command"],
            cwd=payload.get("cwd") or None,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"[job-worker] finished {utc_now()} returncode={process.returncode}\n")

    output = payload.get("output")
    output_exists = bool(output and Path(output).is_file())
    status = "succeeded" if process.returncode == 0 and output_exists else "failed"
    message = "Output video is ready." if status == "succeeded" else "Generation failed. See generation.log."
    update_status(
        status_path,
        status=status,
        message=message,
        finished_at=utc_now(),
        returncode=process.returncode,
        output_exists=output_exists,
    )
    return process.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()

    try:
        return run_job(args.job_dir)
    except Exception:
        job_dir = Path(args.job_dir)
        status_path = job_dir / "status.json"
        log_path = job_dir / "generation.log"
        message = traceback.format_exc()
        with log_path.open("a", encoding="utf-8") as log:
            log.write("[job-worker] unhandled error\n")
            log.write(message)
        if status_path.is_file():
            update_status(
                status_path,
                status="failed",
                message="Worker crashed before generation completed.",
                finished_at=utc_now(),
                returncode=-1,
                error=message[-4000:],
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
