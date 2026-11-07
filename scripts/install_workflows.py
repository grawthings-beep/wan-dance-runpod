#!/usr/bin/env python3
"""Install bundled graphs, backing up a differing existing file before replacing it."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid


def install(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    for graph in sorted(source.glob("*.json")):
        target = destination / graph.name
        if target.exists():
            if target.read_bytes() == graph.read_bytes():
                continue
            backup = destination.parent / "workflow-backups" / stamp / graph.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            print(f"Saved existing workflow: {backup}", flush=True)
        shutil.copy2(graph, target)
        print(f"Installed workflow: {target}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True, action="append")
    args = parser.parse_args()
    for destination in args.destination:
        install(args.source, destination)


if __name__ == "__main__":
    main()
