#!/usr/bin/env python3
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "wan21_scail_pose_dance.json"
MANIFEST = ROOT / "config" / "scail-models.json"
NODES = ROOT / "config" / "custom-nodes.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    required_files = [
        ROOT / "Dockerfile",
        ROOT / "scripts" / "start.sh",
        ROOT / "scripts" / "download_models.py",
        ROOT / "README.md",
        ROOT / "RUNPOD_STEPS.md",
        WORKFLOW,
        MANIFEST,
        NODES,
    ]
    for path in required_files:
        require(path.is_file(), f"Missing required file: {path.relative_to(ROOT)}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(len(manifest.get("models", [])) == 8, "Expected exactly 8 managed model files")
    paths = set()
    for model in manifest["models"]:
        require(model["url"].startswith("https://"), f"Non-HTTPS URL: {model['name']}")
        require(model["path"].startswith("models/"), f"Model path must be under models/: {model['name']}")
        require(model["path"] not in paths, f"Duplicate model path: {model['path']}")
        require(int(model.get("min_bytes", 0)) > 0, f"Missing min_bytes: {model['name']}")
        paths.add(model["path"])

    node_config = json.loads(NODES.read_text(encoding="utf-8"))
    require(len(node_config.get("nodes", [])) == 5, "Expected exactly 5 pinned custom nodes")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    for node in node_config["nodes"]:
        require(re.fullmatch(r"[0-9a-f]{40}", node["commit"]) is not None, f"Bad commit: {node['name']}")
        require(node["repository"] in dockerfile, f"Dockerfile is missing {node['repository']}")
        require(node["commit"] in dockerfile, f"Dockerfile is missing commit for {node['name']}")

    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    node_types = {node.get("type") for node in workflow.get("nodes", [])}
    required_types = {
        "WanVideoModelLoader",
        "WanVideoSamplerv2",
        "RenderNLFPoses",
        "PoseDetectionVitPoseToDWPose",
        "OnnxDetectionModelLoader",
        "VHS_LoadVideo",
        "VHS_VideoCombine",
    }
    require(required_types <= node_types, f"Workflow is missing node types: {sorted(required_types - node_types)}")

    workflow_text = json.dumps(workflow, ensure_ascii=False)
    required_names = [
        "Wan21-14B-SCAIL-preview_fp8_e4m3fn_scaled_KJ.safetensors",
        "Wan2_1_VAE_bf16.safetensors",
        "umt5-xxl-enc-bf16.safetensors",
        "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "clip_vision_h.safetensors",
        "vitpose-l-wholebody.onnx",
        "yolov10m.onnx",
    ]
    for name in required_names:
        require(name in workflow_text, f"Workflow does not reference {name}")
    require("N:\\\\" not in workflow_text, "Workflow contains a local Windows path")

    print("Repository validation passed.")
    print(f"Models managed: {len(manifest['models'])}")
    print(f"Pinned custom nodes: {len(node_config['nodes'])}")
    print(f"Workflow nodes: {len(workflow['nodes'])}")


if __name__ == "__main__":
    main()

