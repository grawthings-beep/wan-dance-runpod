#!/usr/bin/env python3
import json
from pathlib import Path
import re
import py_compile


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "config" / "scail2-runtime.json"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    required_files = [
        ROOT / "Dockerfile",
        ROOT / "scripts" / "start.sh",
        ROOT / "scripts" / "prepare_models.py",
        ROOT / "scripts" / "patch_scail2_attention.py",
        ROOT / "scripts" / "patch_scail2_model_loading.py",
        ROOT / "scripts" / "patch_scail2_lora.py",
        ROOT / "scripts" / "patch_scail2_infinity.py",
        ROOT / "scripts" / "run_scail2.py",
        ROOT / "scripts" / "app.py",
        ROOT / "scripts" / "job_worker.py",
        ROOT / "custom_nodes" / "comfyui-scail2-infinity" / "__init__.py",
        ROOT / "custom_nodes" / "comfyui-scail2-infinity" / "nodes_scail_infinity.py",
        ROOT / "workflows" / "scail2-comfyui-infinity-runpod.json",
        ROOT / "README.md",
        ROOT / "RUNPOD_STEPS.md",
        RUNTIME,
    ]
    for path in required_files:
        require(path.is_file(), f"Missing required file: {path.relative_to(ROOT)}")

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
    scail = runtime["scail2"]
    require(re.fullmatch(r"[0-9a-f]{40}", scail["code_commit"]) is not None, "Bad SCAIL-2 code commit")
    require(re.fullmatch(r"[0-9a-f]{40}", scail["pose_commit"]) is not None, "Bad SCAIL-Pose commit")
    require(re.fullmatch(r"[0-9a-f]{40}", scail["model_revision"]) is not None, "Bad SCAIL-2 model revision")
    require(re.fullmatch(r"[0-9a-f]{40}", scail["scail_weights_revision"]) is not None, "Bad SCAIL-2 weights revision")
    require(scail["code_commit"] in dockerfile, "Dockerfile is missing pinned SCAIL-2 commit")
    require(scail["pose_commit"] in dockerfile, "Dockerfile is missing pinned SCAIL-Pose commit")
    require("COMFYUI_COMMIT=ca3dbe206c2fea84f2af4371ca13e9f2bfeb82e5" in dockerfile, "Dockerfile must pin current ComfyUI")
    require("VIDEO_HELPER_SUITE_COMMIT=4ee72c065db22c9d96c2427954dc69e7b908444b" in dockerfile, "Dockerfile must pin VideoHelperSuite")
    require("ComfyUI-VideoHelperSuite" in dockerfile, "Dockerfile must install VideoHelperSuite")
    require("patch_scail2_attention.py" in dockerfile, "Dockerfile must patch SCAIL-2 attention fallback")
    require("patch_scail2_model_loading.py" in dockerfile, "Dockerfile must patch SCAIL-2 low-memory loading")
    require("patch_scail2_lora.py" in dockerfile, "Dockerfile must patch SCAIL-2 LoRA fusion")
    require("patch_scail2_infinity.py" in dockerfile, "Dockerfile must patch SCAIL-2 infinity windowing")
    require(len(scail["required_files"]) >= 7, "Expected official SCAIL-2 support files")
    require("model/1/fsdp2_rank_0000_checkpoint.pt" not in json.dumps(scail), "FSDP checkpoint should not be downloaded")
    require(scail["scail_path"].endswith("wan2.1_14B_SCAIL_2_fp8_scaled.safetensors"), "Expected fp8 scaled safetensors")
    require(int(scail["scail_min_bytes"]) >= 17600000000, "SCAIL-2 fp8 safetensors min size is too low")
    for item in scail["required_files"]:
        require(int(item["min_bytes"]) > 0, f"Missing min_bytes: {item['path']}")

    sam3 = runtime["sam3"]
    require(sam3["model_repository"] == "facebook/sam3", "SAM3 repo changed unexpectedly")
    require(sam3.get("gated") is True, "SAM3 gated status must be documented")

    lightx2v = runtime["lightx2v_lora"]
    require(
        lightx2v["model_repository"] == "lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v",
        "Lightx2v LoRA repo changed unexpectedly",
    )
    require(
        lightx2v["lora_path"].endswith("Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"),
        "Unexpected Lightx2v LoRA filename",
    )
    require(re.fullmatch(r"[0-9a-f]{40}", lightx2v["model_revision"]) is not None, "Bad Lightx2v LoRA revision")
    require(int(lightx2v["min_bytes"]) >= 730000000, "Lightx2v LoRA min size is too low")
    require(int(lightx2v["recommended_sample_steps"]) == 6, "Lightx2v LoRA should be configured for 6 steps")
    require(float(lightx2v["recommended_sample_shift"]) == 5.0, "Lightx2v LoRA should use workflow shift 5")

    comfy = runtime["comfyui"]
    require(comfy["root"] == "/opt/ComfyUI", "ComfyUI code should run from /opt/ComfyUI")
    require(re.fullmatch(r"[0-9a-f]{40}", comfy["wan_revision"]) is not None, "Bad Comfy Wan revision")
    require(re.fullmatch(r"[0-9a-f]{40}", comfy["sam31_revision"]) is not None, "Bad Comfy SAM3.1 revision")
    require(len(comfy["required_files"]) >= 6, "Expected ComfyUI model links")
    require(any(item["target"].endswith("wan2.1_14B_SCAIL_2_fp8_scaled.safetensors") for item in comfy["required_files"]), "ComfyUI workflow needs SCAIL fp8 diffusion model")
    require(any(item["target"].endswith("umt5_xxl_fp8_e4m3fn_scaled.safetensors") for item in comfy["required_files"]), "ComfyUI workflow needs fp8 UMT5")
    require(any(item["target"].endswith("sam3.1_multiplex_fp16.safetensors") for item in comfy["required_files"]), "ComfyUI workflow needs SAM3.1 checkpoint")

    requirements = (ROOT / "requirements-runtime.txt").read_text(encoding="utf-8")
    require("flash_attn" not in requirements, "flash_attn should remain optional; use the SDPA fallback patch")
    require("torch" not in requirements, "base image should provide torch")
    require("ultralytics==8.4.68" in requirements, "SAM3-capable ultralytics must be pinned")

    patch_script = (ROOT / "scripts" / "patch_scail2_attention.py").read_text(encoding="utf-8")
    require("SCAIL2_RUNPOD_SDPA_FALLBACK" in patch_script, "Attention fallback patch marker is missing")
    require("scaled_dot_product_attention" in patch_script, "Attention fallback patch must use torch SDPA")
    app_script = (ROOT / "scripts" / "app.py").read_text(encoding="utf-8")
    run_script = (ROOT / "scripts" / "run_scail2.py").read_text(encoding="utf-8")
    prepare_script = (ROOT / "scripts" / "prepare_models.py").read_text(encoding="utf-8")
    start_script = (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")
    require("Lightning LoRA" in app_script, "UI must expose the Lightx2v preset")
    require("--lightx2v-lora" in run_script, "CLI must expose the Lightx2v LoRA flag")
    require(".prepare.lock.d" in prepare_script, "Model preparation lock must use a directory lock")
    require("fcntl" not in prepare_script, "Model preparation must not rely on fcntl/flock on /workspace")
    require("PREPARE_LOCK_STALE_SECONDS" in prepare_script, "Model preparation lock must be recoverable")
    require("--download-comfyui" in prepare_script, "Model preparation must support ComfyUI models")
    require("link_or_copy" in prepare_script, "Model preparation must link ComfyUI model files")
    require("START_COMFYUI" in start_script, "Startup must default to ComfyUI")
    require("scail2-comfyui-infinity-runpod.json" in start_script, "Startup must copy the bundled workflow")
    require("--enable-cors-header" in start_script, "ComfyUI must allow RunPod proxy origins")
    require("--user-directory" in start_script, "ComfyUI must use the persistent workspace user directory")
    env_template = (ROOT / "runpod-template.env.example").read_text(encoding="utf-8")
    require("COMFYUI_CORS_ORIGIN=*" in env_template, "RunPod template must expose proxy-friendly ComfyUI CORS")

    model_loading_patch = (ROOT / "scripts" / "patch_scail2_model_loading.py").read_text(encoding="utf-8")
    require("SCAIL2_RUNPOD_GPU_AWARE_FP8_V2_MODEL_LOADING" in model_loading_patch, "GPU-aware fp8 loading patch marker is missing")
    require("safe_open" in model_loading_patch, "Low-memory loading patch must stream safetensors")
    require("_scale_tensor_for_target" in model_loading_patch, "Loader must dequantize fp8 scaled weights")
    require(".scale_weight" in model_loading_patch, "Loader must consume Comfy fp8 scale_weight tensors")
    require('key == "scaled_fp8"' in model_loading_patch, "Loader must ignore Comfy fp8 marker tensor")
    require("init_on_cpu=args.offload_model" in model_loading_patch, "generate.py must pass init_on_cpu from offload_model")
    require("model_device = torch.device(\"cpu\") if init_on_cpu else self.device" in model_loading_patch, "SCAIL model must load on GPU when CPU offload is disabled")
    require("with torch.device(device)" in model_loading_patch, "SCAIL model construction must happen on the selected device")
    require("torch.set_default_dtype(dtype)" in model_loading_patch, "SCAIL model construction must avoid default fp32 weights")

    lora_patch = (ROOT / "scripts" / "patch_scail2_lora.py").read_text(encoding="utf-8")
    require("SCAIL2_RUNPOD_INPLACE_LORA_FUSION" in lora_patch, "In-place LoRA patch marker is missing")
    require("target.add_" in lora_patch, "LoRA patch must apply deltas in-place")

    infinity_patch = (ROOT / "scripts" / "patch_scail2_infinity.py").read_text(encoding="utf-8")
    require("SCAIL2_RUNPOD_INFINITY_WINDOWING" in infinity_patch, "Infinity windowing patch marker is missing")
    require("slice_with_tail_pad" in infinity_patch, "Infinity patch must pad short tail windows")
    require("Trimming stitched output" in infinity_patch, "Infinity patch must trim final overshoot")

    for script in [
        "prepare_models.py",
        "patch_scail2_attention.py",
        "patch_scail2_model_loading.py",
        "patch_scail2_lora.py",
        "patch_scail2_infinity.py",
        "run_scail2.py",
        "app.py",
        "job_worker.py",
        "validate_repo.py",
    ]:
        py_compile.compile(str(ROOT / "scripts" / script), doraise=True)

    all_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix in {".md", ".py", ".json", ".txt", ".sh", ".env", ".example", ""}
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.relative_to(ROOT) != Path("scripts/validate_repo.py")
    )
    forbidden = [
        "Wan21-14B-SCAIL-preview",
        "RenderNLFPoses",
        "PoseDetectionVitPoseToDWPose",
        "ComfyUI-WanVideoWrapper",
        "wan21_scail_pose_dance",
    ]
    for token in forbidden:
        require(token not in all_text, f"Old SCAIL-Preview/ComfyUI token remains: {token}")

    app = (ROOT / "scripts" / "app.py").read_text(encoding="utf-8")
    runner = (ROOT / "scripts" / "run_scail2.py").read_text(encoding="utf-8")
    start = (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")
    require("wan2.1_14B_SCAIL_2_fp8_scaled.safetensors" in runner, "Runner default must use fp8 scaled weights")
    require("wan2.1_14B_SCAIL_2_fp8_scaled.safetensors" in start, "Startup default must use fp8 scaled weights")
    require("REFRESH_RUNTIME_CONFIG" in start, "Startup must refresh stale runtime configs by default")
    require('"sample_steps": 6' in app and '"sample_shift": 5.0' in app, "UI must default to the 6-step fast profile")
    require("--download-lightx2v-lora" in runner, "Runner must request Lightx2v LoRA downloads when needed")

    workflow = json.loads((ROOT / "workflows" / "scail2-comfyui-infinity-runpod.json").read_text(encoding="utf-8"))
    node_ids = {node["id"] for node in workflow["nodes"]}
    link_ids = {link[0] for link in workflow["links"]}
    for link in workflow["links"]:
        require(link[1] in node_ids and link[3] in node_ids, f"Workflow link references missing node: {link}")
    for node in workflow["nodes"]:
        for input_item in node.get("inputs", []) or []:
            link_id = input_item.get("link")
            require(link_id is None or link_id in link_ids, f"Workflow node {node['id']} input has missing link: {link_id}")
        for output_item in node.get("outputs", []) or []:
            for link_id in output_item.get("links") or []:
                require(link_id in link_ids, f"Workflow node {node['id']} output has missing link: {link_id}")
    node_types = {node["type"] for node in workflow["nodes"]}
    require("WanSCAILInfinity" in node_types, "Workflow must use the bundled Infinity node")
    require("VHS_LoadVideo" in node_types and "VHS_VideoCombine" in node_types, "Workflow must load and save video")
    require("SAM3_VideoTrack" in node_types and "SCAIL2ColoredMask" in node_types, "Workflow must build SAM3.1 SCAIL masks")
    forbidden_workflow_nodes = {
        "PathchSageAttentionKJ",
        "RTXVideoSuperResolution",
        "RIFE VFI",
        "easy imageColorMatch",
        "easy clearCacheAll",
        "easy cleanGpuUsed",
        "MathExpression|pysssss",
    }
    require(not (node_types & forbidden_workflow_nodes), f"Workflow still depends on optional nodes: {node_types & forbidden_workflow_nodes}")
    workflow_text = json.dumps(workflow, ensure_ascii=False)
    require("Wan21_PusaV1" not in workflow_text, "Workflow must not require Pusa LoRA")
    require("Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors" in workflow_text, "Workflow must use bundled LightX2V rank64 LoRA")

    print("Repository validation passed.")
    print(f"SCAIL-2 code: {scail['code_commit']}")
    print(f"SCAIL-2 model revision: {scail['model_revision']}")
    print(f"SCAIL-2 weights revision: {scail['scail_weights_revision']}")
    print(f"LightX2V LoRA revision: {lightx2v['model_revision']}")
    print(f"SAM3 auto-mask: optional gated model")
    print(f"Lightx2v LoRA: default 6-step acceleration")


if __name__ == "__main__":
    main()
