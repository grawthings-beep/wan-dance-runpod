#!/usr/bin/env bash
set -Eeuo pipefail

# wan-dance provisioning wrapper.
#
# The base image (runpod/comfyui, source: runpod-workers/comfyui-base) ships
# ComfyUI baked at /opt/comfyui-baked and its own entrypoint /start.sh that:
#   - copies ComfyUI to /workspace/runpod-slim/ComfyUI on first boot
#   - creates the python venv (.venv-cu128, --system-site-packages)
#   - starts SSH / JupyterLab / FileBrowser
#   - launches ComfyUI and keeps the container alive if it crashes
#
# This script only does wan-dance specific setup, then execs the base
# entrypoint. It must never exit non-zero, or RunPod will restart the
# container in a loop with no SSH access for debugging.

BASE_START="/start.sh"
COMFYUI_DIR="${COMFYUI_DIR:-/workspace/runpod-slim/ComfyUI}"
BAKED_COMFYUI="${BAKED_COMFYUI:-/opt/comfyui-baked}"
MODEL_ROOT="${MODEL_ROOT:-/workspace/comfyui}"
CONFIG_DIR="${CONFIG_DIR:-/workspace/config}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
MODEL_MANIFEST="${MODEL_MANIFEST:-${CONFIG_DIR}/scail-models.json}"
ARGS_FILE="${ARGS_FILE:-/workspace/runpod-slim/comfyui_args.txt}"

mkdir -p "${LOG_DIR}" "${CONFIG_DIR}"
exec > >(tee -a "${LOG_DIR}/wan-dance-startup.log") 2>&1
echo "[wan-dance] startup $(date -u +%FT%TZ)"

# --- 1. Provision ComfyUI onto the network volume (same layout as base) ---
if [[ ! -f "${COMFYUI_DIR}/main.py" ]]; then
  if [[ -d "${BAKED_COMFYUI}" ]]; then
    echo "[wan-dance] First boot: copying baked ComfyUI to ${COMFYUI_DIR}"
    mkdir -p "$(dirname "${COMFYUI_DIR}")"
    cp -r "${BAKED_COMFYUI}" "${COMFYUI_DIR}"
  else
    echo "[wan-dance] WARNING: ${BAKED_COMFYUI} missing and no ComfyUI at ${COMFYUI_DIR}." >&2
    echo "[wan-dance] Wrong base image? Handing off to base entrypoint." >&2
    exec "${BASE_START}"
  fi
fi

# --- 2. Persistent model directories ---
mkdir -p \
  "${MODEL_ROOT}/models/clip_vision" \
  "${MODEL_ROOT}/models/detection/onnx" \
  "${MODEL_ROOT}/models/diffusion_models/WanVideo/SCAIL" \
  "${MODEL_ROOT}/models/loras/WanVideo/Lightx2v" \
  "${MODEL_ROOT}/models/nlf" \
  "${MODEL_ROOT}/models/text_encoders" \
  "${MODEL_ROOT}/models/vae/wanvideo"

# --- 3. Non-destructive model path mapping ---
# Do NOT replace ${COMFYUI_DIR}/models with a symlink: on a shared volume
# that risks duplicating or losing existing models. extra_model_paths.yaml
# achieves the same lookup without touching anything.
cat > "${COMFYUI_DIR}/extra_model_paths.yaml" <<YAML
wan_dance_workspace:
  base_path: ${MODEL_ROOT}
  clip_vision: models/clip_vision/
  diffusion_models: models/diffusion_models/
  loras: models/loras/
  text_encoders: models/text_encoders/
  vae: models/vae/
  detection: models/detection/
YAML

# DownloadAndLoadNLFModel resolves folder_paths.models_dir directly and
# ignores extra_model_paths, so link only the nlf subfolder.
if [[ ! -e "${COMFYUI_DIR}/models/nlf" ]]; then
  mkdir -p "${COMFYUI_DIR}/models"
  ln -s "${MODEL_ROOT}/models/nlf" "${COMFYUI_DIR}/models/nlf"
fi

# --- 4. Pinned custom nodes ---
mkdir -p "${COMFYUI_DIR}/custom_nodes"
for source in /opt/wan-dance/custom_nodes/*; do
  name="$(basename "${source}")"
  target="${COMFYUI_DIR}/custom_nodes/${name}"
  if [[ -e "${target}" || -L "${target}" ]]; then
    if [[ "${FORCE_PINNED_NODES:-1}" == "1" ]]; then
      rm -rf "${target}"
    else
      echo "[wan-dance] KEEP existing custom node: ${target}"
      continue
    fi
  fi
  ln -s "${source}" "${target}"
done

# --- 5. Seed manifest and workflow ---
if [[ ! -f "${MODEL_MANIFEST}" ]]; then
  cp /opt/wan-dance/config/scail-models.json "${MODEL_MANIFEST}"
fi
WORKFLOW_TARGET="${COMFYUI_DIR}/user/default/workflows/wan21_scail_pose_dance.json"
mkdir -p "$(dirname "${WORKFLOW_TARGET}")"
if [[ ! -f "${WORKFLOW_TARGET}" ]]; then
  # The bundled workflow was authored on Windows; its model paths use
  # backslashes (e.g. WanVideo\SCAIL\...). On Linux the filename lists are
  # built with forward slashes, so unconverted values fail prompt
  # validation with "value not in list". Normalize while seeding.
  sed 's/\\\\/\//g' /opt/wan-dance/workflows/wan21_scail_pose_dance.json > "${WORKFLOW_TARGET}"
fi

# --- 6. ComfyUI args (read by the base entrypoint from comfyui_args.txt) ---
mkdir -p "$(dirname "${ARGS_FILE}")"
touch "${ARGS_FILE}"
if ! grep -q "wan-dance defaults" "${ARGS_FILE}"; then
  {
    echo "# wan-dance defaults"
    echo "--reserve-vram 3"
  } >> "${ARGS_FILE}"
fi

# --- 7. Model downloads (non-fatal: pod must stay reachable on failure) ---
PYTHON_BIN="$(command -v python3.12 || command -v python3)"
if [[ "${DOWNLOAD_MODELS:-1}" == "1" ]]; then
  if ! "${PYTHON_BIN}" /opt/wan-dance/scripts/download_models.py \
      --manifest "${MODEL_MANIFEST}" \
      --root "${MODEL_ROOT}"; then
    echo "[wan-dance] WARNING: model download failed. ComfyUI will still start." >&2
    echo "[wan-dance] Re-run manually: ${PYTHON_BIN} /opt/wan-dance/scripts/download_models.py --manifest ${MODEL_MANIFEST} --root ${MODEL_ROOT}" >&2
  fi
else
  echo "[wan-dance] Skipping model downloads (DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-0})."
fi

# --- 8. Hand off to the base entrypoint (SSH, Jupyter, venv, ComfyUI) ---
echo "[wan-dance] Setup done. Handing off to ${BASE_START}"
exec "${BASE_START}"
