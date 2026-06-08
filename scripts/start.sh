#!/usr/bin/env bash
set -Eeuo pipefail

find_python() {
  local candidate
  for candidate in /opt/venv/bin/python /venv/bin/python /usr/local/bin/python3 /usr/bin/python3; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  command -v python3 || command -v python
}

find_comfyui() {
  local candidate
  if [[ -n "${COMFYUI_DIR:-}" && -f "${COMFYUI_DIR}/main.py" ]]; then
    printf '%s\n' "${COMFYUI_DIR}"
    return 0
  fi
  for candidate in /opt/ComfyUI /workspace/ComfyUI /workspace/comfyui /ComfyUI /comfyui /app/ComfyUI; do
    if [[ -f "${candidate}/main.py" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(find_python)" || {
  echo "ERROR: Python was not found." >&2
  exit 2
}
COMFYUI_DIR="$(find_comfyui)" || {
  echo "ERROR: ComfyUI main.py was not found. Set COMFYUI_DIR." >&2
  exit 2
}

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/comfyui}"
MODEL_ROOT="${MODEL_ROOT:-${WORKSPACE_DIR}}"
CONFIG_DIR="${CONFIG_DIR:-/workspace/config}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
MODEL_MANIFEST="${MODEL_MANIFEST:-${CONFIG_DIR}/scail-models.json}"
PORT="${PORT:-8188}"
LISTEN="${LISTEN:-0.0.0.0}"

mkdir -p \
  "${WORKSPACE_DIR}/input" \
  "${WORKSPACE_DIR}/output" \
  "${WORKSPACE_DIR}/user/default/workflows" \
  "${MODEL_ROOT}/models/clip_vision" \
  "${MODEL_ROOT}/models/detection/onnx" \
  "${MODEL_ROOT}/models/diffusion_models/WanVideo/SCAIL" \
  "${MODEL_ROOT}/models/loras/WanVideo/Lightx2v" \
  "${MODEL_ROOT}/models/nlf" \
  "${MODEL_ROOT}/models/text_encoders" \
  "${MODEL_ROOT}/models/vae/wanvideo" \
  "${CONFIG_DIR}" \
  "${LOG_DIR}" \
  "${COMFYUI_DIR}/custom_nodes"

exec > >(tee -a "${LOG_DIR}/wan-dance-startup.log") 2>&1

COMFY_MODELS="${COMFYUI_DIR}/models"
PERSISTENT_MODELS="${MODEL_ROOT}/models"
if [[ "$(readlink -f "${COMFY_MODELS}" 2>/dev/null || printf '%s' "${COMFY_MODELS}")" != "$(readlink -f "${PERSISTENT_MODELS}")" ]]; then
  if [[ -d "${COMFY_MODELS}" && ! -L "${COMFY_MODELS}" ]]; then
    cp -an "${COMFY_MODELS}/." "${PERSISTENT_MODELS}/" || true
  fi
  rm -rf "${COMFY_MODELS}"
  ln -s "${PERSISTENT_MODELS}" "${COMFY_MODELS}"
fi

cat > "${COMFYUI_DIR}/extra_model_paths.yaml" <<YAML
wan_dance_workspace:
  base_path: ${MODEL_ROOT}
  clip_vision: models/clip_vision/
  diffusion_models: models/diffusion_models/
  loras: models/loras/
  text_encoders: models/text_encoders/
  vae: models/vae/
YAML

for source in /opt/wan-dance/custom_nodes/*; do
  name="$(basename "${source}")"
  target="${COMFYUI_DIR}/custom_nodes/${name}"
  if [[ -e "${target}" || -L "${target}" ]]; then
    if [[ "${FORCE_PINNED_NODES:-1}" == "1" ]]; then
      rm -rf "${target}"
    else
      echo "KEEP existing custom node: ${target}"
      continue
    fi
  fi
  ln -s "${source}" "${target}"
done

if [[ ! -f "${MODEL_MANIFEST}" ]]; then
  cp /opt/wan-dance/config/scail-models.json "${MODEL_MANIFEST}"
fi

if [[ ! -f "${WORKSPACE_DIR}/user/default/workflows/wan21_scail_pose_dance.json" ]]; then
  cp /opt/wan-dance/workflows/wan21_scail_pose_dance.json \
    "${WORKSPACE_DIR}/user/default/workflows/wan21_scail_pose_dance.json"
fi

if [[ "${DOWNLOAD_MODELS:-1}" == "1" ]]; then
  "${PYTHON_BIN}" /opt/wan-dance/scripts/download_models.py \
    --manifest "${MODEL_MANIFEST}" \
    --root "${MODEL_ROOT}"
else
  echo "Skipping model downloads because DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-0}."
fi

cd "${COMFYUI_DIR}"
exec "${PYTHON_BIN}" main.py \
  --listen "${LISTEN}" \
  --port "${PORT}" \
  --enable-cors-header "${COMFYUI_CORS_ORIGIN:-*}" \
  --input-directory "${WORKSPACE_DIR}/input" \
  --output-directory "${WORKSPACE_DIR}/output" \
  --user-directory "${WORKSPACE_DIR}/user" \
  ${COMFYUI_ARGS:---reserve-vram 3}
