#!/usr/bin/env bash
set -Eeuo pipefail

APP_ROOT="${APP_ROOT:-/opt/wan-dance}"
SCAIL2_REPO="${SCAIL2_REPO:-/opt/SCAIL-2}"
COMFYUI_ROOT="${COMFYUI_ROOT:-/opt/ComfyUI}"
COMFYUI_WORKSPACE_ROOT="${COMFYUI_WORKSPACE_ROOT:-/workspace/ComfyUI}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/scail2}"
MODEL_ROOT="${MODEL_ROOT:-${WORKSPACE_DIR}/models}"
COMFYUI_MODELS_DIR="${COMFYUI_MODELS_DIR:-${COMFYUI_WORKSPACE_ROOT}/models}"
COMFYUI_INPUT_DIR="${COMFYUI_INPUT_DIR:-${COMFYUI_WORKSPACE_ROOT}/input}"
COMFYUI_OUTPUT_DIR="${COMFYUI_OUTPUT_DIR:-${COMFYUI_WORKSPACE_ROOT}/output}"
CONFIG_DIR="${CONFIG_DIR:-/workspace/config}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-${CONFIG_DIR}/scail2-runtime.json}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.12 || command -v python3)}"

mkdir -p \
  "${LOG_DIR}" \
  "${CONFIG_DIR}" \
  "${WORKSPACE_DIR}/input" \
  "${WORKSPACE_DIR}/output" \
  "${MODEL_ROOT}" \
  "${COMFYUI_MODELS_DIR}" \
  "${COMFYUI_INPUT_DIR}" \
  "${COMFYUI_OUTPUT_DIR}" \
  "${COMFYUI_WORKSPACE_ROOT}/workflows" \
  "${COMFYUI_WORKSPACE_ROOT}/user/default/workflows"
exec > >(tee -a "${LOG_DIR}/wan-dance-startup.log") 2>&1
echo "[wan-dance] SCAIL-2 ComfyUI startup $(date -u +%FT%TZ)"

export APP_ROOT SCAIL2_REPO COMFYUI_ROOT COMFYUI_WORKSPACE_ROOT WORKSPACE_DIR MODEL_ROOT COMFYUI_MODELS_DIR
export PYTHONPATH="${SCAIL2_REPO}:${SCAIL2_REPO}/SCAIL-Pose:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/output}"
export SCAIL2_CKPT_DIR="${SCAIL2_CKPT_DIR:-${MODEL_ROOT}/SCAIL-2}"
export SCAIL2_WEIGHTS_DIR="${SCAIL2_WEIGHTS_DIR:-${MODEL_ROOT}/Comfy-Org-SCAIL-2}"
export SCAIL2_SAFETENSORS="${SCAIL2_SAFETENSORS:-${SCAIL2_WEIGHTS_DIR}/diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors}"
export SAM3_MODEL="${SAM3_MODEL:-${MODEL_ROOT}/sam3/sam3.pt}"
export LIGHTX2V_LORA_PATH="${LIGHTX2V_LORA_PATH:-${MODEL_ROOT}/lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v/loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors}"

if [[ "${REFRESH_RUNTIME_CONFIG:-1}" == "1" || ! -f "${RUNTIME_CONFIG}" ]]; then
  cp "${APP_ROOT}/config/scail2-runtime.json" "${RUNTIME_CONFIG}"
fi

sync_comfy_assets() {
  mkdir -p "${COMFYUI_ROOT}/custom_nodes" "${COMFYUI_ROOT}/models"
  cp -a "${APP_ROOT}/custom_nodes/comfyui-scail2-infinity" "${COMFYUI_ROOT}/custom_nodes/"
  cp "${APP_ROOT}/workflows/scail2-comfyui-infinity-runpod.json" "${COMFYUI_WORKSPACE_ROOT}/workflows/"
  cp "${APP_ROOT}/workflows/scail2-comfyui-infinity-runpod.json" "${COMFYUI_WORKSPACE_ROOT}/user/default/workflows/"

  for model_subdir in checkpoints clip_vision diffusion_models loras text_encoders vae; do
    mkdir -p "${COMFYUI_MODELS_DIR}/${model_subdir}"
    target="${COMFYUI_ROOT}/models/${model_subdir}"
    if [[ -L "${target}" ]]; then
      ln -sfn "${COMFYUI_MODELS_DIR}/${model_subdir}" "${target}"
    elif [[ -e "${target}" ]]; then
      if [[ -d "${target}" && -z "$(find "${target}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        rmdir "${target}"
        ln -s "${COMFYUI_MODELS_DIR}/${model_subdir}" "${target}"
      elif [[ "${COMFYUI_ROOT}" == /opt/* ]]; then
        rm -rf "${target}"
        ln -s "${COMFYUI_MODELS_DIR}/${model_subdir}" "${target}"
      fi
    else
      ln -s "${COMFYUI_MODELS_DIR}/${model_subdir}" "${target}"
    fi
  done
}

sync_comfy_assets

prepare_args=("--config" "${RUNTIME_CONFIG}" "--download-comfyui" "--comfyui-only")
if [[ "${DOWNLOAD_MODELS:-1}" != "1" ]]; then
  prepare_args+=("--skip-download")
  echo "[wan-dance] Skipping model downloads (DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-0})."
fi
if [[ "${DOWNLOAD_SAM31:-1}" != "1" ]]; then
  prepare_args+=("--skip-comfyui-sam31")
fi
if [[ "${DOWNLOAD_LIGHTX2V_LORA:-1}" == "1" && "${START_COMFYUI:-1}" != "1" ]]; then
  prepare_args+=("--download-lightx2v-lora")
fi

if [[ "${DOWNLOAD_MODELS:-1}" == "1" || "${DOWNLOAD_SAM31:-1}" == "1" || "${DOWNLOAD_LIGHTX2V_LORA:-1}" == "1" ]]; then
  if [[ "${PREPARE_MODELS_BACKGROUND:-1}" == "1" ]]; then
    echo "[wan-dance] Preparing ComfyUI models in the background; UI will start immediately."
    "${PYTHON_BIN}" "${APP_ROOT}/scripts/prepare_models.py" "${prepare_args[@]}" &
  else
    "${PYTHON_BIN}" "${APP_ROOT}/scripts/prepare_models.py" "${prepare_args[@]}"
  fi
fi

if [[ "${START_COMFYUI:-1}" == "1" ]]; then
  echo "[wan-dance] Starting ComfyUI on ${HOST:-0.0.0.0}:${PORT:-8188}"
  exec "${PYTHON_BIN}" "${COMFYUI_ROOT}/main.py" \
    --listen "${HOST:-0.0.0.0}" \
    --port "${PORT:-8188}" \
    --input-directory "${COMFYUI_INPUT_DIR}" \
    --output-directory "${COMFYUI_OUTPUT_DIR}" \
    ${COMFYUI_ARGS:-}
fi

if [[ "${START_GRADIO:-0}" == "1" ]]; then
  echo "[wan-dance] Starting Gradio on ${HOST:-0.0.0.0}:${PORT:-8188}"
  exec "${PYTHON_BIN}" "${APP_ROOT}/scripts/app.py"
fi

echo "[wan-dance] START_COMFYUI=${START_COMFYUI:-0}; START_GRADIO=${START_GRADIO:-0}; keeping container alive."
tail -f /dev/null
