#!/usr/bin/env bash
set -Eeuo pipefail

APP_ROOT="${APP_ROOT:-/opt/wan-dance}"
SCAIL2_REPO="${SCAIL2_REPO:-/opt/SCAIL-2}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/scail2}"
MODEL_ROOT="${MODEL_ROOT:-${WORKSPACE_DIR}/models}"
CONFIG_DIR="${CONFIG_DIR:-/workspace/config}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-${CONFIG_DIR}/scail2-runtime.json}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.12 || command -v python3)}"

mkdir -p "${LOG_DIR}" "${CONFIG_DIR}" "${WORKSPACE_DIR}/input" "${WORKSPACE_DIR}/output" "${MODEL_ROOT}"
exec > >(tee -a "${LOG_DIR}/wan-dance-startup.log") 2>&1
echo "[wan-dance] SCAIL-2 startup $(date -u +%FT%TZ)"

export APP_ROOT SCAIL2_REPO WORKSPACE_DIR MODEL_ROOT
export PYTHONPATH="${SCAIL2_REPO}:${SCAIL2_REPO}/SCAIL-Pose:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/output}"
export SCAIL2_CKPT_DIR="${SCAIL2_CKPT_DIR:-${MODEL_ROOT}/SCAIL-2}"
export SCAIL2_WEIGHTS_DIR="${SCAIL2_WEIGHTS_DIR:-${MODEL_ROOT}/Comfy-Org-SCAIL-2}"
export SCAIL2_SAFETENSORS="${SCAIL2_SAFETENSORS:-${SCAIL2_WEIGHTS_DIR}/diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors}"
export FAST_LORA_DIR="${FAST_LORA_DIR:-${MODEL_ROOT}/LightX2V-Wan2.1-I2V-14B}"
export FAST_LORA_PATH="${FAST_LORA_PATH:-${FAST_LORA_DIR}/loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors}"
export SAM3_MODEL="${SAM3_MODEL:-${MODEL_ROOT}/sam3/sam3.pt}"

if [[ "${REFRESH_RUNTIME_CONFIG:-1}" == "1" || ! -f "${RUNTIME_CONFIG}" ]]; then
  cp "${APP_ROOT}/config/scail2-runtime.json" "${RUNTIME_CONFIG}"
fi

prepare_args=("--config" "${RUNTIME_CONFIG}")
if [[ "${DOWNLOAD_MODELS:-1}" != "1" ]]; then
  prepare_args+=("--skip-download")
  echo "[wan-dance] Skipping SCAIL-2 model downloads (DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-0})."
fi
if [[ "${DOWNLOAD_FAST_LORA:-1}" == "1" ]]; then
  prepare_args+=("--download-fast-lora")
fi
if [[ "${DOWNLOAD_SAM3:-0}" == "1" ]]; then
  prepare_args+=("--download-sam3")
fi

if [[ "${DOWNLOAD_MODELS:-1}" == "1" || "${DOWNLOAD_FAST_LORA:-1}" == "1" || "${DOWNLOAD_SAM3:-0}" == "1" ]]; then
  if [[ "${PREPARE_MODELS_BACKGROUND:-1}" == "1" ]]; then
    echo "[wan-dance] Preparing models in the background; UI will start immediately."
    "${PYTHON_BIN}" "${APP_ROOT}/scripts/prepare_models.py" "${prepare_args[@]}" &
  else
    "${PYTHON_BIN}" "${APP_ROOT}/scripts/prepare_models.py" "${prepare_args[@]}"
  fi
fi

if [[ "${START_GRADIO:-1}" == "1" ]]; then
  echo "[wan-dance] Starting Gradio on ${HOST:-0.0.0.0}:${PORT:-8188}"
  exec "${PYTHON_BIN}" "${APP_ROOT}/scripts/app.py"
fi

echo "[wan-dance] START_GRADIO=${START_GRADIO:-0}; keeping container alive."
tail -f /dev/null
