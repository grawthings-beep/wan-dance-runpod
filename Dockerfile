# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=runpod/comfyui:1.4.1-cuda12.8@sha256:ec9620c0eee0a3f92b58c1647153d1ac0d4c72fc910a7c882ddd89d90391378a
FROM ${BASE_IMAGE}

ARG WAN_WRAPPER_COMMIT=088128b224242e110d3906c6750e9a3a348a659b
ARG SCAIL_POSE_COMMIT=1db9ac3b44b93c336402512c118d35a89a0bee58
ARG WAN_PREPROCESS_COMMIT=0e0b6a2a555625acf4d4aefb780e27d06937132f
ARG KJNODES_COMMIT=a8fd39cbe6e03249463131f0a407d89729c266e4
ARG VHS_COMMIT=4ee72c065db22c9d96c2427954dc69e7b908444b

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        aria2 \
        ca-certificates \
        curl \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/wan-dance

RUN git clone --filter=blob:none --no-checkout https://github.com/kijai/ComfyUI-WanVideoWrapper.git custom_nodes/ComfyUI-WanVideoWrapper \
    && git -C custom_nodes/ComfyUI-WanVideoWrapper checkout "${WAN_WRAPPER_COMMIT}" \
    && git clone --filter=blob:none --no-checkout https://github.com/kijai/ComfyUI-SCAIL-Pose.git custom_nodes/ComfyUI-SCAIL-Pose \
    && git -C custom_nodes/ComfyUI-SCAIL-Pose checkout "${SCAIL_POSE_COMMIT}" \
    && git clone --filter=blob:none --no-checkout https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git custom_nodes/ComfyUI-WanAnimatePreprocess \
    && git -C custom_nodes/ComfyUI-WanAnimatePreprocess checkout "${WAN_PREPROCESS_COMMIT}" \
    && git clone --filter=blob:none --no-checkout https://github.com/kijai/ComfyUI-KJNodes.git custom_nodes/ComfyUI-KJNodes \
    && git -C custom_nodes/ComfyUI-KJNodes checkout "${KJNODES_COMMIT}" \
    && git clone --filter=blob:none --no-checkout https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git custom_nodes/ComfyUI-VideoHelperSuite \
    && git -C custom_nodes/ComfyUI-VideoHelperSuite checkout "${VHS_COMMIT}" \
    && find custom_nodes -type d -name .git -prune -exec rm -rf {} +

COPY requirements-runtime.txt /opt/wan-dance/requirements-runtime.txt
# Base image python is system python3.12 (EXTERNALLY-MANAGED removed upstream).
# The runtime venv (.venv-cu128) is created with --system-site-packages, so
# packages installed here are visible inside it.
RUN python3.12 -m pip install --no-cache-dir -r /opt/wan-dance/requirements-runtime.txt
RUN python3.12 -c "import cv2, diffusers, onnx, onnxruntime, taichi; import google.protobuf"
RUN set -eux; \
    for source in /opt/wan-dance/custom_nodes/*; do \
      name="$(basename "${source}")"; \
      rm -rf "/opt/comfyui-baked/custom_nodes/${name}"; \
      ln -s "${source}" "/opt/comfyui-baked/custom_nodes/${name}"; \
    done; \
    cd /opt/comfyui-baked; \
    python3.12 main.py \
      --cpu \
      --quick-test-for-ci \
      --disable-api-nodes \
      --database-url sqlite:///:memory: \
      --disable-all-custom-nodes \
      --whitelist-custom-nodes \
        ComfyUI-WanVideoWrapper \
        ComfyUI-SCAIL-Pose \
        ComfyUI-WanAnimatePreprocess \
        ComfyUI-KJNodes \
        ComfyUI-VideoHelperSuite

COPY config/ /opt/wan-dance/config/
COPY scripts/ /opt/wan-dance/scripts/
COPY workflows/ /opt/wan-dance/workflows/
RUN chmod +x /opt/wan-dance/scripts/*.sh

EXPOSE 8188

# Our wrapper provisions wan-dance, then execs the base /start.sh
ENTRYPOINT ["/opt/wan-dance/scripts/start.sh"]
