# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=runpod/comfyui:latest
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
RUN set -eux; \
    PYTHON_BIN=""; \
    for candidate in /opt/venv/bin/python /venv/bin/python /usr/local/bin/python3 /usr/bin/python3; do \
      if [ -x "${candidate}" ]; then PYTHON_BIN="${candidate}"; break; fi; \
    done; \
    if [ -z "${PYTHON_BIN}" ]; then PYTHON_BIN="$(command -v python3 || command -v python || true)"; fi; \
    test -n "${PYTHON_BIN}"; \
    "${PYTHON_BIN}" -m pip install --upgrade pip; \
    "${PYTHON_BIN}" -m pip install -r /opt/wan-dance/requirements-runtime.txt

COPY config/ /opt/wan-dance/config/
COPY scripts/ /opt/wan-dance/scripts/
COPY workflows/ /opt/wan-dance/workflows/
RUN chmod +x /opt/wan-dance/scripts/*.sh

EXPOSE 8188

ENTRYPOINT []
CMD ["/opt/wan-dance/scripts/start.sh"]

