# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=runpod/comfyui:1.4.1-cuda12.8
FROM ${BASE_IMAGE}

ARG SCAIL2_COMMIT=f998bcc29127ae9b177711ee8f39d65ccd73cca1
ARG SCAIL_POSE_COMMIT=519c7f54cb972e7f92684213b7ef6c3e05a8f3b2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    SCAIL2_REPO=/opt/SCAIL-2 \
    APP_ROOT=/opt/wan-dance

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        aria2 \
        ca-certificates \
        curl \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/wan-dance

RUN git clone --filter=blob:none --recurse-submodules https://github.com/zai-org/SCAIL-2.git "${SCAIL2_REPO}" \
    && git -C "${SCAIL2_REPO}" checkout "${SCAIL2_COMMIT}" \
    && git -C "${SCAIL2_REPO}" submodule update --init --recursive \
    && git -C "${SCAIL2_REPO}/SCAIL-Pose" checkout "${SCAIL_POSE_COMMIT}" \
    && find "${SCAIL2_REPO}" -type d -name .git -prune -exec rm -rf {} +

COPY scripts/patch_scail2_attention.py /opt/wan-dance/scripts/patch_scail2_attention.py
RUN python3.12 /opt/wan-dance/scripts/patch_scail2_attention.py "${SCAIL2_REPO}"

COPY requirements-runtime.txt /opt/wan-dance/requirements-runtime.txt
RUN python3.12 -m pip install --no-cache-dir -r /opt/wan-dance/requirements-runtime.txt

COPY config/ /opt/wan-dance/config/
COPY scripts/ /opt/wan-dance/scripts/
RUN chmod +x /opt/wan-dance/scripts/*.sh

EXPOSE 8188

ENTRYPOINT ["/opt/wan-dance/scripts/start.sh"]
