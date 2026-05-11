# Base: GCP Deep Learning image với CUDA 12.1 + PyTorch — sẵn cuDNN, NCCL, gcloud, gsutil.
# Tag pin theo cu121 để khớp T4 driver trên VM image family common-cu121-debian-11.
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.2-1.py310:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WORKDIR=/workspace/Meta-CXR-GCP \
    JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre \
    PATH=/usr/lib/jvm/java-8-openjdk-amd64/jre/bin:${PATH}

# --- OS deps ---
# OpenJDK 8 cho CheXpert labeler; fuse cho gcsfuse; lsb-release cho Cloud SDK apt key.
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-8-jre-headless \
    fuse \
    curl gnupg lsb-release ca-certificates \
    git tini \
 && rm -rf /var/lib/apt/lists/*

# --- gcsfuse từ Google package repo ---
RUN set -eux \
 && GCSFUSE_REPO="gcsfuse-$(lsb_release -c -s)" \
 && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --batch --yes --dearmor -o /usr/share/keyrings/cloud.google.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
    > /etc/apt/sources.list.d/gcsfuse.list \
 && apt-get update && apt-get install -y --no-install-recommends gcsfuse \
 && rm -rf /var/lib/apt/lists/*

# --- Python deps ---
# Pin pip<24.1 vì pytorch_lightning==1.6.5 (transitively required) có metadata
# invalid mà pip ≥24.1 reject. KHÔNG dùng `pip install --upgrade pip` (sẽ bump
# pip lên latest và vỡ).
WORKDIR ${WORKDIR}
COPY requirements.txt .
RUN pip install "pip<24.1" \
 && pip install --no-cache-dir -r requirements.txt \
 && python -m spacy download en_core_web_sm \
 && python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# --- Copy repo (model code + scripts + configs + notebook) ---
COPY . ${WORKDIR}

# --- Runtime defaults ---
EXPOSE 8888
ENTRYPOINT ["/usr/bin/tini", "--"]
# Default = drop into bash; scripts/04_train.sh override với papermill command.
CMD ["/bin/bash"]
