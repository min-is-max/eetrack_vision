FROM nvcr.io/nvidia/isaac-lab:2.1.0

# Install cuda-12.4
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends wget gnupg ca-certificates && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-toolkit-12-4 && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/usr/local/cuda-12.4/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH}"

# Install pointcept dependencies
COPY requirements.docker.txt .
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install -r requirements.docker.txt && \
    /workspace/isaaclab/_isaac_sim/python.sh -m pip install flash-attn --no-build-isolation && \
    rm requirements.docker.txt
