# =============================================================================
# SOL ExecBench Docker Image
# Includes: CUDA 13.1, cuDNN 9.8, CUTLASS v4.4.1, Nsight Systems
# Python deps installed via uv sync from pyproject.toml/uv.lock
# =============================================================================

FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04 AS base

ENV PATH=/usr/local/cuda/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc

# System packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    curl git vim tmux gdb valgrind git-lfs wget \
    python3 python3-dev \
    build-essential libblas-dev liblapack-dev \
    ccache cmake ninja-build && \
    nvcc --version

# Install CUTLASS (headers used by cutlass-dsl and solutions)
RUN git clone --depth 1 -b v4.4.1 https://github.com/NVIDIA/cutlass.git /usr/local/cutlass
ENV CUTLASS_DIR="/usr/local/cutlass" \
    CPLUS_INCLUDE_PATH="/usr/local/cutlass/include"

# User and directory setup
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG HOST_USER=sol-execbench

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

RUN groupadd -f -g ${HOST_GID} ${HOST_USER} && \
    useradd -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_USER} && \
    mkdir -p /sol-execbench /venv && \
    chown -R ${HOST_UID}:${HOST_GID} /home/${HOST_USER} /sol-execbench /venv

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /sol-execbench
USER ${HOST_UID}:${HOST_GID}

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/venv

# Install Python dependencies via uv sync (all deps in pyproject.toml)
COPY --chown=${HOST_UID}:${HOST_GID} pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/home/${HOST_USER}/.cache/uv,uid=${HOST_UID},gid=${HOST_GID} \
    uv sync --frozen --no-install-project

# Add installed cudnn-frontend
ENV CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH};${UV_PROJECT_ENVIRONMENT}/lib/python3.12/site-packages/include"
    
# Copy SOL ExecBench problem dataset including safetensors blobs.
# Requires: ./scripts/download_flashinfer_bench.sh
COPY --chown=${HOST_UID}:${HOST_GID} data/flashinfer-trace /opt/sol-execbench/benchmark

# Copy source code (changes frequently - do this last)
COPY --chown=${HOST_UID}:${HOST_GID} src/ ./src/

ENV PATH="/venv/bin:${PATH}" \
    PYTHONPATH="/sol-execbench/src" \
    SOLEXECBENCH_BENCHMARK_DIR="/opt/sol-execbench/benchmark"

ENTRYPOINT ["/entrypoint.sh"]
