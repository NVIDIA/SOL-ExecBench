# =============================================================================
# Adds cuDNN, CUTLASS, and python dependencies. Used for benchmarking.
# =============================================================================

FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04 AS base

ENV PATH=/usr/local/cuda/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc

# System packages (changes infrequently - cache this layer well)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    curl git vim tmux gdb valgrind git-lfs wget \
    python3 python3-dev \
    build-essential libblas-dev liblapack-dev && \
    nvcc --version

# Install CUTLASS: https://github.com/NVIDIA/cutlass
RUN git clone --depth 1 -b v4.4.1 https://github.com/NVIDIA/cutlass.git /usr/local/cutlass
ENV CUTLASS_DIR="/usr/local/cutlass" \
    CPLUS_INCLUDE_PATH="/usr/local/cutlass/include"

ARG HOST_UID=1000
ARG HOST_GID=1000
ARG HOST_USER=sol-execbench

# Copy uv binary early (changes rarely)
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

# sudo: allows eval driver to call nvidia-smi for GPU clock locking
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y ccache cmake ninja-build sudo

# Create user and setup directories
# Note: Host GID may already exist in base image, so we use -f (force) flag
# When running as root (UID 0), skip user creation since root already exists
RUN if [ "${HOST_UID}" != "0" ]; then \
        groupadd -f -g ${HOST_GID} ${HOST_USER} && \
        useradd -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash ${HOST_USER}; \
    fi && \
    mkdir -p /sol-execbench && mkdir -p /venv && \
    chown -R ${HOST_UID}:${HOST_GID} /sol-execbench /venv && \
    echo "${HOST_USER} ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi" >> /etc/sudoers

# Copy and set up entrypoint while still root (avoid switching back)
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Now switch to user for the rest of the build
WORKDIR /sol-execbench
USER ${HOST_UID}:${HOST_GID}

# Set up UV environment
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/venv

# Copy dependency files (changes less frequently than source code)
COPY --chown=${HOST_UID}:${HOST_GID} pyproject.toml uv.lock README.md ./

# Install base Python dependencies first (this is the slowest part - cache it well)
RUN --mount=type=cache,target=/home/${HOST_USER}/.cache/uv,uid=${HOST_UID},gid=${HOST_GID} \
    uv sync --frozen --no-install-project --all-groups

# Add installed cudnn-frontend
ENV CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH}:${UV_PROJECT_ENVIRONMENT}/lib/python3.12/site-packages/include"

# Copy source code (changes frequently - do this last)
COPY --chown=${HOST_UID}:${HOST_GID} src/ ./src/

# Install the project itself (creates the sol-execbench console script).
RUN --mount=type=cache,target=/home/${HOST_USER}/.cache/uv,uid=${HOST_UID},gid=${HOST_GID} \
    uv sync --frozen --no-editable --all-groups

# Set runtime environment.
# PYTHONPATH lets the volume-mounted src/ override the installed package at runtime.
ENV PATH="/venv/bin:${PATH}" \
    PYTHONPATH="/sol-execbench/src"

ENTRYPOINT ["/entrypoint.sh"]
