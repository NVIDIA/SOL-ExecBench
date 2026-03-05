#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="sol-execbench"
CONTAINER_NAME="sol-execbench-dev"

# Build the image (pass host UID/GID so file permissions match)
docker build \
    -f docker/Dockerfile \
    --build-arg HOST_UID="$(id -u)" \
    --build-arg HOST_GID="$(id -g)" \
    --build-arg HOST_USER="$(whoami)" \
    -t "${IMAGE_NAME}" \
    .

# Remove any stale container with the same name
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# Run the container
exec docker run -it \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    -v "$(pwd)":/sol-execbench \
    -w /sol-execbench \
    "${IMAGE_NAME}" \
    bash
