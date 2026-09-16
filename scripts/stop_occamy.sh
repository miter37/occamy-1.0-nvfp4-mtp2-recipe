#!/usr/bin/env bash
set -euo pipefail
CONTAINER_NAME="self_vllm_occamy"
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  docker rm -f "$CONTAINER_NAME"
else
  echo "Container already absent: $CONTAINER_NAME"
fi
