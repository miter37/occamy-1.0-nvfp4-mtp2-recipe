#!/usr/bin/env bash
set -euo pipefail
# Adapt to your layout. Defaults to the parent of this script.
ROOT="${OCCAMY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONTAINER_NAME="self_vllm_occamy"
if ! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "container: absent"
  exit 1
fi
docker inspect --format 'container: {{.Name}} status={{.State.Status}} running={{.State.Running}}' "$CONTAINER_NAME"
printf 'models: '
curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models || true
printf '\nhealth: '
curl -fsS --max-time 5 -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health || true
printf 'mtp_logs: '
docker logs "$CONTAINER_NAME" 2>&1 | grep -E 'speculative_config|speculative|MTP|num_speculative_tokens' | tail -5 || true
