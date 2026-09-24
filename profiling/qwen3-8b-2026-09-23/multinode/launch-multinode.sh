#!/usr/bin/env bash
set -euo pipefail
TP=${1:?TP required}
if [ "$TP" = 1 ]; then
  VISIBLE=0000:01:00.0
  export TT_MESH_GRAPH_DESC_PATH=/tmp/libtt-profile/tp1.textproto
elif [ "$TP" = 2 ]; then
  VISIBLE=0000:01:00.0,0000:04:00.0
else
  exit 2
fi
exec env -u TT_METAL_RUNTIME_ROOT \
  HF_HOME=/tmp/libtt-profile/hf \
  TT_VISIBLE_DEVICES="$VISIBLE" \
  JAX_PLATFORMS=tt \
  JAX_USE_SHARDY_PARTITIONER="${PROFILE_USE_SHARDY:-true}" \
  JAX_COMPILATION_CACHE_DIR="/tmp/libtt-profile/multinode-fixed-jax-cache-tp$TP" \
  TT_METAL_OPERATION_TIMEOUT_SECONDS=120 \
  /tmp/libtt-profile/multinode-venv/bin/python -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-8B \
    --host 127.0.0.1 --port 31000 \
    --device tt --dtype bfloat16 --attention-backend tt --tp-size "$TP" \
    --max-running-requests 2 --max-total-tokens 1024 \
    --max-prefill-tokens 256 --chunked-prefill-size 256 --page-size 32 \
    --watchdog-timeout 1200 --disable-precompile --skip-server-warmup \
    --disable-overlap-schedule --disable-radix-cache --stream-interval 1
