#!/bin/bash
set -e

DATASET=/data-mnt/ShareGPT_V3_unfiltered_cleaned_split.json
MODEL=/data-mnt/gpt-oss-20b/
PORT=30000
URL=http://127.0.0.1:${PORT}

# ============================================================
# 1. Restart server
# ============================================================
echo "=== Killing existing sglang processes ==="
pkill -f "sglang.launch_server" 2>/dev/null || true
pkill -f "sglang.serve" 2>/dev/null || true
sleep 3

export SGLANG_ELASTIC_MEM_POOL=true
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

LOG_FILE=/tmp/sglang_server_$(date +%Y%m%d_%H%M%S).log

echo "=== Starting SGLang server with ElasticMem (TP=2) ==="
echo "=== Log file: ${LOG_FILE} ==="
python -m sglang.launch_server \
    --model-path ${MODEL} \
    --mem-fraction-static 0.5 \
    --swa-full-tokens-ratio 1 \
    --tp 2 \
    --chunked-prefill-size 1024 \
    --log-level debug \
    --host 127.0.0.1 \
    --port ${PORT} \
    2>&1 | tee ${LOG_FILE} &

SERVER_PID=$!

echo "=== Waiting for server (PID: $SERVER_PID) ==="
for i in $(seq 1 180); do
    if curl -s ${URL}/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died"
        exit 1
    fi
    sleep 2
done

# Quick smoke test
echo ""
echo "=== Smoke test ==="
curl -s ${URL}/generate \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 8}}' \
    | python3 -m json.tool

# ============================================================
# 2. Stress test phases
# ============================================================
# Model info:
#     sliding_window=128, 12 full + 12 SWA layers
#     full_pool=180180 tokens, swa_pool=180180 tokens
#
# Resize triggers when:
#     - One pool usage > 0.7 (CAN_MAP_THRESHOLD)
#     - Another pool usage < 0.3 (CAN_UNMAP_THRESHOLD)
#     - Diff > 0.3 (RESIZE_TRIGGER_DIFF_RATIO)
# ============================================================

echo ""
echo "============================================================"
echo "Phase 1: Long-input to create full/SWA imbalance"
echo "    8192 input tokens x 64 concurrent -> full pool ~70%+"
echo "    SWA pool stays <5% -> should trigger elastic resize"
echo "============================================================"
python3 -m sglang.bench_serving --backend sglang \
    --port ${PORT} \
    --dataset-name random \
    --dataset-path /data-mnt/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 256 \
    --random-input 8192 --random-output 1024 --random-range-ratio 0.5 \
    --max-concurrency 64

echo ""
echo "=== All phases complete. Server still running (PID: $SERVER_PID) ==="
echo "To stop: kill $SERVER_PID"
wait $SERVER_PID
