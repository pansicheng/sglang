# Accuracy + throughput eval: FlashInfer vs Triton on gpt-oss-20b (A10).
#
# For each backend, launch a server, then run gsm8k and mmlu via
# sglang.test.run_eval. Aggregates accuracy + observed gen throughput.
#
# Usage:
#   bash test/manual/eval_gpt_oss_flashinfer_vs_triton.sh

set -uo pipefail

MODEL_PATH=${GPT_OSS_MODEL_PATH:-/data-mnt/gpt-oss-20b/}
GSM8K_PATH=${GSM8K_PATH:-/data-mnt/test.jsonl}
CUDA_DEVICE=${CUDA_DEVICE:-0}
PORT=${PORT:-31000}
HOST=127.0.0.1
NUM_Q=${NUM_Q:-200}
PARALLEL=${PARALLEL:-32}

BACKENDS=(triton flashinfer)
LOG_DIR=/tmp
SUMMARY="$LOG_DIR/eval_summary.md"
: > "$SUMMARY"

wait_ready() {
    local timeout=${1:-300}
    local t0=$SECONDS
    while (( SECONDS - t0 < timeout )); do
        if curl -s "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

run_one_backend() {
    local backend="$1"
    local server_log="$LOG_DIR/eval_server_${backend}.log"
    local gsm8k_log="$LOG_DIR/eval_gsm8k_${backend}.log"
    local mmlu_log="$LOG_DIR/eval_mmlu_${backend}.log"

    echo "=========================================="
    echo "Backend: ${backend}"
    echo "=========================================="
    rm -f "$server_log" "$gsm8k_log" "$mmlu_log"

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
        python3 -m sglang.launch_server \
            --model-path "$MODEL_PATH" \
            --attention-backend "$backend" \
            --host "$HOST" \
            --port "$PORT" \
            --random-seed 1 \
            > "$server_log" 2>&1 &
    local server_pid=$!
    echo "  server pid=$server_pid (log=$server_log)"

    if ! wait_ready 600; then
        echo "  ERROR: server did not become ready in 600s"
        tail -40 "$server_log" | sed 's/^/    /'
        kill -9 "$server_pid" 2>/dev/null
        return 1
    fi
    echo "  server ready."

    echo "  -> gsm8k (${NUM_Q} questions, parallel=${PARALLEL})"
    python3 -m sglang.test.run_eval \
        --base-url "http://${HOST}:${PORT}" \
        --eval-name gsm8k \
        --num-examples "$NUM_Q" \
        --num-threads "$PARALLEL" \
        --gsm8k-data-path "$GSM8K_PATH" \
        > "$gsm8k_log" 2>&1
    echo "    rc=$?"
    tail -20 "$gsm8k_log" | sed 's/^/    /'

    echo "  -> mmlu (${NUM_Q} questions, parallel=${PARALLEL})"
    python3 -m sglang.test.run_eval \
        --base-url "http://${HOST}:${PORT}" \
        --eval-name mmlu \
        --num-examples "$NUM_Q" \
        --num-threads "$PARALLEL" \
        > "$mmlu_log" 2>&1
    echo "    rc=$?"
    tail -20 "$mmlu_log" | sed 's/^/    /'

    echo "  shutting down server pid=$server_pid"
    kill -INT "$server_pid" 2>/dev/null
    sleep 5
    kill -9 "$server_pid" 2>/dev/null
    sleep 5
    pkill -9 -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null
    sleep 3
}

extract_score() {
    local log="$1"
    grep -E "Total: score" "$log" | tail -1 | sed -E 's/.*score:\s*//; s/\s+.*//'
}

extract_latency() {
    local log="$1"
    grep -E "Total latency" "$log" | tail -1 | awk '{print $(NF-1)}'
}

extract_eval_throughput() {
    # Reported in run_eval output (overall sample/sec)
    local log="$1"
    grep -E "Score:" "$log" | tail -1
}

for backend in "${BACKENDS[@]}"; do
    run_one_backend "$backend"
done

# === Aggregate ===
{
    echo "# gpt-oss-20b accuracy/throughput summary"
    echo
    echo "| Backend | gsm8k score | gsm8k latency (s) | mmlu score | mmlu latency (s) |"
    echo "|---|---|---|---|---|"
    for b in "${BACKENDS[@]}"; do
        gs=$(extract_score "$LOG_DIR/eval_gsm8k_${b}.log")
        gl=$(extract_latency "$LOG_DIR/eval_gsm8k_${b}.log")
        ms=$(extract_score "$LOG_DIR/eval_mmlu_${b}.log")
        ml=$(extract_latency "$LOG_DIR/eval_mmlu_${b}.log")
        echo "| ${b} | ${gs:-?} | ${gl:-?} | ${ms:-?} | ${ml:-?} |"
    done
} | tee "$SUMMARY"

echo
echo "Summary: $SUMMARY"
