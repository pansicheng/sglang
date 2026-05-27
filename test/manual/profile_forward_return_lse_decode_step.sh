#!/usr/bin/env bash
# A/B profile of `forward_return_lse` overhead in a full decode step.
#
# Runs sglang.bench_one_batch (FlashInfer backend, gpt-oss-20b) twice per
# workload:
#   arm A: default — uses decode_wrapper.forward_return_lse + sink correction
#   arm B: BENCH_DISABLE_LSE=1 — monkey-patches forward_return_lse to use plain
#          forward (no LSE write), keeping all other kernels (sinks correction
#          included) identical.
#
# Reports median decode latency (ms) and decode throughput (tok/s) for each
# arm, plus the per-step delta — i.e., the wall-clock cost of the LSE write
# integrated across all 24 layers of one full decode step.
set -euo pipefail

MODEL_PATH=${GPT_OSS_MODEL_PATH:-/data-mnt/gpt-oss-20b/}
CUDA_DEVICE=${CUDA_DEVICE:-0}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-256}
BATCHES=${BATCHES:-"1 16 64"}
LOG_DIR=${LOG_DIR:-/tmp/profile_lse}

mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.md"
: > "$SUMMARY"

run_one() {
    local arm=$1 batch=$2 disable_lse=$3
    local log="$LOG_DIR/arm_${arm}_b${batch}.log"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
    BENCH_DISABLE_LSE="$disable_lse" \
        python3 test/manual/profile_forward_return_lse_decode_step.py \
            --model-path "$MODEL_PATH" \
            --attention-backend flashinfer \
            --batch-size "$batch" \
            --input-len "$INPUT_LEN" \
            --output-len "$OUTPUT_LEN" \
            --disable-piecewise-cuda-graph \
            > "$log" 2>&1
    # Parse "Decode.  median latency: X s, median throughput: Y token/s"
    grep "Decode.  median" "$log" | tail -1
}

extract() {
    local pattern=$1 file=$2
    grep "$pattern" "$file" | tail -1 | sed -E 's/.*: ([0-9.]+).*/\1/'
}

echo "# forward_return_lse overhead per full decode step (gpt-oss-20b, FlashInfer, A10)" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "input=${INPUT_LEN} output=${OUTPUT_LEN}" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
printf "| batch | with-LSE (ms) | no-LSE (ms) | Δ (ms/step) | overhead %% | with-LSE tok/s | no-LSE tok/s |\n" | tee -a "$SUMMARY"
printf "|------:|--------------:|------------:|------------:|-----------:|---------------:|-------------:|\n" | tee -a "$SUMMARY"

for batch in $BATCHES; do
    echo "[arm A with-LSE]  batch=$batch"
    run_one A "$batch" 0 || true
    echo "[arm B no-LSE]    batch=$batch"
    run_one B "$batch" 1 || true

    a_log="$LOG_DIR/arm_A_b${batch}.log"
    b_log="$LOG_DIR/arm_B_b${batch}.log"

    # median latency (s) and throughput (tok/s)
    a_lat=$(grep "Decode.  median" "$a_log" | tail -1 | sed -E 's/.*latency: ([0-9.]+) s.*/\1/' || echo "")
    b_lat=$(grep "Decode.  median" "$b_log" | tail -1 | sed -E 's/.*latency: ([0-9.]+) s.*/\1/' || echo "")
    a_tps=$(grep "Decode.  median" "$a_log" | tail -1 | sed -E 's/.*throughput: ([0-9.]+) token.*/\1/' || echo "")
    b_tps=$(grep "Decode.  median" "$b_log" | tail -1 | sed -E 's/.*throughput: ([0-9.]+) token.*/\1/' || echo "")

    if [[ -z "$a_lat" || -z "$b_lat" ]]; then
        printf "| %d | ? | ? | ? | ? | ? | ? |\n" "$batch" | tee -a "$SUMMARY"
        continue
    fi

    delta_ms=$(python3 -c "print(f'{(${a_lat}-${b_lat})*1000:.3f}')")
    a_ms=$(python3 -c "print(f'{${a_lat}*1000:.3f}')")
    b_ms=$(python3 -c "print(f'{${b_lat}*1000:.3f}')")
    pct=$(python3 -c "print(f'{(${a_lat}-${b_lat})/${b_lat}*100:.2f}')")

    printf "| %d | %s | %s | %s | %s%% | %s | %s |\n" \
        "$batch" "$a_ms" "$b_ms" "$delta_ms" "$pct" "$a_tps" "$b_tps" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "Logs: $LOG_DIR" | tee -a "$SUMMARY"
echo "Δ (ms/step) is the per-decode-step wall-clock cost of the FlashInfer LSE write," | tee -a "$SUMMARY"
echo "summed across all 24 layers (sinks-correction kernel runs in both arms)." | tee -a "$SUMMARY"
