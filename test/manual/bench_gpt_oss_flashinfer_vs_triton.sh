# Offline-throughput benchmark: FlashInfer vs Triton on gpt-oss-20b (A10).
#
# A10 is SM86; trtllm_mha (SM100), fa3 (SM90), fa4 (SM100) are not available
# here, so this script restricts the comparison to the two backends that are
# A10-compatible.
#
# Usage:
#   bash test/manual/bench_gpt_oss_flashinfer_vs_triton.sh

set -uo pipefail

MODEL_PATH=${GPT_OSS_MODEL_PATH:-/data-mnt/gpt-oss-20b/}
DATASET_PATH=${DATASET_PATH:-/data-mnt/ShareGPT_V3_unfiltered_cleaned_split.json}
CUDA_DEVICE=${CUDA_DEVICE:-0}
SEED=1

BACKENDS=(triton flashinfer)
WORKLOADS=("128 512 64" "1024 1024 32" "4096 256 16")

LOG_DIR=/tmp
SUMMARY="$LOG_DIR/perf_summary.md"
: > "$SUMMARY"

run_cell() {
    local backend="$1" in_len="$2" out_len="$3" n="$4"
    local log="$LOG_DIR/perf_${backend}_${in_len}_${out_len}_${n}.log"
    echo "=== backend=$backend in=$in_len out=$out_len n=$n -> $log"
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" timeout 600 python3 \
        -m sglang.bench_offline_throughput \
        --model-path "$MODEL_PATH" \
        --attention-backend "$backend" \
        --dataset-name random \
        --dataset-path "$DATASET_PATH" \
        --random-input-len "$in_len" \
        --random-output-len "$out_len" \
        --num-prompts "$n" \
        --seed "$SEED" \
        > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  FAILED (rc=$rc); see tail of $log:"
        tail -20 "$log" | sed 's/^/    /'
    fi
    return $rc
}

extract_metric() {
    local log="$1" key="$2"
    grep -E "^${key}" "$log" | tail -1 | awk -F: '{gsub(/[^0-9.]/,"",$2); print $2}'
}

for w in "${WORKLOADS[@]}"; do
    read -r in_len out_len n <<<"$w"
    for b in "${BACKENDS[@]}"; do
        run_cell "$b" "$in_len" "$out_len" "$n"
    done
done

{
    echo "# FlashInfer vs Triton offline throughput on A10 (gpt-oss-20b)"
    echo
    echo "| Workload (in -> out, n) | Triton total | FI total | Triton out | FI out | total delta |"
    echo "|---|---:|---:|---:|---:|---:|"
    for w in "${WORKLOADS[@]}"; do
        read -r in_len out_len n <<<"$w"
        t_log="$LOG_DIR/perf_triton_${in_len}_${out_len}_${n}.log"
        f_log="$LOG_DIR/perf_flashinfer_${in_len}_${out_len}_${n}.log"
        t_total=$(extract_metric "$t_log" "Total token throughput")
        f_total=$(extract_metric "$f_log" "Total token throughput")
        t_out=$(extract_metric "$t_log" "Output token throughput")
        f_out=$(extract_metric "$f_log" "Output token throughput")
        delta="-"
        if [[ -n "$t_total" && -n "$f_total" ]]; then
            delta=$(awk -v t="$t_total" -v f="$f_total" \
                'BEGIN{ if (t+0>0) printf "%+.1f%%", (f-t)/t*100; else print "-" }')
        fi
        printf "| %s -> %s, n=%s | %s | %s | %s | %s | %s |\n" \
            "$in_len" "$out_len" "$n" \
            "${t_total:--}" "${f_total:--}" \
            "${t_out:--}" "${f_out:--}" "$delta"
    done
} | tee "$SUMMARY"

echo
echo "Summary written to: $SUMMARY"
