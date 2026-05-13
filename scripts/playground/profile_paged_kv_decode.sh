# Profile decode step of bench_one_batch via torch profiler, A/B paged vs baseline.
#
# Assumes HEAD is the paged commit; profiles paged on HEAD, then detaches to
# the pinned baseline commit, and restores the original branch on completion.
#
# Usage:
#   bash scripts/playground/profile_paged_kv_decode.sh
#
# Outputs (in /tmp by default, override via SGLANG_TORCH_PROFILER_DIR):
#   /tmp/prof_<leg>_batch8_input1024_output32_decode.trace.json.gz   # Chrome trace
#   /tmp/prof_<leg>_decode.log                                       # stdout incl. key_averages table
#
# Compare the "Self CPU / Self CUDA time total" and the cudaLaunchKernel / step[DECODE] rows.
set -u

MODEL="${MODEL:-/data-mnt/Qwen3-8B/}"
B="${B:-8}"
I="${I:-1024}"
O="${O:-1024}"
PAGE_SIZE="${PAGE_SIZE:-64}"
EAGER="${EAGER:-1}"   # set EAGER=1 to add --disable-cuda-graph so per-kernel GPU time shows up
BASELINE_SHA="${BASELINE_SHA:-3da87902d75c7dea405909eb9f2ad98c17b9486e}"
export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/tmp}"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

EXTRA_ARGS=()
TAG=""
if [ "${EAGER}" = "1" ]; then
  EXTRA_ARGS+=(--disable-cuda-graph)
  TAG="_eager"
fi

cd /sgl-workspace/sglang

run_one() {
  local LEG="$1"          # "paged" or "base"
  local PAGED_ENV="$2"    # "SGLANG_PAGED_KV_LAYOUT=1" or ""
  local LOG="/tmp/prof_${LEG}${TAG}_decode.log"
  echo "=========================================="
  echo "[$(date +%H:%M:%S)] profile leg=${LEG}${TAG}  HEAD=$(git rev-parse --short HEAD)  env='${PAGED_ENV}'"
  echo "  B=${B} I=${I} O=${O} page_size=${PAGE_SIZE}  eager=${EAGER}"
  echo "=========================================="
  env ${PAGED_ENV} python -m sglang.bench_one_batch \
    --model-path "${MODEL}" \
    --attention-backend flashinfer \
    --page-size "${PAGE_SIZE}" \
    --batch-size "${B}" \
    --input-len "${I}" \
    --output-len "${O}" \
    --tp 1 \
    --profile \
    --profile-stage decode \
    --profile-filename-prefix "prof_${LEG}${TAG}" \
    --log-decode-step 128 \
    "${EXTRA_ARGS[@]}" \
    > "${LOG}" 2>&1
  local RC=$?
  echo "  rc=${RC}  log=${LOG}"
  awk '/Benchmark \.\.\./{flag=1} flag && /Self CPU time total|Self CUDA time total|Decode\.  median latency/' "${LOG}" || echo "  (metrics not found)"
  ls -la "${SGLANG_TORCH_PROFILER_DIR}/prof_${LEG}${TAG}_batch${B}_input${I}_output${O}_decode.trace.json.gz" 2>/dev/null || true
}

START_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
run_one paged "SGLANG_PAGED_KV_LAYOUT=1"
echo ""
echo "=== checkout ${BASELINE_SHA} for baseline ==="
git checkout "${BASELINE_SHA}" 2>&1 | tail -2
run_one base ""
echo ""
echo "=== restoring branch ${START_BRANCH} ==="
git checkout "${START_BRANCH}" 2>&1 | tail -1

echo ""
echo "===== profile done ====="
echo "Inspect key_averages tables:"
echo "  sed -n '/^-*  -/,/Self CPU time total/p' /tmp/prof_paged${TAG}_decode.log"
echo "  sed -n '/^-*  -/,/Self CPU time total/p' /tmp/prof_base${TAG}_decode.log"
echo "Tip: rerun with EAGER=1 to expose per-kernel GPU time (no CUDA graph)."
