# Profile decode step of bench_one_batch via Nsight Systems, A/B paged vs baseline.
#
# Variant of profile_paged_kv_decode.sh that uses `nsys profile` instead of the
# torch profiler. Output-len defaults to 32 so the .nsys-rep stays small.
#
# CUDA graphs stay ON: nsys is invoked with `--cuda-graph-trace=node`, which
# records per-node timing inside captured graphs, so per-kernel timing shows up
# without having to fall back to eager (`--disable-cuda-graph`).
#
# Assumes HEAD is the paged commit; profiles paged on HEAD, then detaches to
# the pinned baseline commit, and restores the original branch on completion.
#
# Usage:
#   bash scripts/playground/nsys_profile_paged_kv_decode.sh
#
# Outputs (in /tmp by default, override via NSYS_OUT_DIR):
#   /tmp/nsys_<leg>_batch8_input1024_output32_decode.nsys-rep  # open in Nsight Systems UI
#   /tmp/nsys_<leg>_decode.log                                 # bench stdout
#
# Inspect from CLI with e.g.:
#   nsys stats --report cuda_gpu_kern_sum /tmp/nsys_paged_batch8_input1024_output32_decode.nsys-rep
set -u

MODEL="${MODEL:-/data-mnt/Qwen3-8B/}"
B="${B:-8}"
I="${I:-1024}"
O="${O:-32}"   # keep small to bound .nsys-rep size
PAGE_SIZE="${PAGE_SIZE:-64}"
BASELINE_SHA="${BASELINE_SHA:-3da87902d75c7dea405909eb9f2ad98c17b9486e}"
NSYS_OUT_DIR="${NSYS_OUT_DIR:-/tmp}"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: nsys not found in PATH. Install Nsight Systems or add it to PATH." >&2
  exit 127
fi

cd /sgl-workspace/sglang

run_one() {
  local LEG="$1"          # "paged" or "base"
  local PAGED_ENV="$2"    # "SGLANG_PAGED_KV_LAYOUT=1" or ""
  local LOG="/tmp/nsys_${LEG}_decode.log"
  local OUT="${NSYS_OUT_DIR}/nsys_${LEG}_batch${B}_input${I}_output${O}_decode"
  echo "=========================================="
  echo "[$(date +%H:%M:%S)] nsys leg=${LEG}  HEAD=$(git rev-parse --short HEAD)  env='${PAGED_ENV}'"
  echo "  B=${B} I=${I} O=${O} page_size=${PAGE_SIZE}  cuda-graph-trace=node"
  echo "  out=${OUT}.nsys-rep"
  echo "=========================================="
  # --trace=cuda,nvtx + no CPU sampling keeps the report small while still
  # giving per-kernel GPU timing and NVTX ranges (prefill/decode markers).
  # --cuda-graph-trace=node breaks captured CUDA graphs down to per-node timing
  # so we don't need to disable cuda graphs to see per-kernel cost.
  env ${PAGED_ENV} nsys profile \
    --output="${OUT}" \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --sample=none \
    --cpuctxsw=none \
    --gpu-metrics-devices=none \
    --stats=false \
    python -m sglang.bench_one_batch \
      --model-path "${MODEL}" \
      --attention-backend flashinfer \
      --page-size "${PAGE_SIZE}" \
      --batch-size "${B}" \
      --input-len "${I}" \
      --output-len "${O}" \
      --tp 1 \
      --log-decode-step 128 \
      > "${LOG}" 2>&1
  local RC=$?
  echo "  rc=${RC}  log=${LOG}"
  awk '/Benchmark \.\.\./{flag=1} flag && /Prefill\.|Decode\.|median latency/' "${LOG}" || echo "  (metrics not found)"
  ls -la "${OUT}.nsys-rep" 2>/dev/null || echo "  (nsys-rep missing)"
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
echo "===== nsys profile done ====="
echo "Per-kernel GPU summary (CLI):"
echo "  nsys stats --report cuda_gpu_kern_sum ${NSYS_OUT_DIR}/nsys_paged_batch${B}_input${I}_output${O}_decode.nsys-rep"
echo "  nsys stats --report cuda_gpu_kern_sum ${NSYS_OUT_DIR}/nsys_base_batch${B}_input${I}_output${O}_decode.nsys-rep"
echo "NVTX range summary:"
echo "  nsys stats --report nvtx_sum ${NSYS_OUT_DIR}/nsys_paged_batch${B}_input${I}_output${O}_decode.nsys-rep"
echo "Tip: open .nsys-rep in the Nsight Systems UI for the timeline view."
