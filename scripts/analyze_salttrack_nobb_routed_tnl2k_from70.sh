#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/data/wyp/SALTTrack"
PYTHON_BIN="/data/envs/wyp_vlt/bin/python"
TRACKER_NAME="salttrack"
DATASET_NAME="tnl2k"
START_EPOCH=70
OUT_DIR="${ROOT_DIR}/output/eval_logs/nobb_routed_tnl2k_metrics"
SUMMARY_CSV="${OUT_DIR}/summary_from${START_EPOCH}.csv"

CONFIGS=(
  "salttrack_base_nobb_routed_direction"
  "salttrack_base_nobb_routed_both"
  "salttrack_large_nobb_routed_direction"
  "salttrack_large_nobb_routed_both"
)

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENCV_OPENCL_RUNTIME=disabled
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

echo "config,run_tag,auc,precision,norm_precision,valid_sequences,total_sequences" > "${SUMMARY_CSV}"

for CONFIG in "${CONFIGS[@]}"; do
  CONFIG_CSV="${OUT_DIR}/${DATASET_NAME}_${CONFIG}_metrics_from${START_EPOCH}.csv"
  LOG_FILE="${OUT_DIR}/${DATASET_NAME}_${CONFIG}_metrics_from${START_EPOCH}.log"

  echo "============================================================"
  echo "Analyzing ${CONFIG}"
  echo "CSV: ${CONFIG_CSV}"
  echo "LOG: ${LOG_FILE}"

  "${PYTHON_BIN}" tracking/analyze_salttrack_checkpoints.py \
    --tracker_name "${TRACKER_NAME}" \
    --tracker_param "${CONFIG}" \
    --dataset_name "${DATASET_NAME}" \
    --start_epoch "${START_EPOCH}" \
    --output_csv "${CONFIG_CSV}" \
    2>&1 | tee "${LOG_FILE}"

  tail -n +2 "${CONFIG_CSV}" | while IFS= read -r line; do
    echo "${CONFIG},${line}" >> "${SUMMARY_CSV}"
  done
done

echo "============================================================"
echo "Summary CSV: ${SUMMARY_CSV}"
echo ""
echo "Top rows by AUC:"
"${PYTHON_BIN}" - <<'PY' "${SUMMARY_CSV}"
import csv
import sys

summary_csv = sys.argv[1]
with open(summary_csv, newline="") as f:
    rows = list(csv.DictReader(f))

rows.sort(key=lambda row: float(row["auc"]), reverse=True)
print("config,run_tag,AUC,P,NP,valid/total")
for row in rows[:20]:
    print("{},{},{:.2f},{:.2f},{:.2f},{}/{}".format(
        row["config"],
        row["run_tag"],
        float(row["auc"]),
        float(row["precision"]),
        float(row["norm_precision"]),
        row["valid_sequences"],
        row["total_sequences"],
    ))
PY
