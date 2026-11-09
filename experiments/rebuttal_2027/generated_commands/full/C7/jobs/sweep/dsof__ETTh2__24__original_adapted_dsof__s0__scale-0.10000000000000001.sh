#!/usr/bin/env bash
set -euo pipefail
cd /opt/data/private/online
gpu_ids="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
gpu_args=(--gpus "$gpu_ids" --max-per-gpu "${MAX_PER_GPU:-1}")
calibration_args=(--gpus "${CALIBRATION_GPU:-${gpu_ids%%,*}}" --max-per-gpu 1)
exec_args=()
if [[ "${EXECUTE:-0}" == 1 ]]; then exec_args+=(--execute); fi
python /opt/data/private/online/scripts/rebuttal_suite.py run --experiment C7 --scope full --run-id rebuttal --output-root /opt/data/private/online/experiments/rebuttal_2027/runs --data-root /opt/data/private/online/data --seeds 0 --time-match --parameter-proxy-size 128 --phase sweep --resume --job dsof__ETTh2__24__original_adapted_dsof__s0__scale-0.10000000000000001 "${gpu_args[@]}" "${exec_args[@]}"
