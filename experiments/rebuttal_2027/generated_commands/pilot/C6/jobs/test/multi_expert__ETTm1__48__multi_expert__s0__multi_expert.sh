#!/usr/bin/env bash
set -euo pipefail
cd /opt/data/private/online
gpu_ids="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
gpu_args=(--gpus "$gpu_ids" --max-per-gpu "${MAX_PER_GPU:-1}")
calibration_args=(--gpus "${CALIBRATION_GPU:-${gpu_ids%%,*}}" --max-per-gpu 1)
exec_args=()
if [[ "${EXECUTE:-0}" == 1 ]]; then exec_args+=(--execute); fi
python /opt/data/private/online/scripts/rebuttal_suite.py run --experiment C6 --scope pilot --run-id rebuttal --output-root /opt/data/private/online/experiments/rebuttal_2027/runs --data-root /opt/data/private/online/data --seeds 0 --time-match --parameter-proxy-size 128 --phase test --resume --job multi_expert__ETTm1__48__multi_expert__s0__multi_expert "${gpu_args[@]}" "${exec_args[@]}"
