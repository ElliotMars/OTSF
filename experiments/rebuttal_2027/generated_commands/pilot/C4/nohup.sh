#!/usr/bin/env bash
set -euo pipefail
if [[ "${EXECUTE:-0}" != 1 ]]; then echo "Set EXECUTE=1 to launch."; exit 1; fi
runner=$(cd "$(dirname "$0")" && pwd)/run.sh
log_dir=/opt/data/private/online/experiments/rebuttal_2027/runs/pilot/rebuttal/C3/launcher/C4
mkdir -p "$log_dir"
attempt=$(mktemp -d "$log_dir/attempt.XXXXXX")
nohup bash -c 'bash "$1"; rc=$?; printf "%s\n" "$rc" > "$2"; exit "$rc"' _ "$runner" "$attempt/exit_code" > "$attempt/run.log" 2>&1 < /dev/null &
printf "%s\n" "$!" > "$attempt/pid"
printf "%s\n" "$attempt"
