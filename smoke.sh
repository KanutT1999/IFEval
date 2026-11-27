#!/usr/bin/env bash
# 2-3 minute GPU smoke test before the full run: 20 items, smallest model, real backend.
#   bash smoke.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
python -m ifeval_v100.preflight --model-set local_test
python -m ifeval_v100.run_model --model qwen2.5-0.5b --conditions greedy_plain_r0 sample_plain_s0 \
  --limit 20 --backend "${BACKEND:-auto}" --force
echo "Smoke OK -> results/smoke/qwen2.5-0.5b/ . Now: bash run_all.sh"
