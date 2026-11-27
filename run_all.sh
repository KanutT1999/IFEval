#!/usr/bin/env bash
# One command, whole study:   bash run_all.sh
#
# Env overrides:
#   MODEL_SET=v100_16gb|v100_32gb|open_only   (configs/grid.yaml: model_sets)
#   PHASES="1 2 3"     phase 1 = headline for every model; 2 = repeats + rewordings;
#                      3 = template ablation + sampling. Phase-major order means a
#                      budget cut-off still leaves every model with its headline.
#   BACKEND=auto|vllm|hf
# Resumable: completed runs are skipped, so re-running after a crash continues.
set -uo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
[ -d .venv ] && source .venv/bin/activate

MODEL_SET=${MODEL_SET:-v100_16gb}
PHASES=${PHASES:-"1 2 3"}
BACKEND=${BACKEND:-auto}
mkdir -p results/logs
LOG=results/logs/run_all_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "== run_all  MODEL_SET=$MODEL_SET  PHASES=$PHASES  BACKEND=$BACKEND  $(date -u)"
T0=$(date +%s)

python -m ifeval_v100.preflight --model-set "$MODEL_SET" || exit 1
MODELS=$(python -c "from ifeval_v100 import common; print(' '.join(common.load_config()['model_sets']['$MODEL_SET']))")
SKIP=$(cat results/env/unavailable_models.txt 2>/dev/null || true)

for PH in $PHASES; do
  for M in $MODELS; do
    if echo "$SKIP" | grep -qx "$M"; then echo "[skip] $M unavailable"; continue; fi
    echo "== phase $PH  model $M  elapsed $(( ($(date +%s)-T0)/60 )) min"
    python -m ifeval_v100.run_model --model "$M" --phase "$PH" --backend "$BACKEND" \
      || echo "$M phase $PH FAILED" | tee -a results/logs/failures.txt
  done
done

python -m ifeval_v100.validate_scorer
python -m ifeval_v100.analyze
echo "== done in $(( ($(date +%s)-T0)/60 )) min.  Report: results/report/REPORT.md"
echo "   Pack for download:  tar czf ifeval_results.tgz results configs"
