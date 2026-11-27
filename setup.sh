#!/usr/bin/env bash
# One-time environment setup (Linux x86_64 or WSL2 Ubuntu; NVIDIA driver >= 550).
#   bash setup.sh
# Creates .venv (Python 3.11 via uv), installs the pinned stack, fetches NLTK data,
# verifies data checksums, and runs the CPU-only scorer validation.
set -euo pipefail
cd "$(dirname "$0")"

echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || {
  echo "nvidia-smi failed: no GPU visible"; exit 1; }

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --quiet --user uv || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "== venv (Python 3.11)"
uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -r requirements.txt

echo "== NLTK data (repo-local)"
python - <<'EOF'
import nltk
for r in ("punkt", "punkt_tab"):
    nltk.download(r, download_dir="nltk_data", quiet=True)
EOF

echo "== data checksums"
python -c "from ifeval_v100 import common; common.verify_data(); print('ok: 541 items, checksums match')"

echo "== scorer validation (CPU, ~2 min)"
python -m ifeval_v100.validate_scorer --runs-glob "/nonexistent/*"

echo
echo "Setup done. If you will run Llama models: export HF_TOKEN=hf_... (license accepted on huggingface.co)"
echo "Next:  bash run_all.sh"
