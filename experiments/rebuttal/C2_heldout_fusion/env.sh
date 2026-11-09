#!/usr/bin/env bash
# Environment of C2 (SAADI fusion weights chosen on held-out queries). Sourced by
# run_gpus_0123.sh (group A) and run_gpus_4567.sh (group B). Writes only to
# experiments/rebuttal/C2_heldout_fusion/{reports,logs}, outputs/rebuttal/C2_heldout_fusion/
# and results/rebuttal/C2_heldout_fusion/.

C2_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${C2_ROOT_DIR}"
# The main study's environment (PYTHONPATH=src, data defaults). It sets -euo pipefail;
# the run scripts handle exit codes themselves.
source scripts/env.sh
set +e

# Hugging Face cache: the server's shared cache, as for SAADI's runs.
export HF_HOME="${C2_HF_HOME:-/mnt/cache/taghavi}"
export SAADI_HF_CACHE_ROOT="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

# The study venv.
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHON="${PYTHON:-python3}"

export C2_EXP_DIR="${C2_ROOT_DIR}/experiments/rebuttal/C2_heldout_fusion"
mkdir -p "${C2_EXP_DIR}/logs" "${C2_EXP_DIR}/reports" outputs/rebuttal/C2_heldout_fusion

# matplotlib (heatmap PNGs only). If the venv lacks it, install it into C2's own pydeps,
# without numpy, so the venv is never modified and its numpy is the one imported.
C2_PYDEPS="${C2_ROOT_DIR}/outputs/rebuttal/C2_heldout_fusion/pydeps"
if ! "${PYTHON}" -c "import matplotlib" >/dev/null 2>&1; then
  if [[ ! -d "${C2_PYDEPS}/matplotlib" ]]; then
    echo "matplotlib not in the venv; installing it into ${C2_PYDEPS} (no numpy)"
    "${PYTHON}" -m pip install --quiet --no-deps --target "${C2_PYDEPS}" \
      matplotlib contourpy cycler fonttools kiwisolver pyparsing python-dateutil six \
      || echo "warning: matplotlib install failed; the heatmap PNGs will be skipped (values still in the tables)"
    if ! "${PYTHON}" -c "import PIL" >/dev/null 2>&1; then
      "${PYTHON}" -m pip install --quiet --no-deps --target "${C2_PYDEPS}" pillow || true
    fi
  fi
  export PYTHONPATH="${PYTHONPATH}:${C2_PYDEPS}"
fi
