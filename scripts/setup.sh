#!/usr/bin/env bash
# Save as scripts/setup.sh; run from any directory.
# Requires Linux x86_64 and Python 3.12 already available.
# On UConn: module load python/3.12.14
# Optional: PYTHON=/path/to/python3.12 VENV=/path/to/venv bash scripts/setup.sh
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
venv_dir=${VENV:-"$repo_dir/venv"}

# Avoid importing packages from an old checkout or user-site installation.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

if [[ ! -x "$venv_dir/bin/python" ]]; then
    "${PYTHON:-python3.12}" -m venv "$venv_dir"
fi
source "$venv_dir/bin/activate"

python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'

python -m pip install --upgrade pip
python -m pip install --only-binary=:all: \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    'torch==2.10.0+cu126' \
    'https://aihwkit-gpu-demo.s3.us-east.cloud-object-storage.appdomain.cloud/aihwkit-1.1.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
python -m pip check

python -I - <<'PY'
import torch, torchvision, aihwkit
from aihwkit.simulator import rpu_base
assert torch.__version__ == "2.10.0+cu126"
assert aihwkit.__version__ == "1.1.0"
assert rpu_base.cuda.is_compiled(), "AIHWKIT has no CUDA backend"
print("Installed:", aihwkit.__version__, torch.__version__, torchvision.__version__)
print("Imports passed; GPU execution must be tested in a GPU allocation.")
PY

echo "Ready: $py"

