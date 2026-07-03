#!/usr/bin/env bash
# Create/refresh the `sim2real` conda env and install the package (editable).
# Idempotent: safe to re-run.
set -euo pipefail

ENV="${CONDA_ENV:-sim2real}"
PYVER="${PYVER:-3.11}"
CONDA="${CONDA:-conda}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v "$CONDA" >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda or set CONDA=/path/to/conda." >&2
  exit 1
fi

if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV"; then
  echo ">> creating conda env '$ENV' (python $PYVER)"
  "$CONDA" create -y -n "$ENV" "python=$PYVER"
else
  echo ">> conda env '$ENV' already exists"
fi

# Use `python -m pip` explicitly: `conda run pip` can resolve to the system pip.
echo ">> installing package into '$ENV'"
"$CONDA" run -n "$ENV" python -m pip install --upgrade pip
"$CONDA" run -n "$ENV" python -m pip install -e "${HERE}[dev]"

echo
echo ">> done. Next:"
echo "     conda activate $ENV"
echo "     make smoke        # verify the pipeline"
echo "     make train        # full training run"
