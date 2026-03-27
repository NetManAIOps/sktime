#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash skills/time-series-sandbox/setup.sh
#   bash skills/time-series-sandbox/setup.sh /path/to/sktime

REPO_URL="https://github.com/NetManAIOps/sktime.git"
TARGET_DIR="${1:-sktime}"

if [[ ! -d "$TARGET_DIR/.git" ]]; then
  if [[ -e "$TARGET_DIR" ]]; then
    echo "Target path exists but is not a git repository: $TARGET_DIR" >&2
    exit 1
  fi
  git clone "$REPO_URL" "$TARGET_DIR"
fi

cd "$TARGET_DIR"

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[all_extras]"
python -c "import sktime; print('Time Series Sandbox ready:', sktime.__version__)"

echo "Setup complete. Activate with: source $PWD/.venv/bin/activate"
