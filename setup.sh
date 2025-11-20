#!/usr/bin/env bash
set -euo pipefail

# run from repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# configuration
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ_FILE="${REQ_FILE:-requirements.txt}"

echo "setting up environment in '$VENV_DIR'..."

# create venv if does not exist
if [ ! -d "$VENV_DIR" ]; then
  echo "creating virtualenv using $PYTHON_BIN..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# activate venv
source "$VENV_DIR/bin/activate"

# upgrade
python -m pip install --upgrade pip

# install requirements if exists
if [ -f "$REQ_FILE" ]; then
  echo "installing dependencies from $REQ_FILE..."
  pip install -r "$REQ_FILE"
else
  echo "no $REQ_FILE found; skipping dependency install"
fi

echo "setup complete"

# keep terminal open if interactive
if [ -t 0 ]; then
  read -rp "press Enter to close"
fi
