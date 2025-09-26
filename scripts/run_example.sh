#!/usr/bin/env bash
set -euo pipefail

# Move to project root of the artifact
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${SCRIPT_DIR%/scripts}"
cd "$ARTIFACT_ROOT"

# Python venv
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

# Install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# Ensure outputs dir exists
mkdir -p outputs

# Run example with predefined styles and example logo
python test_naming_convention.py --logo ethicai
