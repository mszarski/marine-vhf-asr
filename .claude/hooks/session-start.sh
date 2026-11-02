#!/bin/bash
# Session startup hook for Claude Code on the web
set -euo pipefail

# Only run in remote Claude Code environment
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
    exit 0
fi

echo "Setting up development environment..."

# GPU training deps (marine_vhf_asr/requirements-train.txt) are only needed in HF Jobs
pip install -r "$CLAUDE_PROJECT_DIR/marine_vhf_asr/requirements.txt" pytest

echo "Session startup complete"
