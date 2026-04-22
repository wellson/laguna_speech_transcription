#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install sounddevice numpy scipy mlx-whisper deep-translator
else
  source .venv/bin/activate
fi

python test.py
