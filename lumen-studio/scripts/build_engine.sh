#!/usr/bin/env bash
# Freeze the Python engine into a self-contained folder (engine/dist/lumen-engine)
# for packaging with electron-builder:  cd desktop && npm run dist
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
pip install "pyinstaller>=6.5"
cd engine
pyinstaller --noconfirm --name lumen-engine --onedir \
  --collect-all mediapipe --collect-all onnxruntime --collect-data cv2 \
  --collect-submodules numba --collect-submodules pyamg \
  --hidden-import uvicorn.logging --hidden-import uvicorn.loops.auto --hidden-import uvicorn.protocols.http.auto \
  lumen_entry.py
echo "Engine bundle: engine/dist/lumen-engine"
