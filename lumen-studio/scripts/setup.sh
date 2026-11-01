#!/usr/bin/env bash
# One-time setup for Lumen Studio on macOS / Linux.
#   ./scripts/setup.sh            CPU
#   ./scripts/setup.sh --gpu      NVIDIA CUDA (Linux/Windows with CUDA 12 + cuDNN 9)
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=0; BROWSER=0; COMPAT=0
for arg in "$@"; do
  case "$arg" in
    --gpu) GPU=1 ;;
    --browser) BROWSER=1 ;;
    --compat) COMPAT=1; BROWSER=1 ;;
    *) echo "Usage: $0 [--gpu] [--browser] [--compat]" >&2; exit 2 ;;
  esac
done

ok_python() { "$1" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; }
PY=${PYTHON:-python3}
if ! ok_python "$PY"; then
  # Rolling-release distros (Arch, Fedora…) ship a newer Python than the ML
  # wheels support. uv downloads a private Python 3.12 without touching the system.
  if command -v uv >/dev/null; then
    echo "==> System Python is not 3.10-3.12; fetching Python 3.12 with uv"
    uv python install 3.12
    PY=$(uv python find 3.12)
  else
    echo "Python 3.10-3.12 is required (or install 'uv' and re-run)."; exit 1
  fi
fi
echo "==> Creating Python environment (.venv)"
"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel >/dev/null
pip install -e "engine[dev]"
if [[ $GPU == 1 ]]; then
  pip uninstall -y onnxruntime && pip install "onnxruntime-gpu>=1.17"
fi

echo "==> Building the user interface"
(cd frontend && npm ci --no-audit --no-fund && npm run build)
if [[ $BROWSER == 0 ]]; then
echo "==> Installing the desktop shell"
(cd desktop && npm ci --no-audit --no-fund && \
  { [ -f node_modules/electron/path.txt ] || node node_modules/electron/install.js || echo "(npm Electron download failed; the system electron package will be used)"; })  # newer npm blocks install scripts

fi
echo "==> Downloading models (one-time, then fully offline)"
if [[ $COMPAT == 1 ]]; then
  # The .task is only a checksum-verified conversion source. All runtime
  # inference uses ONNX; no LBF/MediaPipe/TFLite interpreter is loaded.
  python scripts/download_models.py --only face-landmarker face-mesh-onnx face-detector-yunet face-parsing depth-anything-v2-small segment-human segment-silueta
else
  python scripts/download_models.py
fi

# Optional learned inpainting (MI-GAN ~30 MB, LaMa ~210 MB, hosted on Hugging Face): better
# fills when removing objects and for what is hidden behind people or outside the frame in 3D
# scenes. This step never fails the setup: --optional only warns about models that cannot be
# downloaded, the built-in PatchMatch inpainting is used without them, and they can be
# installed later from the Models page.
echo "==> Optional: learned inpainting models (MI-GAN, LaMa) - setup continues if this fails"
python scripts/download_models.py --optional --only migan lama || true

echo
if [[ $BROWSER == 1 ]]; then
  echo "Setup complete. Run: LUMEN_ASSUME_NO_AVX=1 LUMEN_FACE_BACKEND=opencv ./scripts/run-browser.sh"
else
  echo "Setup complete. Start Lumen Studio with: ./scripts/run.sh"
fi
