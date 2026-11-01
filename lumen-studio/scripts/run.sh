#!/usr/bin/env bash
# Start the desktop application (development layout).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/desktop"; export LUMEN_PYTHON="$ROOT/.venv/bin/python"; if command -v electron >/dev/null; then exec electron . "$@"; else exec npx electron . "$@"; fi
