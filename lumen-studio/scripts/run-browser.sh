#!/usr/bin/env bash
# Run Lumen Studio in your own web browser (no Electron needed).
# The engine listens on 127.0.0.1 only - nothing is reachable from the network
# and nothing is sent to the internet. If the engine ever stops unexpectedly it
# is restarted automatically with the same session key, so the open page keeps working.
cd "$(dirname "$0")/.."
source .venv/bin/activate
PORT=${PORT:-8765}
TOKEN=$(python -c "import secrets;print(secrets.token_urlsafe(18))")
echo
echo "  Lumen Studio is running. Open this link in your browser:"
echo
echo "      http://127.0.0.1:${PORT}/?token=${TOKEN}"
echo
echo "  Keep this terminal open while you use the app. Press Ctrl+C to quit."
echo
trap 'echo; echo "Lumen Studio stopped."; exit 0' INT TERM
while true; do
  python -X faulthandler -m lumen_engine.server.main --port "$PORT" --static frontend/dist --token "$TOKEN"
  code=$?
  [ "$code" -eq 0 ] && break
  case $code in
    137) echo "!! The engine was killed by the system (usually: out of memory). Give the VM more RAM or use smaller images." ;;
    134|139) echo "!! The engine crashed (exit $code). The lines above show where; please send them if this repeats." ;;
    *) echo "!! The engine stopped unexpectedly (exit $code)." ;;
  esac
  echo "   Restarting in 2 seconds - the browser page will reconnect by itself."
  sleep 2
done
