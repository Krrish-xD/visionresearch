#!/usr/bin/env bash
# Start VisionResearch backend and frontend together silently.
# Press Ctrl+C to stop both.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔬 Starting VisionResearch (Quiet Mode)..."

# Ensure we clean up child processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping VisionResearch..."
    if [ -n "${BACKEND_PID:-}" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo "✨ Cleaned up successfully."
}

trap cleanup EXIT INT TERM

# Start backend quietly
cd "$SCRIPT_DIR/backend"
uv run uvicorn main:app --host 127.0.0.1 --port 8000 --log-level warning > /dev/null 2>&1 &
BACKEND_PID=$!

# Start frontend quietly
cd "$SCRIPT_DIR/frontend"
npm run dev > /dev/null 2>&1 &
FRONTEND_PID=$!

echo "🚀 VisionResearch is running!"
echo "   - Backend:  http://127.0.0.1:8000"
echo "   - Frontend: http://127.0.0.1:5173"
echo "   Press Ctrl+C to shutdown."

# Keep the script running to wait for Ctrl+C
wait
