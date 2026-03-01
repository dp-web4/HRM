#!/bin/bash
# ensure_daemon.sh — Verify SAGE daemon is running and up-to-date before a raising session.
#
# Usage:
#   source sage/scripts/ensure_daemon.sh           # Uses defaults
#   SAGE_PORT=8750 source sage/scripts/ensure_daemon.sh
#
# What it does:
#   1. Checks if daemon is running via /health
#   2. If running, compares daemon's git commit against current HEAD
#   3. If stale (code updated since daemon started), restarts daemon
#   4. If not running, pulls latest code and starts daemon
#   5. Waits for /health to respond before returning
#
# Environment variables:
#   SAGE_PORT     — Gateway port (default: 8750)
#   SAGE_MACHINE  — Machine name override (auto-detected if not set)
#   HRM_DIR       — Path to HRM repo root (auto-detected from script location)
#   SAGE_NO_BROWSER — Set to 1 to suppress dashboard auto-open (default: 1 for automated sessions)
#
# After sourcing, these variables are available:
#   SAGE_DAEMON_VERSION  — Git commit hash of the running daemon
#   SAGE_DAEMON_RUNNING  — "true" if daemon was already running, "false" if we started it
#   SAGE_DAEMON_UPDATED  — "true" if daemon was restarted due to stale code
#
# Platform support:
#   - Linux (systemd on Jetsons, manual elsewhere)
#   - macOS (launchd or manual)
#   - WSL2 (manual)

set -e

# --- Resolve paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HRM_DIR="${HRM_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SAGE_PORT="${SAGE_PORT:-8750}"
HEALTH_URL="http://localhost:${SAGE_PORT}/health"
export SAGE_NO_BROWSER="${SAGE_NO_BROWSER:-1}"

# --- Output variables (set by this script) ---
SAGE_DAEMON_VERSION=""
SAGE_DAEMON_RUNNING="false"
SAGE_DAEMON_UPDATED="false"

# --- Helpers ---
log() { echo "[ensure_daemon] $*"; }

get_current_head() {
    git -C "$HRM_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown"
}

check_health() {
    # Returns 0 if daemon is alive, 1 otherwise.
    # Sets SAGE_DAEMON_VERSION from the response.
    local resp
    resp=$(curl -s --max-time 3 "$HEALTH_URL" 2>/dev/null) || return 1
    # Check for valid JSON with status field
    echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='alive'" 2>/dev/null || return 1
    SAGE_DAEMON_VERSION=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('daemon_version','unknown'))" 2>/dev/null)
    return 0
}

wait_for_health() {
    local max_wait="${1:-60}"
    local waited=0
    log "Waiting for daemon to become healthy (max ${max_wait}s)..."
    while [ "$waited" -lt "$max_wait" ]; do
        if check_health; then
            log "Daemon healthy (version: $SAGE_DAEMON_VERSION)"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    log "ERROR: Daemon did not become healthy within ${max_wait}s"
    return 1
}

stop_daemon() {
    log "Stopping existing daemon..."
    # Try graceful SIGTERM first
    pkill -f "sage.gateway.sage_daemon" 2>/dev/null || true
    sleep 2
    # Force kill if still running
    if pgrep -f "sage.gateway.sage_daemon" >/dev/null 2>&1; then
        pkill -9 -f "sage.gateway.sage_daemon" 2>/dev/null || true
        sleep 1
    fi
}

start_daemon() {
    log "Starting SAGE daemon..."
    cd "$HRM_DIR"
    export PYTHONPATH="$HRM_DIR"
    # Detect python
    local PYTHON
    if command -v python3 >/dev/null 2>&1; then
        PYTHON="python3"
    elif [ -f /opt/homebrew/bin/python3 ]; then
        PYTHON="/opt/homebrew/bin/python3"
    else
        PYTHON="python"
    fi
    # Start daemon in background, redirect output to log
    local LOG_DIR="$HRM_DIR/sage/logs"
    mkdir -p "$LOG_DIR"
    local LOG_FILE="$LOG_DIR/daemon_$(date +%Y%m%d_%H%M%S).log"
    nohup $PYTHON -m sage.gateway.sage_daemon > "$LOG_FILE" 2>&1 &
    local PID=$!
    log "Daemon PID: $PID, log: $LOG_FILE"
}

pull_latest() {
    log "Pulling latest code..."
    cd "$HRM_DIR"
    git pull --rebase --quiet 2>/dev/null || {
        log "WARN: git pull failed (merge conflict?), continuing with current code"
    }
}

# --- Main logic ---

CURRENT_HEAD=$(get_current_head)
log "Current code HEAD: $CURRENT_HEAD"

if check_health; then
    # Daemon is running — check if it's up to date
    SAGE_DAEMON_RUNNING="true"
    log "Daemon running (version: $SAGE_DAEMON_VERSION)"

    if [ "$SAGE_DAEMON_VERSION" = "$CURRENT_HEAD" ]; then
        log "Daemon is up to date. Proceeding."
    elif [ "$SAGE_DAEMON_VERSION" = "unknown" ]; then
        # Old daemon without version support — restart to get versioned one
        log "Daemon has no version info (pre-versioning). Restarting..."
        stop_daemon
        pull_latest
        CURRENT_HEAD=$(get_current_head)
        start_daemon
        wait_for_health 90
        SAGE_DAEMON_UPDATED="true"
    else
        # Daemon is stale — check if there are actually new commits
        log "Daemon version ($SAGE_DAEMON_VERSION) != HEAD ($CURRENT_HEAD)"
        log "Pulling latest and restarting daemon..."
        stop_daemon
        pull_latest
        CURRENT_HEAD=$(get_current_head)
        start_daemon
        wait_for_health 90
        SAGE_DAEMON_UPDATED="true"
    fi
else
    # Daemon is not running — pull latest and start
    SAGE_DAEMON_RUNNING="false"
    log "Daemon not running. Starting fresh..."
    pull_latest
    CURRENT_HEAD=$(get_current_head)
    start_daemon
    wait_for_health 120  # First start can take longer (model loading)
fi

# Final state
log "Ready. daemon_version=$SAGE_DAEMON_VERSION running=$SAGE_DAEMON_RUNNING updated=$SAGE_DAEMON_UPDATED"

# Export for downstream scripts
export SAGE_DAEMON_VERSION
export SAGE_DAEMON_RUNNING
export SAGE_DAEMON_UPDATED
