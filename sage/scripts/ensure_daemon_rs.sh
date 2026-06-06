#!/bin/bash
# ensure_daemon_rs.sh — Rust sage-daemon equivalent of ensure_daemon.sh.
#
# Replaces the Python ensure pattern with a manual-start path for machines
# that have no systemd (WSL2 in particular). Run on machines that have
# cut over per SAGE/sage-rs/CUTOVER.md (Sprint 7+ env-var paths).
#
# Usage:
#   SAGE_MACHINE=nomad SAGE_MODEL=gemma4:e2b source sage/scripts/ensure_daemon_rs.sh
#
# What it does:
#   1. Checks /health on SAGE_PORT (default 8760)
#   2. If running, returns (Rust binary is compiled — no "stale code" check)
#   3. If not running, starts the compiled binary under nohup, waits for
#      /health to become ready
#
# After sourcing:
#   SAGE_DAEMON_VERSION   — version string from /health (e.g. "0.1.0")
#   SAGE_DAEMON_RUNNING   — "true" if daemon was already running
#   SAGE_DAEMON_UPDATED   — always "false" (no in-place restart; rebuild is
#                           the upgrade path for the compiled binary)
#
# Environment:
#   SAGE_PORT     — daemon port (default: 8760)
#   SAGE_MACHINE  — REQUIRED for non-sprout machines (Sprint 7 default is "sprout")
#   SAGE_MODEL    — model name for /chat dispatch (default per binary)
#   SAGE_DIR      — repo root (auto-detected)
#   SAGE_DAEMON_BIN — override path to sage-daemon binary

set -e

# --- Resolve paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAGE_DIR="${SAGE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SAGE_PORT="${SAGE_PORT:-8760}"
HEALTH_URL="http://localhost:${SAGE_PORT}/health"
DAEMON_BIN="${SAGE_DAEMON_BIN:-$SAGE_DIR/sage-rs/target/release/sage-daemon}"

SAGE_DAEMON_VERSION=""
SAGE_DAEMON_RUNNING="false"
SAGE_DAEMON_UPDATED="false"

log() { echo "[ensure_daemon_rs] $*"; }

check_health() {
    local resp
    resp=$(curl -s --max-time 3 "$HEALTH_URL" 2>/dev/null) || return 1
    echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('status')=='ok'" 2>/dev/null || return 1
    SAGE_DAEMON_VERSION=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('version','unknown'))" 2>/dev/null)
    return 0
}

wait_for_health() {
    local max_wait="${1:-30}"
    local waited=0
    log "Waiting for daemon to become healthy (max ${max_wait}s)..."
    while [ "$waited" -lt "$max_wait" ]; do
        if check_health; then
            log "Daemon healthy (version: $SAGE_DAEMON_VERSION)"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    log "ERROR: Daemon did not become healthy within ${max_wait}s"
    return 1
}

start_daemon() {
    log "Starting sage-daemon (Rust)..."
    if [ ! -x "$DAEMON_BIN" ]; then
        log "ERROR: $DAEMON_BIN not found or not executable. Build first:"
        log "  cd $SAGE_DIR/sage-rs && cargo build --release"
        return 1
    fi
    local LOG_DIR="$SAGE_DIR/sage/logs"
    mkdir -p "$LOG_DIR"
    local LOG_FILE="$LOG_DIR/daemon_rs_$(date +%Y%m%d_%H%M%S).log"
    # Env vars: SAGE_MACHINE / SAGE_MODEL already in this shell; nohup inherits them.
    cd "$SAGE_DIR"
    nohup "$DAEMON_BIN" > "$LOG_FILE" 2>&1 &
    local PID=$!
    log "Daemon PID: $PID, log: $LOG_FILE"
}

# --- Main ---

# Ensure the binary is built; if not, build it.
if [ ! -x "$DAEMON_BIN" ]; then
    log "Binary not built; running cargo build --release..."
    (cd "$SAGE_DIR/sage-rs" && cargo build --release 2>&1 | tail -3)
fi

if check_health; then
    SAGE_DAEMON_RUNNING="true"
    log "Daemon already running (version: $SAGE_DAEMON_VERSION)"
else
    SAGE_DAEMON_RUNNING="false"
    log "Daemon not running. Starting..."
    start_daemon
    wait_for_health 30
fi

log "Ready. daemon_version=$SAGE_DAEMON_VERSION running=$SAGE_DAEMON_RUNNING updated=$SAGE_DAEMON_UPDATED"

export SAGE_DAEMON_VERSION
export SAGE_DAEMON_RUNNING
export SAGE_DAEMON_UPDATED
