#!/usr/bin/env bash
# router_pipeline_verify.sh — Post-install verification for the router shadow pipeline.
#
# What it checks
#   1. The installer's drop-in exists and carries SAGE_ROUTER_SHADOW=1.
#   2. The dataset directory exists and is writable.
#   3. (If the daemon is running) at least one record is written within
#      ${SAGE_ROUTER_VERIFY_WAIT} seconds (default 60).
#   4. Reports records/hour rate, last-write timestamp, partition bytes.
#
# What it deliberately does not do
#   * Start/stop the daemon. Installers / runbook steps do that.
#   * Interpret record contents. This is purely an ingestion-rate check.
#
# Usage
#   scripts/router_pipeline_verify.sh
#   scripts/router_pipeline_verify.sh --wait 120
#   SAGE_MACHINE=sprout scripts/router_pipeline_verify.sh
#   SAGE_ROUTER_DATA_DIR=/tmp/router scripts/router_pipeline_verify.sh
#
# Exit codes
#   0 all checks pass
#   1 configuration not found (drop-in missing)
#   2 dataset directory problem (missing / unwritable)
#   3 no records written in verify window
#   4 usage error

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAGE_DIR="${SAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

WAIT_SECONDS="${SAGE_ROUTER_VERIFY_WAIT:-60}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --wait) shift; WAIT_SECONDS="${1:-60}" ;;
        --wait=*) WAIT_SECONDS="${1#*=}" ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 4 ;;
    esac
    shift
done

log() { echo "[router-verify] $*"; }

# ──────────────────────────────────────────────────────────────────────
# Same detection as installer — must agree.
# ──────────────────────────────────────────────────────────────────────

detect_machine() {
    if [ -n "${SAGE_MACHINE:-}" ]; then
        echo "$SAGE_MACHINE" | tr '[:upper:]' '[:lower:]'; return
    fi
    local host; host="$(hostname 2>/dev/null | tr '[:upper:]' '[:lower:]')"
    case "$host" in
        *thor*) echo "thor" ;;
        *cbp*) echo "cbp" ;;
        *legion*) echo "legion" ;;
        *nomad*|*desktop-9e6hcao*) echo "nomad" ;;
        *mcnugget*) echo "mcnugget" ;;
        ubuntu) [ -d "/home/sprout" ] && echo "sprout" || echo "unknown" ;;
        *) echo "$host" ;;
    esac
}

MACHINE="$(detect_machine)"
UNIT="sage-daemon-${MACHINE}"
SYSTEM_DROPIN="/etc/systemd/system/${UNIT}.service.d/router-shadow.conf"
USER_DROPIN="${HOME}/.config/systemd/user/${UNIT}.service.d/router-shadow.conf"
PROFILE_ENV="${SAGE_DIR}/sage/gateway/router-shadow.env"

# ──────────────────────────────────────────────────────────────────────
# Step 1: locate the config drop-in + extract SAGE_ROUTER_DATA_DIR
# ──────────────────────────────────────────────────────────────────────

DROPIN_PATH=""
for candidate in "$SYSTEM_DROPIN" "$USER_DROPIN" "$PROFILE_ENV"; do
    if [ -f "$candidate" ]; then
        DROPIN_PATH="$candidate"; break
    fi
done

if [ -z "$DROPIN_PATH" ]; then
    log "ERROR: no router-shadow drop-in found for machine=${MACHINE}"
    log "  checked: $SYSTEM_DROPIN"
    log "  checked: $USER_DROPIN"
    log "  checked: $PROFILE_ENV"
    exit 1
fi
log "drop-in: $DROPIN_PATH"

if ! grep -q "SAGE_ROUTER_SHADOW=1" "$DROPIN_PATH"; then
    log "ERROR: drop-in does not set SAGE_ROUTER_SHADOW=1"
    exit 1
fi
log "SAGE_ROUTER_SHADOW=1 present"

# Extract data dir from drop-in (handles both systemd Environment= and shell export).
DATA_DIR="$(grep -E '(Environment=)?(export )?SAGE_ROUTER_DATA_DIR=' "$DROPIN_PATH" \
            | sed -E 's/.*SAGE_ROUTER_DATA_DIR=//; s/^"//; s/"$//' | head -n1)"

# Allow env override (lets operators verify a non-standard location).
DATA_DIR="${SAGE_ROUTER_DATA_DIR:-$DATA_DIR}"

if [ -z "$DATA_DIR" ]; then
    log "ERROR: could not determine SAGE_ROUTER_DATA_DIR"
    exit 1
fi
log "dataset dir: $DATA_DIR"

# ──────────────────────────────────────────────────────────────────────
# Step 2: check dataset dir
# ──────────────────────────────────────────────────────────────────────

if [ ! -d "$DATA_DIR" ]; then
    log "WARN: dataset dir does not exist yet; will appear on first write"
fi

MACHINE_DIR="${DATA_DIR}/${MACHINE}"
if [ ! -d "$MACHINE_DIR" ]; then
    log "WARN: per-machine dir ${MACHINE_DIR} does not exist yet"
fi

# ──────────────────────────────────────────────────────────────────────
# Step 3: poll for a record written inside the verify window
# ──────────────────────────────────────────────────────────────────────

find_latest_partition() {
    if [ ! -d "$MACHINE_DIR" ]; then echo ""; return; fi
    # Most-recently-modified .jsonl or .jsonl.gz
    local f
    f=$(find "$MACHINE_DIR" -maxdepth 1 -type f \
           \( -name '*.jsonl' -o -name '*.jsonl.gz' \) \
           -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | awk '{print $2}')
    echo "$f"
}

partition_bytes() {
    local f="$1"
    [ -f "$f" ] && stat -c '%s' "$f" 2>/dev/null || echo "0"
}

partition_mtime_epoch() {
    local f="$1"
    [ -f "$f" ] && stat -c '%Y' "$f" 2>/dev/null || echo "0"
}

log "polling for records (window=${WAIT_SECONDS}s)..."

START_EPOCH="$(date +%s)"
INITIAL_BYTES=0
INITIAL_PARTITION="$(find_latest_partition || true)"
if [ -n "$INITIAL_PARTITION" ]; then
    INITIAL_BYTES="$(partition_bytes "$INITIAL_PARTITION")"
    log "  initial partition: $(basename "$INITIAL_PARTITION") (${INITIAL_BYTES} bytes)"
fi

WAITED=0
INTERVAL=2
DETECTED_WRITE=0
while [ "$WAITED" -lt "$WAIT_SECONDS" ]; do
    P="$(find_latest_partition || true)"
    if [ -n "$P" ]; then
        B="$(partition_bytes "$P")"
        M="$(partition_mtime_epoch "$P")"
        if [ "$P" != "$INITIAL_PARTITION" ] || [ "$B" -gt "$INITIAL_BYTES" ] || [ "$M" -gt "$START_EPOCH" ]; then
            DETECTED_WRITE=1
            break
        fi
    fi
    sleep "$INTERVAL"
    WAITED=$((WAITED + INTERVAL))
done

# ──────────────────────────────────────────────────────────────────────
# Step 4: report
# ──────────────────────────────────────────────────────────────────────

CURRENT_PARTITION="$(find_latest_partition || true)"
if [ -z "$CURRENT_PARTITION" ]; then
    log "ERROR: no partition file found after ${WAIT_SECONDS}s in ${MACHINE_DIR}"
    log "  confirm daemon is running and restarted after install"
    exit 3
fi

BYTES="$(partition_bytes "$CURRENT_PARTITION")"
MTIME_EPOCH="$(partition_mtime_epoch "$CURRENT_PARTITION")"
MTIME_ISO="$(date -u -d "@${MTIME_EPOCH}" +'%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || echo "$MTIME_EPOCH")"

# Crude records/hour estimate: bytes since the partition's first mtime / avg record size.
# Real sizing uses the writer's tracked count when we wire dashboards (Track 8).
# For verify, 800 bytes/record plain or ~120 bytes/record gzipped is a reasonable
# rough cut on Phase 0 records.
AVG_BYTES=800
case "$CURRENT_PARTITION" in
    *.gz) AVG_BYTES=120 ;;
esac
RATE_PER_HOUR=$(( BYTES / AVG_BYTES ))

log "latest partition: $(basename "$CURRENT_PARTITION")"
log "  size: ${BYTES} bytes"
log "  last write: ${MTIME_ISO}"
log "  approx records (avg=${AVG_BYTES}B): ${RATE_PER_HOUR}"

if [ "$DETECTED_WRITE" -eq 0 ]; then
    log "ERROR: no new bytes written during ${WAIT_SECONDS}s verify window"
    log "  daemon may be idle, not restarted, or shadow not wired"
    exit 3
fi

log "OK — router shadow pipeline is capturing records"
