#!/usr/bin/env bash
# router_pipeline_disk_monitor.sh — Periodic disk-usage check for the router
# shadow dataset (Phase 0 Track 7).
#
# Threshold rationale (per sprint doc + Track 4 sizing):
#   The sprint doc cites 100 MB/day as the original alert threshold. Track 4
#   landed with gzip on by default — real partitions on synthetic data
#   compress ~70× (and noticeably less on real records, but still a large
#   factor). Under gzip, "normal" traffic should sit comfortably under
#   10 MB/day/machine.
#
#   This monitor ships two thresholds:
#     WARN  = 50 MB/day   (notable but not alarming under gzip)
#     ALERT = 200 MB/day  (hard — something is wrong: gzip off? loop stuck?)
#
#   Both are overridable via flags or env vars so fleet-wide monitoring can
#   tune per machine.
#
# Usage
#   scripts/router_pipeline_disk_monitor.sh
#   scripts/router_pipeline_disk_monitor.sh --json
#   scripts/router_pipeline_disk_monitor.sh --warn-mb 50 --alert-mb 200
#   scripts/router_pipeline_disk_monitor.sh --data-dir /var/sage/router
#
# Cron example (every 15 min, JSON to a log file):
#   */15 * * * * /path/to/router_pipeline_disk_monitor.sh --json \
#                >> /var/log/router-disk-monitor.log 2>&1
#
# Exit codes
#   0 OK or WARN
#   2 ALERT threshold exceeded
#   3 configuration/dataset not found
#   4 usage error

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAGE_DIR="${SAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

WARN_MB="${SAGE_ROUTER_WARN_MB:-50}"
ALERT_MB="${SAGE_ROUTER_ALERT_MB:-200}"
JSON_OUT=0
DATA_DIR_OVERRIDE=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --json) JSON_OUT=1 ;;
        --warn-mb) shift; WARN_MB="${1:-50}" ;;
        --warn-mb=*) WARN_MB="${1#*=}" ;;
        --alert-mb) shift; ALERT_MB="${1:-200}" ;;
        --alert-mb=*) ALERT_MB="${1#*=}" ;;
        --data-dir) shift; DATA_DIR_OVERRIDE="${1:-}" ;;
        --data-dir=*) DATA_DIR_OVERRIDE="${1#*=}" ;;
        -h|--help) sed -n '2,35p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 4 ;;
    esac
    shift
done

log() {
    if [ "$JSON_OUT" -eq 0 ]; then
        echo "[router-disk] $*"
    fi
}

# ──────────────────────────────────────────────────────────────────────
# Machine detection
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

# ──────────────────────────────────────────────────────────────────────
# Locate DATA_DIR (same resolution as verify script, minus interactive noise)
# ──────────────────────────────────────────────────────────────────────

if [ -n "$DATA_DIR_OVERRIDE" ]; then
    DATA_DIR="$DATA_DIR_OVERRIDE"
elif [ -n "${SAGE_ROUTER_DATA_DIR:-}" ]; then
    DATA_DIR="$SAGE_ROUTER_DATA_DIR"
else
    SYSTEM_DROPIN="/etc/systemd/system/${UNIT}.service.d/router-shadow.conf"
    USER_DROPIN="${HOME}/.config/systemd/user/${UNIT}.service.d/router-shadow.conf"
    PROFILE_ENV="${SAGE_DIR}/sage/gateway/router-shadow.env"

    DATA_DIR=""
    for c in "$SYSTEM_DROPIN" "$USER_DROPIN" "$PROFILE_ENV"; do
        if [ -f "$c" ]; then
            DATA_DIR="$(grep -E '(Environment=)?(export )?SAGE_ROUTER_DATA_DIR=' "$c" \
                        | sed -E 's/.*SAGE_ROUTER_DATA_DIR=//; s/^"//; s/"$//' | head -n1)"
            [ -n "$DATA_DIR" ] && break
        fi
    done
fi

if [ -z "$DATA_DIR" ]; then
    log "ERROR: cannot resolve SAGE_ROUTER_DATA_DIR for machine=${MACHINE}"
    exit 3
fi

MACHINE_DIR="${DATA_DIR}/${MACHINE}"

# ──────────────────────────────────────────────────────────────────────
# Measure today's partition
# ──────────────────────────────────────────────────────────────────────

TODAY="$(date -u +%Y-%m-%d)"
PARTITIONS=()
if [ -d "$MACHINE_DIR" ]; then
    while IFS= read -r p; do
        [ -n "$p" ] && PARTITIONS+=("$p")
    done < <(find "$MACHINE_DIR" -maxdepth 1 -type f -name "${TODAY}.jsonl*" 2>/dev/null)
fi

TODAY_BYTES=0
for p in "${PARTITIONS[@]}"; do
    sz="$(stat -c '%s' "$p" 2>/dev/null || echo 0)"
    TODAY_BYTES=$((TODAY_BYTES + sz))
done

# Grand total (all partitions for this machine, not just today)
TOTAL_BYTES=0
if [ -d "$MACHINE_DIR" ]; then
    TOTAL_BYTES="$(find "$MACHINE_DIR" -maxdepth 1 -type f \
                    \( -name '*.jsonl' -o -name '*.jsonl.gz' \) \
                    -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')"
fi

TODAY_MB=$(( TODAY_BYTES / 1024 / 1024 ))
TOTAL_MB=$(( TOTAL_BYTES / 1024 / 1024 ))

# Any gzip active today?
GZIP_ACTIVE="unknown"
for p in "${PARTITIONS[@]}"; do
    case "$p" in
        *.jsonl.gz) GZIP_ACTIVE="yes"; break ;;
        *.jsonl)    GZIP_ACTIVE="no" ;;
    esac
done

# ──────────────────────────────────────────────────────────────────────
# Determine severity
# ──────────────────────────────────────────────────────────────────────

LEVEL="OK"
EXIT=0
if [ "$TODAY_MB" -ge "$ALERT_MB" ]; then
    LEVEL="ALERT"
    EXIT=2
elif [ "$TODAY_MB" -ge "$WARN_MB" ]; then
    LEVEL="WARN"
    EXIT=0
fi

# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

if [ "$JSON_OUT" -eq 1 ]; then
    # Compact single-line JSON — suitable for log aggregation.
    printf '{"timestamp":"%s","machine":"%s","data_dir":"%s","today":"%s","today_mb":%d,"total_mb":%d,"warn_mb":%d,"alert_mb":%d,"level":"%s","gzip_active":"%s","partition_count":%d}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$MACHINE" \
        "$DATA_DIR" \
        "$TODAY" \
        "$TODAY_MB" \
        "$TOTAL_MB" \
        "$WARN_MB" \
        "$ALERT_MB" \
        "$LEVEL" \
        "$GZIP_ACTIVE" \
        "${#PARTITIONS[@]}"
else
    log "machine: $MACHINE"
    log "data dir: $DATA_DIR"
    log "today's partition ($TODAY): ${TODAY_MB} MB across ${#PARTITIONS[@]} file(s)"
    log "all-time on this machine: ${TOTAL_MB} MB"
    log "gzip active: $GZIP_ACTIVE"
    log "thresholds: warn=${WARN_MB}MB alert=${ALERT_MB}MB"
    log "LEVEL=${LEVEL}"
fi

exit "$EXIT"
