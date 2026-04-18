#!/usr/bin/env bash
# router_federation_install.sh — Sprint 2 R3 installer for the fleet
# federation aggregator (Legion = data czar).
#
# What this does
#   Installs/uninstalls a user-crontab entry that runs the fleet
#   aggregator nightly. The aggregator pulls per-machine router shards
#   (via local fs or rsync-over-ssh) and writes a deduplicated,
#   partitioned corpus at {aggregate_dir}/{YYYY-MM-DD}.jsonl.gz.
#
# What this does NOT do
#   * Does not start/stop the SAGE daemon.
#   * Does not touch /etc/crontab or any system-wide cron dir.
#   * Does not manage SSH keys — relies on ssh-agent already loaded.
#   * Does not run as root. Everything lives in the invoking user's
#     crontab + home paths.
#
# Usage
#   scripts/router_federation_install.sh --enable-cron   # add user-cron entry
#   scripts/router_federation_install.sh --disable-cron  # remove it
#   scripts/router_federation_install.sh --run-now       # execute immediately
#   scripts/router_federation_install.sh --run-now --dry-run
#   scripts/router_federation_install.sh --config PATH   # point at custom config
#   scripts/router_federation_install.sh --status        # show install state
#
# Config
#   Default: $SAGE_DIR/sage/gateway/fleet_shards.json — lists peer
#   machines, their shard dirs, and transports. Ship a copy at deploy
#   time with real fleet paths filled in.
#
# Exit codes
#   0  success
#   1  prereq failure (python / module not importable)
#   2  usage error
#   3  config missing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAGE_DIR="${SAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
CONFIG_PATH_DEFAULT="${SAGE_DIR}/sage/gateway/fleet_shards.json"

MODE=""
CONFIG_PATH="$CONFIG_PATH_DEFAULT"
DRY_RUN=0
SCHEDULE_CRON="${ROUTER_FEDERATION_CRON:-0 2 * * *}"
CRON_TAG="# router-federation-aggregator (sprint2-r3)"

log() { echo "[router-federation] $*"; }

usage() {
    sed -n '2,40p' "$0"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --enable-cron)  MODE="enable" ;;
        --disable-cron) MODE="disable" ;;
        --run-now)      MODE="run" ;;
        --status)       MODE="status" ;;
        --dry-run)      DRY_RUN=1 ;;
        --config)       shift; CONFIG_PATH="${1:-}" ;;
        --config=*)     CONFIG_PATH="${1#*=}" ;;
        --cron)         shift; SCHEDULE_CRON="${1:-}" ;;
        --cron=*)       SCHEDULE_CRON="${1#*=}" ;;
        -h|--help)      usage; exit 0 ;;
        *)
            log "ERROR: unknown arg: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [ -z "$MODE" ]; then
    log "ERROR: must pass one of --enable-cron, --disable-cron, --run-now, --status" >&2
    exit 2
fi

# ── prereqs ────────────────────────────────────────────────────────

check_prereqs() {
    if [ -z "$PYTHON_BIN" ] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        log "ERROR: python3 not found in PATH" >&2
        return 1
    fi
    if ! PYTHONPATH="$SAGE_DIR" "$PYTHON_BIN" -c \
        "import sage.cognition.router.data.federation" >/dev/null 2>&1; then
        log "ERROR: sage.cognition.router.data.federation not importable "\
"from SAGE_DIR=$SAGE_DIR" >&2
        return 1
    fi
    return 0
}

# ── cron helpers ───────────────────────────────────────────────────

# Build the exact command the cron entry executes. Paths are absolute.
cron_command() {
    echo "cd \"$SAGE_DIR\" && PYTHONPATH=\"$SAGE_DIR\" \"$PYTHON_BIN\" "\
"-m sage.cognition.router.data.federation --run --config \"$CONFIG_PATH\" "\
">> \"$HOME/.sage-federation.log\" 2>&1"
}

cron_line() {
    echo "$SCHEDULE_CRON $(cron_command) $CRON_TAG"
}

# Read the current crontab, filtering OUT any prior router-federation
# entries (identified by the tag comment). Returns filtered crontab on
# stdout. Never fails even when no crontab exists.
current_crontab_filtered() {
    local current=""
    if current="$(crontab -l 2>/dev/null)"; then
        : # ok
    else
        current=""
    fi
    # Drop any existing entries tagged as ours.
    echo "$current" | grep -v -F "$CRON_TAG" || true
}

enable_cron() {
    check_prereqs || return 1
    if [ ! -f "$CONFIG_PATH" ]; then
        log "ERROR: config not found: $CONFIG_PATH"
        log "  create from template: $SAGE_DIR/sage/gateway/fleet_shards.json"
        return 3
    fi
    local line
    line="$(cron_line)"
    local filtered
    filtered="$(current_crontab_filtered)"
    log "adding cron line:"
    echo "  $line"
    {
        if [ -n "$filtered" ]; then
            echo "$filtered"
        fi
        echo "$line"
    } | crontab -
    log "enabled. Cron schedule: $SCHEDULE_CRON"
    log "log file: $HOME/.sage-federation.log"
}

disable_cron() {
    local filtered
    filtered="$(current_crontab_filtered)"
    if [ -z "$filtered" ]; then
        crontab -r 2>/dev/null || true
    else
        echo "$filtered" | crontab -
    fi
    log "disabled. Any router-federation cron entries removed."
}

status() {
    log "SAGE_DIR=$SAGE_DIR"
    log "config=$CONFIG_PATH"
    if [ -f "$CONFIG_PATH" ]; then
        log "  config present"
    else
        log "  config MISSING"
    fi
    log "python=$PYTHON_BIN"
    log "schedule=$SCHEDULE_CRON"
    log "cron entry:"
    if crontab -l 2>/dev/null | grep -F "$CRON_TAG" >/dev/null; then
        crontab -l 2>/dev/null | grep -F "$CRON_TAG" | sed 's/^/  /'
    else
        log "  (not installed)"
    fi
}

run_now() {
    check_prereqs || return 1
    if [ ! -f "$CONFIG_PATH" ]; then
        log "ERROR: config not found: $CONFIG_PATH"
        return 3
    fi
    local extra_flags=()
    if [ "$DRY_RUN" -eq 1 ]; then
        extra_flags+=("--dry-run")
    fi
    log "running aggregator now (config=$CONFIG_PATH)"
    cd "$SAGE_DIR"
    PYTHONPATH="$SAGE_DIR" "$PYTHON_BIN" \
        -m sage.cognition.router.data.federation \
        --run --config "$CONFIG_PATH" "${extra_flags[@]}"
}

case "$MODE" in
    enable)  enable_cron ;;
    disable) disable_cron ;;
    run)     run_now ;;
    status)  status ;;
esac
