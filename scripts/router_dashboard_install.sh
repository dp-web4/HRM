#!/usr/bin/env bash
# router_dashboard_install.sh — Sprint 2 R4 scheduler installer.
#
# Installs (or removes) a user-scope cron entry that runs
# router_dashboard_render.py hourly in --quiet mode, emitting markdown to
# the shared-context dashboard path. Stderr only fires on SNARC drift
# alerts (PRD §4.7.G), so an inbox full of mails from this cron is a
# signal, not noise.
#
# What this does
#   * Adds/removes an hourly user-crontab entry (no sudo, no root cron).
#   * Can run the render immediately via --run-now.
#   * Is idempotent — re-running install replaces the existing line.
#
# What this does NOT do
#   * Does not install system cron or systemd timers (pool-consistency —
#     every machine has its user crontab).
#   * Does not install python or any dependencies.
#   * Does not modify ~/.bashrc or shell rc files.
#
# Usage
#   scripts/router_dashboard_install.sh --enable-cron
#   scripts/router_dashboard_install.sh --disable-cron
#   scripts/router_dashboard_install.sh --run-now
#   scripts/router_dashboard_install.sh --enable-cron --output PATH
#   scripts/router_dashboard_install.sh --enable-cron --base-dir PATH
#
# Env overrides
#   SAGE_DIR              SAGE checkout root (default: inferred from script path)
#   SAGE_ROUTER_DATA_DIR  dataset root (default: matches pipeline installer)
#   SAGE_PYTHON           python interpreter (default: python3 in PATH)
#
# Exit codes
#   0  success
#   1  prereq failure (python / crontab / render script missing)
#   2  usage error

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAGE_DIR="${SAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RENDER_SCRIPT="${SAGE_DIR}/scripts/router_dashboard_render.py"

# ── Defaults matching router_pipeline_install.sh ─────────────────────

default_data_dir() {
    local candidates=(
        "${HOME}/ai-workspace/private-context/training-data/router"
        "${HOME}/ai-agents/private-context/training-data/router"
        "/mnt/c/exe/projects/ai-agents/private-context/training-data/router"
    )
    for c in "${candidates[@]}"; do
        local grandparent
        grandparent="$(dirname "$(dirname "$c")")"
        if [ -d "$grandparent" ]; then
            echo "$c"
            return
        fi
    done
    echo "${candidates[0]}"
}

default_output() {
    # Matches router_dashboard_render.py's _DEFAULT_OUTPUT resolution:
    # shared-context lives as a sibling of the SAGE checkout.
    local candidates=(
        "$(dirname "$SAGE_DIR")/shared-context/arc-agi-3/phase2/brain-arch/router-pipeline-dashboard.md"
        "${HOME}/ai-workspace/shared-context/arc-agi-3/phase2/brain-arch/router-pipeline-dashboard.md"
        "${HOME}/ai-agents/shared-context/arc-agi-3/phase2/brain-arch/router-pipeline-dashboard.md"
    )
    for c in "${candidates[@]}"; do
        local grandparent
        grandparent="$(dirname "$(dirname "$c")")"
        if [ -d "$grandparent" ] || [ -d "$(dirname "$grandparent")" ]; then
            echo "$c"
            return
        fi
    done
    echo "${candidates[0]}"
}

ENABLE=0
DISABLE=0
RUN_NOW=0
DATA_DIR="${SAGE_ROUTER_DATA_DIR:-$(default_data_dir)}"
OUTPUT_PATH=""
PYTHON_BIN="${SAGE_PYTHON:-$(command -v python3 || true)}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --enable-cron) ENABLE=1 ;;
        --disable-cron) DISABLE=1 ;;
        --run-now) RUN_NOW=1 ;;
        --output) shift; OUTPUT_PATH="${1:-}" ;;
        --output=*) OUTPUT_PATH="${1#*=}" ;;
        --base-dir) shift; DATA_DIR="${1:-}" ;;
        --base-dir=*) DATA_DIR="${1#*=}" ;;
        --python) shift; PYTHON_BIN="${1:-}" ;;
        --python=*) PYTHON_BIN="${1#*=}" ;;
        -h|--help)
            sed -n '2,36p' "$0"
            exit 0
            ;;
        *)
            echo "[dashboard-install] unknown arg: $1" >&2
            exit 2
            ;;
    esac
    shift
done

if [ -z "$OUTPUT_PATH" ]; then
    OUTPUT_PATH="$(default_output)"
fi

log() { echo "[dashboard-install] $*"; }

# ── Prereq checks ────────────────────────────────────────────────────

check_prereqs() {
    if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
        log "ERROR: python interpreter not found (tried SAGE_PYTHON, python3). Pass --python PATH to override." >&2
        return 1
    fi
    if [ ! -f "$RENDER_SCRIPT" ]; then
        log "ERROR: render script not found: $RENDER_SCRIPT" >&2
        return 1
    fi
    if [ "$ENABLE" -eq 1 ] || [ "$DISABLE" -eq 1 ]; then
        if ! command -v crontab >/dev/null 2>&1; then
            log "ERROR: crontab(1) not found in PATH (needed for cron install)" >&2
            return 1
        fi
    fi
    return 0
}

# ── Cron line management ─────────────────────────────────────────────

CRON_MARKER="# router_dashboard_install.sh — Sprint 2 R4"

cron_line() {
    # Hourly at :00. --quiet so only drift alerts ever reach stderr /
    # MAILTO. Bake in base-dir + output so the cron needs no env.
    printf '0 * * * * %s %s --base-dir %s --output %s --quiet %s\n' \
        "$PYTHON_BIN" "$RENDER_SCRIPT" "$DATA_DIR" "$OUTPUT_PATH" "$CRON_MARKER"
}

# Strip any previous lines we installed (match by marker).
strip_ours() {
    local tmp
    tmp="$(mktemp)"
    if crontab -l 2>/dev/null | grep -v -F "$CRON_MARKER" > "$tmp"; then
        :
    fi
    # Empty crontab handling: if `crontab -l` returns non-zero (no
    # crontab), the redirect leaves $tmp empty — that's fine.
    crontab "$tmp"
    rm -f "$tmp"
}

install_cron() {
    log "installing hourly cron (quiet mode — only drift alerts reach stderr)"
    log "  python      : $PYTHON_BIN"
    log "  render      : $RENDER_SCRIPT"
    log "  base-dir    : $DATA_DIR"
    log "  output      : $OUTPUT_PATH"
    mkdir -p "$(dirname "$OUTPUT_PATH")"
    # Remove any existing entry first so this is idempotent.
    strip_ours
    local tmp
    tmp="$(mktemp)"
    (crontab -l 2>/dev/null || true) > "$tmp"
    cron_line >> "$tmp"
    crontab "$tmp"
    rm -f "$tmp"
    log "done. Current crontab lines for the dashboard:"
    crontab -l 2>/dev/null | grep -F "$CRON_MARKER" || true
}

uninstall_cron() {
    log "removing hourly cron (if present)"
    strip_ours
    log "done. Crontab no longer contains: $CRON_MARKER"
}

# ── Run-now ──────────────────────────────────────────────────────────

run_now() {
    log "running render immediately (--quiet)"
    log "  base-dir : $DATA_DIR"
    log "  output   : $OUTPUT_PATH"
    mkdir -p "$(dirname "$OUTPUT_PATH")"
    "$PYTHON_BIN" "$RENDER_SCRIPT" \
        --base-dir "$DATA_DIR" \
        --output "$OUTPUT_PATH" \
        --quiet
    log "render complete."
}

# ── Main ─────────────────────────────────────────────────────────────

if [ "$ENABLE" -eq 0 ] && [ "$DISABLE" -eq 0 ] && [ "$RUN_NOW" -eq 0 ]; then
    log "no action requested. Pass --enable-cron, --disable-cron, or --run-now."
    log "  Current render : $RENDER_SCRIPT"
    log "  Default base   : $DATA_DIR"
    log "  Default output : $OUTPUT_PATH"
    exit 0
fi

if ! check_prereqs; then
    exit 1
fi

if [ "$DISABLE" -eq 1 ]; then
    uninstall_cron
fi

if [ "$ENABLE" -eq 1 ]; then
    install_cron
fi

if [ "$RUN_NOW" -eq 1 ]; then
    run_now
fi

exit 0
