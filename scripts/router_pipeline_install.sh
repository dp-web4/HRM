#!/usr/bin/env bash
# router_pipeline_install.sh — Phase 0 Track 7 installer for the router shadow pipeline.
#
# What this does
#   Writes a per-machine drop-in that tells the SAGE daemon to enable the
#   router shadow hook (SAGE_ROUTER_SHADOW=1) and routes captured records to
#   SAGE_ROUTER_DATA_DIR. The installer targets the SAGE daemon's environment
#   only — it NEVER touches system-wide /etc/environment or the user profile.
#
# What this does NOT do
#   * Does not start/stop the daemon. Operators restart it themselves after
#     install so they can cross-check timing with dashboards/runbooks.
#   * Does not write any dataset rows. Capture only happens once the daemon
#     is restarted with the env loaded.
#   * Does not alter router code. Wiring for SAGE_ROUTER_DATA_DIR lives in
#     sage/core/sage_consciousness.py (Track 5 + Track 7 patch).
#
# Idempotency model
#   Configuration is persisted to a *drop-in file*, not appended to existing
#   shell RC files. Re-running the installer rewrites the drop-in from scratch.
#   The presence-or-absence of lines inside the drop-in is the only source of
#   truth. There is nothing to deduplicate because there is nothing appended.
#
# Locations by install mode
#   systemd (Linux, service exists)   → /etc/systemd/system/sage-daemon-${machine}.service.d/router-shadow.conf
#   user-systemd (Linux, no sudo)     → ~/.config/systemd/user/sage-daemon-${machine}.service.d/router-shadow.conf
#   profile drop-in (macOS / WSL)     → $SAGE_DIR/sage/gateway/router-shadow.env
#                                        (sourced by operators' launch wrapper)
#
# Usage
#   scripts/router_pipeline_install.sh                 # install for this machine
#   scripts/router_pipeline_install.sh --dry-run       # print what WOULD happen
#   scripts/router_pipeline_install.sh --uninstall     # remove the drop-in
#   scripts/router_pipeline_install.sh --data-dir PATH # override dataset root
#
# Environment overrides (read once, echoed in --dry-run output)
#   SAGE_MACHINE         explicit machine name (overrides hostname detection)
#   SAGE_ROUTER_DATA_DIR absolute dataset root (default per PRD §5:
#                        $HOME/ai-workspace/private-context/training-data/router)
#   SAGE_DIR             SAGE checkout root (default: repo inferred from script)
#
# Exit codes
#   0  success (or dry-run succeeded)
#   1  prereq failure (python/sage not importable, dataset dir not writable)
#   2  usage error
#   3  target daemon config not locatable

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Path resolution + arg parsing
# ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAGE_DIR="${SAGE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

DRY_RUN=0
UNINSTALL=0
DATA_DIR_OVERRIDE=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --uninstall) UNINSTALL=1 ;;
        --data-dir) shift; DATA_DIR_OVERRIDE="${1:-}" ;;
        --data-dir=*) DATA_DIR_OVERRIDE="${1#*=}" ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *)
            echo "[install] unknown arg: $1" >&2
            exit 2
            ;;
    esac
    shift
done

log() { echo "[router-install] $*"; }

run() {
    # Echo under dry-run, otherwise execute.
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] $*"
    else
        eval "$@"
    fi
}

# ──────────────────────────────────────────────────────────────────────
# Machine detection (matches sage/gateway/machine_config.py)
# ──────────────────────────────────────────────────────────────────────

detect_machine() {
    if [ -n "${SAGE_MACHINE:-}" ]; then
        echo "$SAGE_MACHINE" | tr '[:upper:]' '[:lower:]'
        return
    fi
    local host
    host="$(hostname 2>/dev/null | tr '[:upper:]' '[:lower:]')"
    case "$host" in
        *thor*) echo "thor" ;;
        *cbp*) echo "cbp" ;;
        *legion*) echo "legion" ;;
        *nomad*|*desktop-9e6hcao*) echo "nomad" ;;
        *mcnugget*) echo "mcnugget" ;;
        ubuntu)
            # Sprout's default hostname on Jetson Orin Nano.
            if [ -d "/home/sprout" ]; then
                echo "sprout"
            else
                echo "unknown"
            fi
            ;;
        *) echo "$host" ;;
    esac
}

MACHINE="$(detect_machine)"

# ──────────────────────────────────────────────────────────────────────
# Dataset dir + prereqs
# ──────────────────────────────────────────────────────────────────────

default_data_dir() {
    # Per PRD §5: records go to private-context/training-data/router/{machine}.
    # SAGE_ROUTER_DATA_DIR should point at the *root*; the writer builds the
    # per-machine subdir. We prefer $HOME-relative paths so systemd units can
    # resolve them under the daemon's User=.
    local candidates=(
        "${HOME}/ai-workspace/private-context/training-data/router"
        "${HOME}/ai-agents/private-context/training-data/router"
        "/mnt/c/exe/projects/ai-agents/private-context/training-data/router"
    )
    for c in "${candidates[@]}"; do
        # Use the first candidate whose grandparent exists — that tells us
        # we're on the fleet-topology that matches.
        local grandparent
        grandparent="$(dirname "$(dirname "$c")")"
        if [ -d "$grandparent" ]; then
            echo "$c"
            return
        fi
    done
    # Last resort: first candidate (operator will create via runbook).
    echo "${candidates[0]}"
}

if [ -n "$DATA_DIR_OVERRIDE" ]; then
    DATA_DIR="$DATA_DIR_OVERRIDE"
elif [ -n "${SAGE_ROUTER_DATA_DIR:-}" ]; then
    DATA_DIR="$SAGE_ROUTER_DATA_DIR"
else
    DATA_DIR="$(default_data_dir)"
fi

check_prereqs() {
    # Python 3.10+
    if ! command -v python3 >/dev/null 2>&1; then
        log "ERROR: python3 not found in PATH" >&2
        return 1
    fi
    local pyver
    pyver="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    local major minor
    major="${pyver%.*}"; minor="${pyver#*.}"
    if [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -lt 10 ]; }; then
        log "ERROR: python $pyver found; need 3.10+" >&2
        return 1
    fi

    # Importable sage package — look in SAGE_DIR first.
    if ! PYTHONPATH="$SAGE_DIR" python3 -c "import sage" >/dev/null 2>&1; then
        log "ERROR: 'sage' package not importable from SAGE_DIR=$SAGE_DIR" >&2
        return 1
    fi

    # Dataset dir: must be creatable. We don't actually create it here unless
    # installing for real; in dry-run mode we only check that the parent
    # exists + is writable so an operator can pre-flight the install.
    local parent
    parent="$(dirname "$DATA_DIR")"
    if [ -d "$parent" ]; then
        if [ ! -w "$parent" ]; then
            log "ERROR: dataset parent $parent not writable by $(whoami)" >&2
            return 1
        fi
    else
        log "WARN: dataset parent $parent does not exist; runbook step required" >&2
        # Non-fatal: operators following the runbook create it explicitly.
    fi
    return 0
}

# ──────────────────────────────────────────────────────────────────────
# Target detection: how does THIS machine start the SAGE daemon?
# ──────────────────────────────────────────────────────────────────────

SYSTEMD_SYSTEM_UNIT="sage-daemon-${MACHINE}"
SYSTEMD_SYSTEM_DROPIN_DIR="/etc/systemd/system/${SYSTEMD_SYSTEM_UNIT}.service.d"
SYSTEMD_USER_DROPIN_DIR="${HOME}/.config/systemd/user/${SYSTEMD_SYSTEM_UNIT}.service.d"
PROFILE_ENV_FILE="${SAGE_DIR}/sage/gateway/router-shadow.env"

DROPIN_FILENAME="router-shadow.conf"

detect_target() {
    # systemd unit file exists system-wide → system drop-in.
    if systemctl list-unit-files "${SYSTEMD_SYSTEM_UNIT}.service" 2>/dev/null | grep -q "^${SYSTEMD_SYSTEM_UNIT}.service"; then
        echo "systemd-system"
        return
    fi
    # systemd user unit present?
    if systemctl --user list-unit-files "${SYSTEMD_SYSTEM_UNIT}.service" 2>/dev/null | grep -q "^${SYSTEMD_SYSTEM_UNIT}.service"; then
        echo "systemd-user"
        return
    fi
    # No systemd → fallback env file sourced by ensure_daemon.sh.
    echo "profile-env"
}

TARGET="$(detect_target)"

# ──────────────────────────────────────────────────────────────────────
# Drop-in content
# ──────────────────────────────────────────────────────────────────────

systemd_dropin_body() {
    # Minimal drop-in: ONLY the router-shadow env.
    cat <<EOF
# Installed by scripts/router_pipeline_install.sh — Phase 0 Track 7.
# Remove with: scripts/router_pipeline_install.sh --uninstall
# Detected machine: ${MACHINE}
[Service]
Environment=SAGE_ROUTER_SHADOW=1
Environment=SAGE_ROUTER_DATA_DIR=${DATA_DIR}
EOF
}

profile_env_body() {
    cat <<EOF
# Installed by scripts/router_pipeline_install.sh — Phase 0 Track 7.
# Sourced by sage/scripts/ensure_daemon.sh and raising wrappers.
# Remove with: scripts/router_pipeline_install.sh --uninstall
export SAGE_ROUTER_SHADOW=1
export SAGE_ROUTER_DATA_DIR=${DATA_DIR}
EOF
}

# ──────────────────────────────────────────────────────────────────────
# Install / uninstall drivers
# ──────────────────────────────────────────────────────────────────────

install_systemd_system() {
    log "target: systemd system unit (${SYSTEMD_SYSTEM_UNIT}.service)"
    run "sudo mkdir -p \"$SYSTEMD_SYSTEM_DROPIN_DIR\""
    local body
    body="$(systemd_dropin_body)"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] would write ${SYSTEMD_SYSTEM_DROPIN_DIR}/${DROPIN_FILENAME}:"
        echo "$body" | sed 's/^/[dry-run]   /'
    else
        echo "$body" | sudo tee "${SYSTEMD_SYSTEM_DROPIN_DIR}/${DROPIN_FILENAME}" >/dev/null
        sudo systemctl daemon-reload
        log "installed. Restart with: sudo systemctl restart ${SYSTEMD_SYSTEM_UNIT}"
    fi
}

install_systemd_user() {
    log "target: systemd user unit (${SYSTEMD_SYSTEM_UNIT}.service, user scope)"
    run "mkdir -p \"$SYSTEMD_USER_DROPIN_DIR\""
    local body
    body="$(systemd_dropin_body)"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] would write ${SYSTEMD_USER_DROPIN_DIR}/${DROPIN_FILENAME}:"
        echo "$body" | sed 's/^/[dry-run]   /'
    else
        echo "$body" > "${SYSTEMD_USER_DROPIN_DIR}/${DROPIN_FILENAME}"
        systemctl --user daemon-reload
        log "installed. Restart with: systemctl --user restart ${SYSTEMD_SYSTEM_UNIT}"
    fi
}

install_profile_env() {
    log "target: profile env file (${PROFILE_ENV_FILE})"
    run "mkdir -p \"$(dirname "$PROFILE_ENV_FILE")\""
    local body
    body="$(profile_env_body)"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] would write ${PROFILE_ENV_FILE}:"
        echo "$body" | sed 's/^/[dry-run]   /'
    else
        echo "$body" > "$PROFILE_ENV_FILE"
        chmod 0644 "$PROFILE_ENV_FILE"
        log "installed. Daemon launchers should source this file before exec."
        log "  e.g.  set -a; . ${PROFILE_ENV_FILE}; set +a; python3 -m sage.gateway"
    fi
}

uninstall_systemd_system() {
    local path="${SYSTEMD_SYSTEM_DROPIN_DIR}/${DROPIN_FILENAME}"
    if [ -f "$path" ]; then
        run "sudo rm -f \"$path\""
        run "sudo rmdir --ignore-fail-on-non-empty \"$SYSTEMD_SYSTEM_DROPIN_DIR\" 2>/dev/null || true"
        run "sudo systemctl daemon-reload"
        log "removed ${path}"
    else
        log "nothing to remove at ${path}"
    fi
}

uninstall_systemd_user() {
    local path="${SYSTEMD_USER_DROPIN_DIR}/${DROPIN_FILENAME}"
    if [ -f "$path" ]; then
        run "rm -f \"$path\""
        run "rmdir --ignore-fail-on-non-empty \"$SYSTEMD_USER_DROPIN_DIR\" 2>/dev/null || true"
        run "systemctl --user daemon-reload"
        log "removed ${path}"
    else
        log "nothing to remove at ${path}"
    fi
}

uninstall_profile_env() {
    if [ -f "$PROFILE_ENV_FILE" ]; then
        run "rm -f \"$PROFILE_ENV_FILE\""
        log "removed ${PROFILE_ENV_FILE}"
    else
        log "nothing to remove at ${PROFILE_ENV_FILE}"
    fi
}

# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

log "SAGE_DIR=${SAGE_DIR}"
log "machine=${MACHINE}"
log "dataset dir=${DATA_DIR}"
log "target=${TARGET}"
log "mode=$([ "$UNINSTALL" -eq 1 ] && echo uninstall || echo install)$([ "$DRY_RUN" -eq 1 ] && echo " (dry-run)" || echo "")"

if [ "$UNINSTALL" -eq 1 ]; then
    case "$TARGET" in
        systemd-system) uninstall_systemd_system ;;
        systemd-user)   uninstall_systemd_user ;;
        profile-env)    uninstall_profile_env ;;
        *) log "ERROR: unknown target $TARGET" >&2; exit 3 ;;
    esac
    exit 0
fi

# Install path — run prereqs first.
if ! check_prereqs; then
    exit 1
fi

case "$TARGET" in
    systemd-system) install_systemd_system ;;
    systemd-user)   install_systemd_user ;;
    profile-env)    install_profile_env ;;
    *) log "ERROR: unknown target $TARGET" >&2; exit 3 ;;
esac

log "done. Verify after daemon restart with: scripts/router_pipeline_verify.sh"
