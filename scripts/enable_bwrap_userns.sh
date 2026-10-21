#!/usr/bin/env bash
# Let bubblewrap (and only bubblewrap) create user namespaces under Ubuntu 24.04's
# kernel.apparmor_restrict_unprivileged_userns=1, so the being's `run` sandbox works.
# See sage/gateway/being_gate_client.py ENABLEMENT.
#
# OPERATOR ONLY, BY HAND. It needs root, and nothing in SAGE runs it: no gateway start-up, no
# beat, no verb (test_no_sage_code_runs_the_userns_setup). A being's `run` is in a sandbox with no
# root and no /etc, and its request_run goes to a seat, which declines sudo.
#
# WHAT IT CHANGES ON THE HOST. One AppArmor profile, /etc/apparmor.d/bwrap, which lets any process
# that executes /usr/bin/bwrap, for any user, create an unprivileged user namespace. The kernel
# setting stays 1 for every other binary; the alternative, setting it to 0, would open user
# namespaces to every program on the machine. What remains: any local user can now use bwrap's
# namespaces, as on distributions without the restriction (Ubuntu ships the same pattern for
# flatpak, crun and ch-run). The profile is unconfined apart from that: it adds no other permission.
#
# REVERSAL: sudo bash enable_bwrap_userns.sh --remove
# It unloads and deletes the profile, only if this script wrote it. After that, `run` on this box
# refuses again ("run needs its sandbox and cannot get one"); it never falls back to running
# unsandboxed (test_run_fails_closed_when_no_sandbox_is_usable).
#
# Usage: sudo bash enable_bwrap_userns.sh [--remove]
set -euo pipefail

PROFILE=/etc/apparmor.d/bwrap
BWRAP=/usr/bin/bwrap
MARK="# written by SAGE scripts/enable_bwrap_userns.sh"

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

if [ "${1:-}" = "--remove" ]; then
  [ -e "$PROFILE" ] || { echo "$PROFILE is not there; nothing to remove"; exit 0; }
  grep -qxF "$MARK" "$PROFILE" || { echo "$PROFILE was not written by this script; leaving it"; exit 1; }
  apparmor_parser -R "$PROFILE" || true
  rm -f "$PROFILE"
  echo "removed $PROFILE; bwrap user namespaces are restricted again"
  exit 0
fi

[ -x "$BWRAP" ] || { echo "bubblewrap not installed: apt install bubblewrap"; exit 1; }
[ -n "${SUDO_USER:-}" ] || { echo "run through sudo from the account that runs the beings"; exit 1; }

echo "restrict_unprivileged_userns = $(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo n/a)"

if [ -e "$PROFILE" ]; then
  echo "$PROFILE already exists; leaving it as is:"
  cat "$PROFILE"
else
  cat > "$PROFILE" <<EOF
$MARK
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,

  include if exists <local/bwrap>
}
EOF
  echo "wrote $PROFILE"
fi

apparmor_parser -r "$PROFILE"
echo "profile loaded"

# The same probe SAGE uses (sandbox_available), run as the user who invoked sudo.
if sudo -u "$SUDO_USER" "$BWRAP" --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind-try /lib64 /lib64 \
     --unshare-pid /usr/bin/true; then
  echo "sandbox ok for $SUDO_USER"
else
  echo "sandbox STILL FAILS for $SUDO_USER"; exit 2
fi
