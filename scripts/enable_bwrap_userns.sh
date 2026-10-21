#!/usr/bin/env bash
# Let bubblewrap (and only bubblewrap) create user namespaces under Ubuntu 24.04's
# kernel.apparmor_restrict_unprivileged_userns=1, so the being's `run` sandbox works.
# See sage/gateway/being_gate_client.py ENABLEMENT. Run as the operator: sudo bash this-script
set -euo pipefail

PROFILE=/etc/apparmor.d/bwrap
BWRAP=/usr/bin/bwrap

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
[ -x "$BWRAP" ] || { echo "bubblewrap not installed: apt install bubblewrap"; exit 1; }

echo "restrict_unprivileged_userns = $(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo n/a)"

if [ -e "$PROFILE" ]; then
  echo "$PROFILE already exists; leaving it as is:"
  cat "$PROFILE"
else
  cat > "$PROFILE" <<'EOF'
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
RUN_AS="${SUDO_USER:-dp}"
if sudo -u "$RUN_AS" "$BWRAP" --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind-try /lib64 /lib64 \
     --unshare-pid /usr/bin/true; then
  echo "sandbox ok for $RUN_AS"
else
  echo "sandbox STILL FAILS for $RUN_AS"; exit 2
fi
