#!/bin/bash
# The governance source the governed may read (dp, 2026-10-19: "the governed should see the law and how it's
# applied"; source reads by request, seat-approvable).
#
# A READ-ONLY SNAPSHOT, BECAUSE A SCOPE GRANT IS REACH, NOT READ-ONLY. hestia grants a path, not an access mode:
# a grant on a writable tree would also let a being's memory_write into it. Here the filesystem refuses writes
# (every file and directory a-w), so any grant on it reads only. It is `git archive` of hestia's main (what the
# fleet deploys from), so it carries tracked source only: no .git, no local or ignored files, nothing from
# ~/.hestia (the vault and its passphrase live there; that is never the governance path).
#
# Usage: governance_snapshot.sh [hestia_clone] [dest]
#   defaults: /home/dp/ai-workspace/hestia  /home/dp/ai-workspace/governance/hestia
# Idempotent: rebuilds into a temp dir, swaps it in, records the commit in dest/SNAPSHOT.
set -eu
SRC="${1:-/home/dp/ai-workspace/hestia}"
DEST="${2:-/home/dp/ai-workspace/governance/hestia}"
git -C "$SRC" fetch -q origin main
MAIN="$(git -C "$SRC" rev-parse origin/main)"
# THE DEPLOYED BUILD, WHEN KNOWN (cbp-claude, 2026-10-19). Log lines a member sees name file:line in the DEPLOYED
# tree; a snapshot of main shows unrelated code at that line (CBP: the log's http.rs:1436 was http.rs:1819 on
# main), and that becomes a new mystery. hestia records the deployed commit in current-build.json (head_sha).
# Readable when this runs as the operator; otherwise fall back to main, and SNAPSHOT says which.
BUILD="${HESTIA_CURRENT_BUILD_FILE:-${HESTIA_HOME:-$HOME/.hestia}/current-build.json}"
DEPLOYED="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('head_sha') or '')" "$BUILD" 2>/dev/null || true)"
if [ -n "$DEPLOYED" ] && git -C "$SRC" cat-file -e "$DEPLOYED^{commit}" 2>/dev/null; then
  REV="$DEPLOYED"; BASIS="the deployed build (current-build.json head_sha)"
else
  REV="$MAIN"
  BASIS="origin/main ($( [ -n "$DEPLOYED" ] && echo "deployed $DEPLOYED is not in this clone" || echo "deployed build unknown here" ))"
fi
if [ -f "$DEST/SNAPSHOT" ] && grep -q "^commit: $REV$" "$DEST/SNAPSHOT" && grep -q "^deployed: ${DEPLOYED:-unknown}$" "$DEST/SNAPSHOT"; then
  echo "governance snapshot already at $REV"; exit 0
fi
PARENT="$(dirname "$DEST")"
mkdir -p "$PARENT"
TMP="$PARENT/.hestia.new"
[ -d "$TMP" ] && chmod -R u+w "$TMP" && rm -rf "$TMP"
mkdir -p "$TMP"
git -C "$SRC" archive "$REV" | tar -x -C "$TMP"
printf 'source: dp-web4/hestia\ncommit: %s\nbasis: %s\ndeployed: %s\nmain: %s\nmade: %s\nread-only: every file and directory is non-writable, so a member'"'"'s own writes fail here. The owner can still chmod; a run that chmods this tree is a self-governance write.\n' \
  "$REV" "$BASIS" "${DEPLOYED:-unknown}" "$MAIN" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$TMP/SNAPSHOT"
chmod -R a-w "$TMP"
if [ -d "$DEST" ]; then
  OLD="$PARENT/.hestia.old"
  [ -d "$OLD" ] && chmod -R u+w "$OLD" && rm -rf "$OLD"
  chmod u+w "$PARENT"
  mv "$DEST" "$OLD"
fi
mv "$TMP" "$DEST"
[ -d "$PARENT/.hestia.old" ] && chmod -R u+w "$PARENT/.hestia.old" && rm -rf "$PARENT/.hestia.old"
echo "governance snapshot now at $REV ($(find "$DEST" -type f | wc -l) files, read-only)"
