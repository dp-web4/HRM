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
REV="$(git -C "$SRC" rev-parse origin/main)"
if [ -f "$DEST/SNAPSHOT" ] && grep -q "^commit: $REV$" "$DEST/SNAPSHOT"; then
  echo "governance snapshot already at $REV"; exit 0
fi
PARENT="$(dirname "$DEST")"
mkdir -p "$PARENT"
TMP="$PARENT/.hestia.new"
[ -d "$TMP" ] && chmod -R u+w "$TMP" && rm -rf "$TMP"
mkdir -p "$TMP"
git -C "$SRC" archive "$REV" | tar -x -C "$TMP"
printf 'source: dp-web4/hestia\ncommit: %s\nmade: %s\nread-only: every file and directory is non-writable; a grant here reads only\n' \
  "$REV" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$TMP/SNAPSHOT"
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
