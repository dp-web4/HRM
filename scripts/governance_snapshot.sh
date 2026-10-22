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
#   defaults: <workspace>/hestia (or <workspace>/HESTIA)  <workspace>/governance/hestia
# Idempotent: rebuilds into a temp dir, swaps it in, records the commit in dest/SNAPSHOT.
set -eu
# THE WORKSPACE IS WHERE THIS SCRIPT LIVES, never a literal path (fleet directive: no hardcoded paths).
# scripts/ sits in the SAGE checkout, which sits in the workspace: ai-workspace on most boxes, ai-agents on
# nomad. HESTIA_WORKSPACE overrides, as it does for the gate.
WS="${HESTIA_WORKSPACE:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}"
HESTIA_SRC="$WS/hestia"; [ -e "$HESTIA_SRC/.git" ] || [ ! -e "$WS/HESTIA/.git" ] || HESTIA_SRC="$WS/HESTIA"
SRC="${1:-$HESTIA_SRC}"
DEST="${2:-$WS/governance/hestia}"
git -C "$SRC" fetch -q origin main
MAIN="$(git -C "$SRC" rev-parse origin/main)"
# THE DEPLOYED BUILD, WHEN KNOWN (cbp-claude, 2026-10-19). Log lines a member sees name file:line in the DEPLOYED
# tree; a snapshot of main shows unrelated code at that line (CBP: the log's http.rs:1436 was http.rs:1819 on
# main), and that becomes a new mystery. hestia records the deployed commit in current-build.json (head_sha).
#
# ASKED OF THE DAEMON, NOT READ FROM ~/.hestia (cbp-claude, 2026-10-20). Any member with the workspace may run
# this, and ~/.hestia is outside every member's grant: a script that opened current-build.json would be reaching
# it by a route the gate never sees. The running daemon publishes the same commit on /api/version (build_id
# "...-g<sha>"), at the endpoint the gate already uses (HESTIA_ENDPOINT). An operator may still name a build
# file explicitly with HESTIA_CURRENT_BUILD_FILE. Neither answering falls back to main, and SNAPSHOT says which.
API="${HESTIA_ENDPOINT:-http://127.0.0.1:7711/mcp}"; API="${API%/mcp}/api/version"
DEPLOYED="$(curl -fsS --max-time 5 "$API" 2>/dev/null | python3 -c "import json,sys; b=json.load(sys.stdin).get('build_id') or ''; print(b.rsplit('-g',1)[1] if '-g' in b else '')" 2>/dev/null || true)"
HOW="the running daemon's /api/version"
if [ -z "$DEPLOYED" ] && [ -n "${HESTIA_CURRENT_BUILD_FILE:-}" ]; then
  DEPLOYED="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('head_sha') or '')" "$HESTIA_CURRENT_BUILD_FILE" 2>/dev/null || true)"
  HOW="current-build.json head_sha"
fi
if [ -n "$DEPLOYED" ]; then DEPLOYED="$(git -C "$SRC" rev-parse --verify -q "$DEPLOYED^{commit}" 2>/dev/null || echo "$DEPLOYED")"; fi
if [ -n "$DEPLOYED" ] && git -C "$SRC" cat-file -e "$DEPLOYED^{commit}" 2>/dev/null; then
  REV="$DEPLOYED"; BASIS="the deployed build ($HOW)"
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
