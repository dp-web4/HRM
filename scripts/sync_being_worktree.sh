#!/usr/bin/env bash
# Sync a being's worktree forward WITHOUT destroying its work.
#
# Until M1 the seat synced with `git reset --hard`, which was fine while the being could only
# read its tree. It can write it now. A hard reset would silently delete a test it has been
# authoring across beats — the exact kind of loss that would look, from its side, like the
# world eating its work. So: fast-forward only, and only when clean; otherwise say why.
set -euo pipefail
WT="${1:-/home/dp/ai-workspace/being-worktrees/legion-being}"
REF="${2:-legion/mission-artifact}"
cd "$WT"
if [ -n "$(git status --porcelain)" ]; then
  echo "NOT SYNCED: $WT has uncommitted changes (the being's work). Leaving it alone." >&2
  git status --short >&2
  exit 3
fi
BR="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BR" != "legion-being/work" ]; then
  echo "NOT SYNCED: worktree is on '$BR' (a PR branch, presumably). Leaving it alone." >&2
  exit 4
fi
git fetch -q origin "$REF"
if git merge --ff-only -q FETCH_HEAD 2>/dev/null; then
  echo "synced to $(git log --oneline -1)"
else
  echo "NOT SYNCED: legion-being/work has diverged from $REF; needs a human." >&2
  exit 5
fi
