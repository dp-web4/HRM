#!/bin/bash
# Fleet gameplay capture — runs gameplay_capture on a machine's assigned games.
#
# Each machine runs a subset of the 25 ARC-AGI-3 games, producing
# source=gameplay router-shadow records that Phase 1 training consumes.
#
# Usage:
#   ./scripts/fleet_gameplay_capture.sh [MACHINE]
#
# MACHINE is auto-detected from $SAGE_MACHINE or hostname if not passed.
# Set SAGE_ROUTER_DATA_DIR to override the output partition location.
#
# Exit codes:
#   0 — all assigned games captured successfully
#   1 — MACHINE unknown / not in assignment table
#   2 — at least one game failed; others succeeded (partial)
#   3 — all games failed

set -u   # catch unset vars; don't -e (we handle per-game failures)

SAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SAGE_DIR"

MACHINE="${1:-${SAGE_MACHINE:-$(hostname | tr '[:upper:]' '[:lower:]')}}"
MACHINE="${MACHINE,,}"   # lowercase

# Normalize common hostnames to fleet machine slugs
case "$MACHINE" in
  *sprout*|*orin*) MACHINE="sprout" ;;
  *thor*)          MACHINE="thor" ;;
  *legion*)        MACHINE="legion" ;;
  *nomad*)         MACHINE="nomad" ;;
  *mcnugget*|*mac*) MACHINE="mcnugget" ;;
  *cbp*|*ubuntu*)  MACHINE="cbp" ;;
esac

# ── Per-machine game assignments ────────────────────────────────────────
# Allocation principle: each game captured once across the fleet. Lighter
# games to lighter machines. Coverage across all 25 games + 1 overlap on
# CBP (origin canary) so we have at least one cross-machine comparison.
case "$MACHINE" in
  sprout)
    # Edge device — smaller traces, faster playback
    GAMES=(cd82 ft09 lp85)
    ;;
  nomad)
    # 4B laptop
    GAMES=(sb26 sc25 tr87 cn04)
    ;;
  mcnugget)
    # 12B Mac — cerebellum owner
    GAMES=(r11l g50t vc33 wa30)
    ;;
  legion)
    # 14B Linux — RPE + training infra
    GAMES=(ar25 su15 sp80 ls20 m0r0)
    ;;
  thor)
    # 14B high-RAM — episodic owner; gets the hardest games
    GAMES=(ka59 sk48 tu93 s5i5 tn36)
    ;;
  cbp)
    # Coordinator; re-runs cd82 + ft09 as proof-of-concept + adds our
    # self-produced winning traces (bp35, lf52, dc22, re86)
    GAMES=(cd82 ft09 bp35 lf52 dc22 re86)
    ;;
  *)
    echo "[fleet-gameplay] ERROR: unknown machine '$MACHINE'"
    echo "  Valid: sprout, nomad, mcnugget, legion, thor, cbp"
    echo "  Pass explicitly: $0 <machine>"
    exit 1
    ;;
esac

echo "[fleet-gameplay] machine=$MACHINE games=${GAMES[*]}"
echo "[fleet-gameplay] SAGE_DIR=$SAGE_DIR"
echo "[fleet-gameplay] data_dir=${SAGE_ROUTER_DATA_DIR:-/mnt/c/exe/projects/ai-agents/private-context/training-data/router}"
echo

success_count=0
failure_count=0
failed_games=()

for game in "${GAMES[@]}"; do
  echo "═══════════════════════════════════════════════════════════"
  echo "[fleet-gameplay] Capturing $game..."
  echo "═══════════════════════════════════════════════════════════"
  if PYTHONPATH="$SAGE_DIR" python3 -m sage.cognition.thalamic_router.gameplay_capture \
       --game "$game" --machine "$MACHINE"; then
    ((success_count++))
  else
    ((failure_count++))
    failed_games+=("$game")
  fi
  echo
done

echo "═══════════════════════════════════════════════════════════"
echo "[fleet-gameplay] DONE: $success_count succeeded, $failure_count failed"
if [ $failure_count -gt 0 ]; then
  echo "  Failed: ${failed_games[*]}"
fi
echo "═══════════════════════════════════════════════════════════"

# Exit code
if [ $success_count -eq 0 ]; then
  exit 3
elif [ $failure_count -gt 0 ]; then
  exit 2
else
  exit 0
fi
