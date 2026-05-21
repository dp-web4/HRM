#!/bin/bash
# 25-game sweep with v8 substrate adapter + granite3.2-vision
# McNugget, 2026-04-19

set -u
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ARC_SAGE_DIR=/Users/dennispalatov/repos/ARC-SAGE
export PYTHONPATH=/Users/dennispalatov/repos/SAGE

ADAPTER=/Users/dennispalatov/repos/shared-context/arc-agi-3/phase2/adapters/thor-framerouter-v2invoke-2026-04-20
DATA_DIR=/Users/dennispalatov/repos/private-context/training-data/router
MACHINE=mcnugget
MODEL=qwen3.5:9b
OUT_DIR=/tmp/sweep-v8-mcnugget
MAX_STEPS=500  # real budget — game-overs will terminate early

mkdir -p "$OUT_DIR"

GAMES="cd82 cd82-fb555c5d
sb26 sb26-7fbdac44
ft09 ft09-0d8bbf25
r11l r11l-aa269680
sc25 sc25-635fd71a
tn36 tn36-ab4f63cc
vc33 vc33-9851e02b
tr87 tr87-cd924810
tu93 tu93-2b534c15
lp85 lp85-305b61c3
sp80 sp80-0ee2d095
ls20 ls20-9607627b
su15 su15-4c352900
g50t g50t-5849a774
ar25 ar25-0c556536
s5i5 s5i5-18d95033
bp35 bp35-0a0ad940
sk48 sk48-41055498
cn04 cn04-2fe56bfb
ka59 ka59-38d34dbb
m0r0 m0r0-492f87ba
re86 re86-8af5384d
dc22 dc22-fdcac232
lf52 lf52-271a04aa
wa30 wa30-ee6fef47"

WON=0
PLAYED=0
FAILED=0

echo "=========================================="
echo "25-game sweep — v8 + granite3.2-vision"
echo "=========================================="

while IFS=' ' read -r game gid; do
  PLAYED=$((PLAYED + 1))
  echo ""
  echo "--- [$PLAYED/25] $game ---"

  python3 -m sage.cognition.thalamic_router.llm_dispatch \
    --adapter "$ADAPTER" \
    --game "$game" --game-id "$gid" \
    --machine "$MACHINE" \
    --data-dir "$DATA_DIR" \
    --llm-model "$MODEL" \
    --max-steps "$MAX_STEPS" \
    --json-out "$OUT_DIR/$game.json" \
    2>&1 | grep -E "SAGE took|Final|Outcome|Actions|Invokes|LLM calls|First LLM|rationale|action:|latency" || {
      echo "  FAILED"
      FAILED=$((FAILED + 1))
      continue
    }

  # Check if any level cleared
  levels=$(python3 -c "import json; d=json.load(open('$OUT_DIR/$game.json')); print(d.get('final_levels',0))" 2>/dev/null)
  if [ "$levels" != "0" ] && [ -n "$levels" ]; then
    echo "  *** WON $levels LEVEL(S)! ***"
    WON=$((WON + 1))
  fi
done <<< "$GAMES"

echo ""
echo "=========================================="
echo "SWEEP COMPLETE"
echo "  Played: $PLAYED"
echo "  Won ≥1 level: $WON"
echo "  Failed: $FAILED"
echo "=========================================="

# Summary JSON
python3 -c "
import json, pathlib
results = {}
for f in sorted(pathlib.Path('$OUT_DIR').glob('*.json')):
    d = json.load(open(f))
    results[f.stem] = {
        'steps': d.get('n_steps',0),
        'levels': d.get('final_levels',0),
        'outcome': d.get('outcome','?'),
        'invokes': d.get('invoke_count',0),
        'llm_calls': d.get('llm_calls',0),
        'actions': d.get('action_counts',{}),
    }
json.dump(results, open('$OUT_DIR/summary.json','w'), indent=2)
print(json.dumps(results, indent=2))
"
