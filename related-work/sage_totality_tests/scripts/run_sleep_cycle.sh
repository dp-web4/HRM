
#!/usr/bin/env bash
set -euo pipefail
BASE=${BASE:-http://localhost:8080}

echo "== Before: Read =="
BEFORE=$(curl -s "${BASE}/totality/read?activation_min=0.0")
BEFORE_CANV=$(echo "$BEFORE" | python - <<'PY'
import sys, json
d=json.load(sys.stdin)
print(len(d.get("canvases",[])))
PY
)
echo "Canvas count (before): $BEFORE_CANV"

SCHEME_ID=$(echo "$BEFORE" | python - <<'PY'
import sys, json
d=json.load(sys.stdin)
schemes=d.get("schemes",[])
print(schemes[0]["id"] if schemes else "")
PY
)

echo "== Dream/Imagine 2 =="
curl -s -X POST "${BASE}/totality/imagine"   -H "Content-Type: application/json"   -d "{"seed_scheme_ids": ["${SCHEME_ID}"], "ops": [{"op": "value_perm"}], "count": 2}" | python -m json.tool

echo "== Write distilled abstraction =="
curl -s -X POST "${BASE}/totality/write"   -H "Content-Type: application/json"   -d '{"schemes": [{"label": "manipulate(item)","slots": [{"name":"item","value":"cube"}],"activation": 0.6}],"mode": "merge"}' | python -m json.tool

echo "== After: Snapshot =="
AFTER=$(curl -s "${BASE}/totality/snapshot")
echo "$AFTER" | python -m json.tool | head -n 50
AFTER_CANV=$(echo "$AFTER" | python - <<'PY'
import sys, json
d=json.load(sys.stdin)["snapshot"]
print(len(d.get("canvases",[])))
PY
)
echo "Canvas count (after): $AFTER_CANV"
if [ "$AFTER_CANV" -gt "$BEFORE_CANV" ]; then
  echo "PASS: dreams increased canvases."
else
  echo "FAIL: no new canvases."
  exit 1
fi
