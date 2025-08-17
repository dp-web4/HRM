
#!/usr/bin/env bash
set -euo pipefail
BASE=${BASE:-http://localhost:8080}

echo "== Read to get a scheme id =="
SCHEME_ID=$(curl -s "${BASE}/totality/read?activation_min=0.0" | python - <<'PY'
import sys, json
d=json.load(sys.stdin)
schemes=d.get("schemes",[])
print(schemes[0]["id"] if schemes else "")
PY
)
echo "Using scheme: $SCHEME_ID"

echo "== Activate +0.2 =="
curl -s -X POST "${BASE}/totality/activate"   -H "Content-Type: application/json"   -d "{"targets": [{"id": "${SCHEME_ID}", "delta": 0.2}]}" | python -m json.tool
