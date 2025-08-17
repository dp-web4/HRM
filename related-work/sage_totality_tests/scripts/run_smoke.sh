
#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-http://localhost:8080}

echo "== Health =="
curl -s ${BASE}/health | python -m json.tool

echo "== Read (activation_min=0.0) =="
curl -s "${BASE}/totality/read?activation_min=0.0" | python -m json.tool | head -n 50
