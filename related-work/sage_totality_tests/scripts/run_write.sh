
#!/usr/bin/env bash
set -euo pipefail
BASE=${BASE:-http://localhost:8080}

echo "== Write a distilled abstraction =="
curl -s -X POST "${BASE}/totality/write"   -H "Content-Type: application/json"   -d '{"schemes": [{"label": "manipulate(item)","slots": [{"name":"item","value":"cube"}],"activation": 0.5}],"mode": "merge"}' | python -m json.tool
