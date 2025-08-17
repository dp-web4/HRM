
# SAGE ↔ Totality Service (Integrated)

A FastAPI service that **uses the minimal Totality/SubThought testbed** under the hood. This lets you test SAGE's cognitive-sensor integration without any external dependencies.

## Endpoints
- `GET /health` — liveness check
- `GET /totality/read?activation_min=0.3` — read schemes/canvases (filters by activation)
- `POST /totality/imagine` — create new canvases from seed schemes + ops
- `POST /totality/activate` — adjust activation values
- `POST /totality/write` — commit/merge schemes
- `GET /totality/snapshot` — dump the in-memory store

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

### Try it

```bash
curl http://localhost:8080/health

curl "http://localhost:8080/totality/read?activation_min=0.0"

curl -X POST http://localhost:8080/totality/imagine \
  -H "Content-Type: application/json" \
  -d '{"seed_scheme_ids": [], "ops": [{"op": "value_perm"}], "count": 1}'

curl -X POST http://localhost:8080/totality/activate \
  -H "Content-Type: application/json" \
  -d '{"targets": [{"id": "REPLACE_WITH_SCHEME_ID", "delta": 0.2}]}'

curl -X POST http://localhost:8080/totality/write \
  -H "Content-Type: application/json" \
  -d '{"schemes": [{"label": "manipulate(item)", "slots": [{"name":"item","value":"cube"}], "activation": 0.5}], "mode": "merge"}'
```

## Notes
- Backed by `totality_min` in-memory store (included here).
- Replace the naive imagination op with your augmentation engine later.
- This is intentionally small so it runs on Jetson. Persist to disk if needed.
