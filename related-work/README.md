
# SAGE ↔ Totality Adapter (FastAPI Stub)

A minimal stub service that implements the adapter endpoints so you can treat a Totality-style world model as a **Cognitive Sensor** inside SAGE.

## Endpoints
- `GET /totality/read` — return CognitiveReading from in-memory store
- `POST /totality/imagine` — create imagined canvases
- `POST /totality/activate` — up/down-weight activations
- `POST /totality/write` — commit distilled schemes
- `GET /totality/snapshot` — capture a snapshot of the store

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

Then hit:
```bash
curl http://localhost:8080/totality/read
```

## Notes
- This is a **stub**: persistence, auth, and a real graph/ontology backend are intentionally omitted.
- Map **Plutchik → SNARC** and **Trust → Strategy** in your SAGE logic around this service.
