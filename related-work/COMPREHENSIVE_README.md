
# SAGE ↔ Totality Integrated Service + Test Pack

This bundle combines:

- **`sage_totality_service/`** — a FastAPI microservice wrapping the **minimal Totality/SubThought testbed** (`totality_min`).
- **`sage_totality_tests/`** — scripts and Python tests to validate behavior.
- **`totality_min/`** — a tiny, dependency-free prototype of Totality/SubThought (schemes, canvases, activations, imagination).

The goal: give SAGE a **modular cognitive sensor** that can read, imagine, activate, write, and snapshot knowledge structures — and verify the loop end-to-end.

---

## File Layout

```
sage_totality_service/
  app/main.py          # FastAPI service exposing Totality ops
  requirements.txt     # FastAPI + Pydantic runtime deps
  README.md            # Service-specific instructions
  totality_min/        # Minimal Totality prototype
    totality_core.py   # In-memory store (Scheme, Canvas, ops)
    transforms.py      # Naive context/semantic transforms
    demo.py            # Run small standalone scenario
    tests/test_totality.py
    README.md

sage_totality_tests/
  README_TESTS.md      # Checklist & instructions (copied below)
  requirements_dev.txt # pytest + requests for API tests
  scripts/             # Bash curl-based tests
    run_smoke.sh
    run_imagine.sh
    run_activate.sh
    run_write.sh
    run_sleep_cycle.sh
  tests/
    test_api.py        # pytest version of API tests
    run_tests.py       # lightweight runner with requests
```

---

## Service Setup

```bash
unzip sage_totality_service.zip
cd sage_totality_service

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8080
```

Now the API is live at `http://localhost:8080`.

---

## API Endpoints

- `GET /health` → liveness check
- `GET /totality/read?activation_min=0.3` → get schemes/canvases (filter by activation)
- `POST /totality/imagine` → dream new canvases from seeds
- `POST /totality/activate` → adjust salience/attention
- `POST /totality/write` → commit/merge schemes
- `GET /totality/snapshot` → dump full in-memory store

Seeding: the service auto-seeds two schemes (`grasp(object)`, `place(object,location)`) + one canvas.

---

## Test Pack

### Option A — Bash Scripts (curl-based)

```bash
unzip sage_totality_tests.zip
cd sage_totality_tests

bash scripts/run_smoke.sh          # health + seeded data
bash scripts/run_imagine.sh        # dream 2 new canvases
bash scripts/run_activate.sh       # upweight scheme activation
bash scripts/run_write.sh          # commit distilled abstraction
bash scripts/run_sleep_cycle.sh    # simulate a sleep cycle
```

**Pass/Fail Criteria:**
- **Smoke:** returns at least 1 scheme + 1 canvas.  
- **Imagine:** returns new canvas IDs; total canvases increases.  
- **Activate:** target scheme activation increases.  
- **Write:** `"status": "ok"` and scheme committed.  
- **Sleep Cycle:** after imagine+write, canvases > before; snapshot shows abstraction.

### Option B — Python Runner

```bash
cd sage_totality_tests
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_dev.txt

pytest -q                 # run tests/test_api.py
python tests/run_tests.py # or lightweight runner without pytest
```

---

## Quick Demo (standalone)

Run the mini Totality demo without FastAPI:

```bash
cd sage_totality_service/totality_min
python demo.py
```

Expected: seeds schemes, creates base canvas, imagines variants, adjusts activation, commits abstraction, and prints final store.

---

## Notes

- **Stateless**: in-memory store only. Restarting clears schemes/canvases. Extend with file-backed persistence if needed.
- **Imagination Ops**: current “dreaming” is naive (slot value permutation). Plug in richer augmentation later.
- **Lightweight**: designed for Jetson/Nano deployment. No DB, no heavy deps.

---

👉 Recommended first test:  
1. Start the service.  
2. Run `bash scripts/run_sleep_cycle.sh`.  
3. Check output: canvases before vs. after, plus new abstraction.  

That confirms **read → imagine → activate → write → snapshot** all work.
