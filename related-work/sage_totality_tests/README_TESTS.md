
# SAGE ↔ Totality — Test Pack

This pack contains **what to test** and small scripts/tests to run against the integrated FastAPI service.

Service repo: `sage_totality_service` (from previous download). Start it first:

```bash
cd sage_totality_service
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

Then, in another terminal, run tests from this pack.

---

## What to Test (Checklist)

1) **Health & Seeding**
- `GET /health` returns `status=ok`.
- `GET /totality/read?activation_min=0.0` returns at least **one scheme** and **one canvas** (pre-seeded).

2) **Imagine (Dream Augmentation)**
- `POST /totality/imagine` with a seeded scheme id creates **N new canvases**.
- Re-run read; verify **canvases count increased**.

3) **Activate (Attention/Salience)**
- `POST /totality/activate` with a scheme id and delta `+0.2` raises its **activation** (bounded [0,1]).

4) **Write (Distilled Abstractions)**
- `POST /totality/write` commits a new or merged **scheme**.
- Re-run read; verify scheme exists with **expected label and slots**.

5) **Snapshot**
- `GET /totality/snapshot` returns `"snapshot"` with current store contents.

6) **Mini Sleep Cycle (End-to-End)**
- Read → pick scheme ids → Imagine (count=2) → Write new abstraction → Snapshot.
- Expectation: post-sleep, store has **more canvases**, and a **new or merged scheme** representing the distilled pattern.

---

## How to Run

### Option A — Bash scripts (curl-based)
```bash
bash scripts/run_smoke.sh
bash scripts/run_imagine.sh
bash scripts/run_activate.sh
bash scripts/run_write.sh
bash scripts/run_sleep_cycle.sh
```

### Option B — Python test (requests + pytest)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_dev.txt
pytest -q
```

> If you don't want dev deps, you can run `python tests/run_tests.py` (no pytest needed).

---

## Pass/Fail Criteria

- **Smoke:** `run_smoke.sh` prints health OK and shows at least one scheme & canvas.  
- **Imagine:** `run_imagine.sh` shows >0 new canvas IDs.  
- **Activate:** `run_activate.sh` shows the chosen scheme's activation increase.  
- **Write:** `run_write.sh` shows `"status": "ok"` and the new scheme ID in `"committed"`.  
- **Sleep cycle:** `run_sleep_cycle.sh` shows increased canvas count and a committed abstraction.

Happy testing. :)
