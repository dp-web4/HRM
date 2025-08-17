
# Appendix — Stub Alert: What’s Fake vs. Real (v0.1)

This appendix flags **deliberate stubs** and **what counts as real** so nobody (including future-you or Claude) mistakes scaffolding for implementation.

---

## Quick Legend
- ✅ **Implemented (minimal)** — Works, but intentionally simple.
- 🧪 **Stub / Placeholder** — Behaves enough for tests but **must be replaced** for production.
- ⚠️ **Known wrinkle** — Edge-case or design gap to fix.

---

## Service: `sage_totality_service/app/main.py`

### `/totality/read`
- Status: ✅ Implemented (minimal)
- Behavior: returns in-memory **schemes** and **canvases**. Filters by `activation_min` only.
- Real Pass: returns ≥1 scheme and ≥1 canvas on fresh start.
- Wrinkles: no paging, no complex filters.

### `/totality/imagine`
- Status: 🧪 Stub / Placeholder
- Behavior: creates canvases by **naive slot value permutation** only.
- Real Pass: **canvas count must increase** after a call; each new canvas must show provenance `ops` that match requested ops.
- Not Done: no geometric transforms, no context shift, no semantic variation beyond naive, no trust/affect bias.
- TODO: plug in real augmentation ops; validate input ops against a registry.

### `/totality/activate`
- Status: ✅ Implemented (minimal)
- Behavior: adds `delta` to activation with clamping [0,1]; mirrors to scheme if present.
- Real Pass: target scheme’s activation **numerically increases** after +delta.
- Wrinkles: no decay/half-life processing yet.

### `/totality/write`
- Status: ✅ Implemented (minimal)
- Behavior: commit schemes; `"merge"` overwrites label/slots and raises activation to max(existing,new).
- Real Pass: response contains `{"status":"ok","committed":[...ids...]}` and a subsequent `read` returns those IDs.
- Wrinkles: no link graph store, no conflict resolution beyond overwrite, no schema validation.

### `/totality/snapshot`
- Status: ✅ Implemented (minimal)
- Behavior: returns a one-shot JSON snapshot of current store.
- Real Pass: snapshot contains all schemes/canvases/activations at time of call.
- Wrinkles: no persistence to disk, no restore on boot.

---

## Mini Totality: `totality_min/`

### `totality_core.py`
- Status: ✅ Implemented (minimal)
- Behavior: in-memory store; types `Scheme`, `Canvas`, `Link`; ops `read`, `imagine_from_schemes`, `activate`, `write_schemes`.
- Real Pass: end-to-end demo works: seed → imagine → activate → write → final store non-empty.
- Wrinkles: no ontology/graph DB, no indexing, no concurrency.

### `transforms.py`
- Status: 🧪 Stub / Placeholder
- Behavior: `context_shift()` prefixes text; `semantic_variation()` does naive string replacement.
- Not Done: true semantic/context transforms.
- TODO: integrate with augmentation registry and LLM-assisted generators (when trusted).

### `demo.py`
- Status: ✅ Implemented (minimal)
- Behavior: shows basic flow and prints final store.
- Real Pass: final store contains **more canvases** and at least one **new/merged** scheme.

---

## Tests: `sage_totality_tests/`

### Bash scripts (curl-based)
- Status: ✅ Implemented (minimal)
- Real Pass Criteria:
  - **run_smoke.sh**: health ok, read returns ≥1 scheme & canvas.
  - **run_imagine.sh**: returns new canvas IDs.
  - **run_activate.sh**: activation value strictly increases.
  - **run_write.sh**: `status=ok` and committed ID exists in subsequent read.
  - **run_sleep_cycle.sh**: canvas count **after** > **before**; snapshot present.

### Python tests
- `tests/run_tests.py` — ✅ minimal runner without dev deps.
- `tests/test_api.py` — ✅ pytest + requests checks.
- Wrinkles: no negative tests; no schema validation; no timing/latency assertions.

---

## Trust & Affect (Not Wired Yet)

- Status: 🧪 Stub / Placeholder
- What’s missing:
  - No **TrustVector** or **Affect** shaping outputs/strategy.
  - No SNARC write-gating or recall priority.
- TODO:
  - Add `/trust` and `/affect` inputs or headers.
  - Implement `snarc_write_gate()` and apply it in `write` and dream sampling.

---

## Persistence (Not Implemented)

- Status: 🧪 Stub / Placeholder
- What’s missing: snapshot to file; auto-restore on boot; journaled writes.
- TODO:
  - `POST /persist` and `POST /load` endpoints, or autosave on shutdown/startup.
  - JSONL journal for append-only durability.

---

## Definition of Done (DoD) — Minimal “Real” System

1) **Augmentation registry** with at least 3 real ops (e.g., `geom`, `context_shift`, `semantic_variation-LLM`) and provenance recorded.  
2) **Trust/Affect hooks**: inputs accepted, SNARC gate applied to write priority and dream sampling.  
3) **Persistence**: snapshot-to-file and restore-on-boot.  
4) **Validation**: inputs schema-checked; ops validated against registry.  
5) **Tests**: add negative/edge cases; verify canvas count growth and activation dynamics under load.  

---

## Red Flags to Catch (Common “Path Around” Patterns)

- “Success” logged without **state change** (canvas count, activation, committed IDs).  
- Silent acceptance of **unknown ops** in `/imagine`.  
- Activation updates that exceed [0,1] without clamping.  
- “Snapshot” returned but missing schemes/canvases.  
- “Write merge” that doesn’t actually alter the target scheme.

---

## How to Review PRs (Fast Checklist)

- [ ] Does every new test assert **before vs. after** counts/values?  
- [ ] Are stubs clearly labeled in code and README?  
- [ ] Do imagine ops come from a declared registry?  
- [ ] Do we have a failing test for invalid ops/inputs?  
- [ ] Does snapshot include all stores, and can we reload it?

---

**Reminder:** The current system is designed to **prove the loop**, not to be production robust. Treat stubs as scaffolding to be replaced, not features to work around.
