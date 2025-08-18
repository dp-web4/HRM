# Entity Memory vs. Sidecar: Comparison & Integration Map

## 1. Side-by-Side Comparison

| Dimension | **Entity Memory** | **Sidecar (Transformer-Sidecar)** | Notes / Implications |
|---|---|---|---|
| **Primary purpose** | Registry + reputation for *entities* (sensors/effectors/dictionaries); provenance; availability; trust over contexts | Selective, affect-gated *experience memory* for the agent; fast associative recall | Separate concerns: “*who to use*” vs “*what we experienced*” |
| **Core objects** | Entities, roles, episodes (entity use), reputations, retrieval plans | Traces, keys, embeddings, eligibility/affect signals (SNARC) | EM uses Sidecar episodes as evidence signals |
| **Granularity** | Coarser: per entity/pipeline, per context bucket | Finer: per span/chunk/interaction, token-level or segment-level | EM aggregates; Sidecar stores raw-ish |
| **Temporal focus** | Long-horizon trends & decay (days→months) | Short→mid horizon recency & consolidation (seconds→days; sleep compaction) | Complementary time constants |
| **Trust math** | Reputation curves, context-conditioned priors, caps; conflict/staleness penalties | SNARC-weighted salience → write/keep; recall strength | EM’s trust can consume Sidecar’s SNARC as signals |
| **Read path** | Query by role/domain/context to *select entities* and weight them | Recall by key/situation to *retrieve experiences* | Selection (EM) then reasoning (Sidecar-aided) |
| **Write path** | Append “entity episode” with outcome & SNARC; update stats/decay | Append traces opportunistically; consolidate during sleep | Both are append-heavy; different schemas |
| **Provenance** | Strong: signer, build hash, license, device prefs, install state | Weaker/implicit: context of traces, model snapshots | EM is the provenance ground truth |
| **Availability** | Knows installed vs retrievable; sidecar-style *retrieval plan* for entities | N/A (except where memories persist on disk) | EM can request installs/loads |
| **Device placement** | Host DB (SQLite/LMDB/DuckDB); hot indices cached; GPU-side hints mirrored | GPU-adjacent vector stores; on-device KV for fast recall | Split placement; mailbox ties them |
| **APIs** | Search/select entities; record episode; get reputation; get install plan | Put/recall trace; consolidate; similarity search | Clean seams; keep APIs small |
| **Mailbox use** | Heartbeats, entity_missing, episode_result → update reputation | Recall hits, salience events, sleep summaries → influence EM | Both publish/subscribe; decouple modules |
| **Sleep role** | Batch update reps (low weight for synthetic), decay, snapshot | Generate augmentations, replay, compress, link traces | Joint nightly cycle |
| **Security** | Attestation & policy caps; quarantine on anomaly | Data hygiene; PII controls in traces | EM enforces policy; Sidecar respects it |
| **Failure modes** | Bad priors/overtrust; stale reputations | Over-retention/noisy recall; drift | Cross-checks: conflict reduces trust; sleep prunes |

---

## 2. Integration Patterns

### A) Parallel Modules + Bridge (recommended)
- Keep **Entity Memory (EM)** and **Sidecar** separate.
- Add a thin **Bridge** that:
  - Consumes Sidecar SNARC summaries as *signals* for EM’s reputation updates.
  - Publishes EM trust priors back to the scheduler/selector.
- Pros: clean separation, easy to evolve each; minimal coupling.
- Cons: one more tiny component.

### B) Entity Memory as Superset
- EM owns entities **and** stores a *summary view* of Sidecar traces (not raw).
- Sidecar still holds raw traces; EM periodically ingests aggregates.
- Pros: fewer moving parts at API surface.
- Cons: schema bloat; risk of mixing concerns.

### C) Sidecar-backed Feature Store
- Sidecar exposes a “reputation features” view; EM is a thin index over it.
- Pros: maximum reuse of Sidecar infra.
- Cons: EM becomes dependent on Sidecar uptime/format.

**Recommended**: **A) Parallel + Bridge** — clearest contracts, lowest coupling.

---

## 3. Minimal Bridge Contracts

**From Sidecar → EM (signals):**
```json
{
  "kind":"snarc_summary",
  "episode_id":"ep:...",
  "entity_id":"entity:asr/openen@1.3#...",
  "ctx_key":"audio@16k|lang=en",
  "score_components":{"reward":0.9,"coherence":0.78,"conflict":0.1},
  "latency_ms":95,
  "ts":"..."
}
```

**From EM → Scheduler/H-module (priors):**
```json
{
  "kind":"entity_prior",
  "entity_id":"entity:asr/openen@1.3#...",
  "ctx_key":"audio@16k|lang=en",
  "trust_prior":0.86,
  "trust_cap":0.92,
  "staleness_penalty":0.03
}
```

---

## 4. Decision Heuristics (When to Consult Which)

- **Choosing *who* to invoke** (ASR vs. multilingual ASR vs. “low-trust peripheral”): **EM**
- **Choosing *what* to recall** (similar past utterance/context): **Sidecar**
- **Nightly updates**: Sidecar synthesizes; EM adjusts reputations with low weight
- **Conflicts** (vision contradicts ASR): EM applies conflict penalty; Sidecar retains the episode with *conflict tag* for future analysis

---

## 5. Example Pipeline (Speech → Cognition → Speech)

1. **Registry** lists available ASR entities (installed + retrievable).  
2. **EM** ranks by context (16k/en, device budget).  
3. Selected ASR runs → **Sidecar** logs traces & SNARC.  
4. LLM reasoning uses Sidecar recalls (similar past Q&A).  
5. **EM** records entity episode with outcome (reward/coherence/latency).  
6. **TTS** selection repeats steps 1–5 for output path.  
7. **Sleep**: Sidecar augments/replays; EM updates reputations.

---

## 6. Summary

- **Entity Memory** = trust + provenance + availability of entities.  
- **Sidecar** = episodic/experiential traces with affect gating.  
- Integration via a **bridge** lets EM use Sidecar’s SNARC signals while preserving separation.  
- Together, they provide both *who to trust* and *what to recall*, spanning different timescales and abstraction levels.
