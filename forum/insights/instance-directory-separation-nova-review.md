# Instance Directory Separation — Nova's Review & Next Phase

**Date**: 2026-02-28
**Context**: After implementing instance directory separation (commit `82dd2311`), Nova reviewed the architecture and provided two rounds of feedback. Key recommendations are captured here for the next iteration.

---

## What Was Implemented (Phase 1)

- `sage/instances/` package: resolver, init, migrate, sleep_capability
- 6 self-contained instance directories (sprout, thor, legion, mcnugget, cbp, nomad)
- `InstancePaths` as single source of truth for all file paths
- `_seed/` template for bootstrapping new instances
- Capability-based sleep tiers: LoRA → JSONL dream bundles → remote
- Identity-drift guard tracking (`last_sleep_mode`, `last_consolidation_at`)
- `machine_config.py` uses `instance_dir` instead of separate state paths
- Daemon, raising scripts, shell wrappers all updated with fallback to legacy paths

---

## Nova's Architectural Feedback

### 1. Species vs Instance Split

The current structure conflates "what SAGE is" (code) with "who this SAGE is" (identity/state). Nova recommends a cleaner conceptual split:

- **Species** = `sage/core/`, `sage/gateway/`, `sage/irp/` — ships with the repo, same for all instances
- **Instance** = `sage/instances/{slug}/` — per-machine identity, state, sessions

This is already directionally correct. The next step is ensuring NO instance-specific data leaks back into `sage/raising/state/` or other shared directories. The old `state/` directory should eventually be removed once all machines have migrated.

### 2. Identity Should Not Be Tied to Machine

Current: instance slug = `{machine}-{model}` (e.g., `sprout-qwen2.5-0.5b`). This means if Sprout's hardware dies, the identity is "stuck" in a machine-named directory.

Nova's recommendation: identity is portable. The machine is where it runs *now*, not who it *is*. Consider:
- Identity name (SAGE-Sprout) is a development-era convention, not a constraint
- `instance.json` manifest already separates `machine` from identity
- Future: identity migration tool (move identity between instance dirs without losing continuity)

### 3. Schema Versioning for identity.json

Current `identity.json` has no version field. As the schema evolves (new fields, renamed fields, structural changes), older files become ambiguous.

Recommendation: Add `"schema_version": 1` to identity.json (and instance.json). Migration logic can then branch on version. The seed template should set the current version.

### 4. InstanceAdapter Pattern

Rather than having every script directly access `InstancePaths.identity`, `InstancePaths.experience_buffer`, etc., wrap instance access in an adapter:

```python
class InstanceAdapter:
    """Provides structured access to instance state with validation."""
    def __init__(self, paths: InstancePaths):
        self.paths = paths

    def load_identity(self) -> dict:
        """Load and validate identity state."""
        ...

    def save_identity(self, state: dict):
        """Validate and persist identity state."""
        ...

    def append_experience(self, experience: dict):
        """Thread-safe experience buffer append."""
        ...
```

Benefits: validation on load/save, schema migration hooks, thread safety, single place for backup logic. Not urgent — current direct file access works — but becomes important as more scripts and the daemon concurrently access instance state.

---

## Nova's Sleep/Training Feedback

### 5. Sleep as Capability, Not Configuration

**Implemented.** `SleepCapability.detect()` probes runtime environment rather than reading a config flag. Three tiers:

| Tier | Requirement | What it does |
|------|------------|--------------|
| `sleep_lora` | torch + transformers + peft | Full weight updates |
| `sleep_jsonl` | writable filesystem | Dream bundle export |
| `sleep_remote` | federation peer online | Delegate to torch-capable peer |

### 6. Dream Bundles as Portable Artifacts

**Implemented.** `write_dream_bundle()` exports high-salience SNARC experiences to `instance/dream_bundles/dream_*.jsonl` with provenance headers (machine, model, LoRA hash, format version). Ollama-only nodes (CBP, McNugget, Nomad) produce these instead of running LoRA.

### 7. Remote Sleep as Federation Service (NOT YET IMPLEMENTED)

Nova's key insight: "Sleep is computationally expensive. An Ollama-only node doesn't need local torch — it needs a *peer* with torch."

Design:
- Ollama node exports dream bundle → sends to torch-capable peer via federation
- Peer runs LoRA training on the dream bundle → returns updated LoRA adapter
- Node applies adapter (if supported) or stores for next capable session
- This is a **payable ATP service** — the peer spends compute, gets ATP credit

Requires: federation messaging (partially built via PeerClient), LoRA adapter portability, ATP accounting for training services.

### 8. Identity-Drift Guard

**Partially implemented.** `SleepCapability` tracks `last_sleep_mode` and `consolidation_count`. Nova's full vision:

- Compare `last_sleep_mode` across consecutive consolidations
- If mode changes (e.g., LoRA → JSONL because machine changed), flag it
- Log drift events to instance state for longitudinal analysis
- Prevent silent identity divergence when the same SAGE runs on different hardware

Next step: Wire drift detection into the consciousness loop's DREAM state handler. Currently only records mode — doesn't yet *compare* across consolidations.

---

## Next Phase Priorities

1. **Schema versioning** — add `schema_version` to identity.json and instance.json (low effort, high value)
2. **Remove legacy fallbacks** — once all machines have pulled and migrated, remove old `state/` path fallbacks from scripts
3. **InstanceAdapter** — wrap file access with validation/migration hooks (medium effort, pays off at scale)
4. **Remote sleep federation** — dream bundle exchange between peers (depends on federation messaging)
5. **Identity migration tool** — move identity between instance dirs (important for hardware changes)
6. **Merge raising scripts** — `legion_raising_session.py` and `mcnugget_raising_session.py` are structurally identical, only paths differ. With InstancePaths, they can become one OllamaIRP runner.

---

## Related Documents

- Plan: `/home/dp/.claude/plans/lexical-sniffing-hippo.md` (original implementation plan)
- Instance resolver: `sage/instances/resolver.py`
- Sleep capability: `sage/instances/sleep_capability.py`
- Migration script: `sage/instances/migrate.py`
- Society infrastructure: `sage/federation/` (fleet.json, peer client, peer monitor)
- Identity portability insight: `forum/insights/identity-portability-first-contact.md`
