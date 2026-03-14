# HRM Changelog

All notable changes to the HRM (Hierarchical Reasoning Model) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Multimodal Plugin Integration + Defensive Trust (2026-03-14)
- Generic plugin execution bridge: `_execute_via_orchestrator()` routes all modalities through IRP orchestrator
- Smart mock detection: observations with real sensor payloads → orchestrator; heartbeat data → mock path
- Config-gated real sensor sources: `vision_source` (file/camera), `audio_source` (file/mic)
- `_poll_vision_sensor()` / `_poll_audio_sensor()` with PIL/OpenCV/wave backends
- Fixed VisionIRP device mismatch (models not moved to configured device)
- TTS effector wiring: `enable_tts=True` loads AudioOutputEffector (NeuTTS Air) into effect pipeline
- Fixed mock effector action mismatch: 'speak'/'display' actions now handled
- Full effect pipeline: sensor → plugin → EffectExtractor → PolicyGate → EffectorRegistry → execute

### Changed - Defensive Trust Model (2026-03-14)
- Plugin trust initialization: 1.0 → 0.0 (zero-base, evidence-earned)
- Sensor trust initialization: 1.0 → 0.0 (zero-base)
- Mock telemetry: hardcoded 0.9 → honest 0.0 with `mock: True` flag
- Trust updater skips mock-flagged results (trust must be earned from real execution)
- All default trust fallbacks corrected from 1.0 → 0.0
- Probe budget (2% ATP floor) breaks bootstrap deadlock for untrusted plugins
- First-contact bump: first real execution → plugin + sensor trust = 0.1, then V3 EMA
- Silence decay for mock-executed plugins (0.001/cycle, floor 0.1 = aware-but-unconfirmed)
- Trust persists across restarts via daemon_state.json

### Added - Tool Use Live Activation on Nomad (2026-03-06)
- Tool use pipeline end-to-end on Nomad (gemma3:4b, T2/xml_tags grammar)
- All 7 built-in tools verified: get_time, calculate, web_search, web_fetch, read_file, write_note, peer_ask
- Dashboard multi-turn conversation memory (tracks conversation_id across messages)
- MemoryHub SQLite storage (was silently broken due to code ordering bug)
- Residual `<tool_call>` XML stripping from responses exceeding max_tool_rounds
- `SAGE:` prefix deduplication when model echoes prompt suffix
- `web_fetch` policy: `elevated` → `standard`, `max_chars`: 2000 → 4000
- Tool restraint guidance in xml_tags grammar injection prompt

### Added - PolicyGate Phase 4-5a (2026-03-05/06, autonomous)
- Phase 4: Experience buffer integration for policy decisions
- Phase 5a: Trust weight learning with salience-weighted compliance tracking
- 29/29 tests passing across Phase 4 and 5a

### Added - Snapshot State Infrastructure (2026-03-01)
- `sage/instances/snapshot.py` — Library for timestamped state snapshots with archive and manifest
- `sage/scripts/snapshot_state.py` — CLI entry point (`python3 -m sage.scripts.snapshot_state --machine <name>`)
- `sage/scripts/nomad_raising.sh` — Template raising script with snapshot + auto-commit + push
- `InstancePaths.snapshot()` method with JSON validation, identity archiving, and pruning
- `InstancePaths.list_instances()` for fleet-wide operations
- Nomad, McNugget, CBP, Thor all have `snapshots/` directories
- Live state files (identity, experience_buffer, peer_trust, daemon_state) gitignored across all instances

### Added - Seed Identity v2 + Raising Guide (2026-02-28/03-01)
- `sage/instances/_seed/identity.json` — Rewritten to schema v2 with federation, capacity, phase transitions
- `sage/instances/_seed/RAISING_GUIDE.md` — 207-line guide encoding 117+ sessions of raising knowledge
- `sage/instances/init.py` — `--operator-name` CLI arg, `__OPERATOR__` placeholder processing
- `sage/raising/scripts/ollama_raising_session.py` — Loads RAISING_GUIDE.md, injects raising preamble
- Operator relationship replaces hardcoded "dennis" in seed template
- Frozen weights awareness, partnership framing, federation context in every new instance

### Added - Instance Directory Separation (2026-02-28)
- Each machine+model pair gets self-contained dir under `sage/instances/<slug>/`
- `sage/instances/resolver.py` — `InstancePaths` resolver (resolution: SAGE_INSTANCE → SAGE_MACHINE+SAGE_MODEL → detect_machine())
- `sage/instances/init.py` — Bootstrap new instances from seed template
- Migration script for existing scattered state files (non-destructive)
- 7 active instances: thor, sprout, mcnugget, legion (x2), nomad, cbp

### Added - SAGE Society Infrastructure (2026-02-28)
- `sage/federation/fleet.json` — Static registry of 6 machines (IPs, ports, LCT IDs)
- `sage/federation/fleet_registry.py` — FleetRegistry with `get_peers()`, `get_gateway_url()`
- `sage/federation/peer_monitor.py` — Background health polling every 30s
- `sage/federation/peer_client.py` — HTTP client for peer-to-peer `/chat` messaging
- `sage/federation/peer_trust.py` — Per-peer T3 scores with EMA updates and JSON persistence
- Network effector wired for real peer messaging (was no-op)
- ATP RewardPool for conservation-safe task delegation

### Added - PolicyGate Phase 2: Consciousness Loop Integration (2026-03-01)
- PolicyGate integrated at consciousness loop step 8.6 (effect filtering)
- 50-cycle integration test: 4 state transitions, 19 plugins, 89.83 ATP consumed
- Accountability frame adapts to metabolic state (NORMAL/DEGRADED/DURESS)
- Policy decisions recorded as SNARC experiences
- Fractal self-similarity validated: "plugin of plugins of plugins"

### Fixed - Small Model Compatibility (2026-02-28)
- Rewrote `_build_conversation_prompt()` from structured metadata to first-person prose
- Added `_resolve_sender_name()` to map system senders to operator name from identity LCT
- Fixed dashboard sender from `dashboard@localhost` to `operator` (prevented TinyLlama generating "Dashboard:" character)
- Fixed `OllamaIRP.init_state()` call — try two-arg form first, fall back to one-arg for IRP contract compatibility

### Added - Documentation Honesty Pass (2026-02-27)
- All docstrings and status docs now split claims into "what's real" vs "what's mocked"
- Removed "production-ready" language across all files (R&D terminology only)
- Three SAGE improvements: ATP token coupling, DREAM consolidation to disk, response accessor

### Added - Unified SAGE Entry Point + LLM Wiring (2026-02-26/27)
- `SAGE.create(use_real_llm=True)` wires LLMRuntime → consciousness loop end-to-end
- `_SyncLLMAdapter` bridges async LLM for thread pool via `run_coroutine_threadsafe`
- `sage.send_message()` injects text → LLM inference on next cycle
- Real Ollama inference tested: 400 tokens, 1.3s avg, ATP 100→33.5

### Added - PolicyGate Phase 1 (2026-02-18)
- `sage/irp/plugins/policy_gate.py` — 684 lines, 8/8 tests passing
- SOIA-SAGE convergence mapping documented

### Added - FlashAttention Integration (2026-01-10)

#### Phase 1: Trust-Weighted Expert Selection
- **NEW**: `sage/core/flash_attention_expert_selection.py` - GQA-based expert selection
  - `TrustWeightedExpertAttention`: 12 query heads + 4 KV heads for 3x efficiency
  - `FlashAttentionExpertSelector`: Numpy-compatible interface for drop-in replacement
  - Uses PyTorch 2.9's built-in `F.scaled_dot_product_attention`
  - Works out-of-the-box on CUDA 13.0 / Jetson AGX Thor
  - Attention weights visualization for interpretability

#### Phase 2: Metabolic State-Dependent ATP Allocation
- **NEW**: `sage/core/flash_attention_metabolic.py` - State-specific attention patterns
  - `MetabolicAttentionAllocator`: Implements 5 metabolic states
    - WAKE: Full bidirectional attention (distributed allocation)
    - FOCUS: Causal attention (99.76% concentration, sequential inhibition)
    - DREAM: Random dropout (exploration mode)
    - CRISIS: Sharp softmax (97.27% emergency concentration)
    - REST: Standard attention (consolidation)
  - `FlashAttentionMetabolicAllocator`: Numpy-compatible wrapper
  - Gini coefficient analysis validates biological parallels

#### Documentation
- **NEW**: `sage/docs/FLASH_ATTENTION_INTEGRATION.md` - Complete integration guide
  - 3 integration points identified (expert selection, metabolic, sensor fusion)
  - 4-week implementation timeline
  - Migration guide with configuration examples
  - Performance benchmarks and validation plan

### Technical Details

**Key Discovery**: PyTorch 2.9 includes built-in FlashAttention via `F.scaled_dot_product_attention`
- ✅ No standalone `flash-attention` package required
- ✅ Works immediately on CUDA 13.0 (no compilation needed)
- ✅ Supports all key features: causal, GQA, custom scale, multiple dtypes
- ✅ Comparable performance to standalone package

**Performance Gains**:
- 3x efficiency from Grouped Query Attention (12 Q heads : 4 KV heads)
- O(N) memory scaling instead of O(N²)
- 0.33-2.39ms per forward pass on Thor
- 1.7-6.1M tokens/sec throughput

**Integration Status**:
- ✅ Phase 1: Trust-weighted expert selection (COMPLETE)
- ✅ Phase 2: Metabolic attention allocation (COMPLETE)
- 🚧 Phase 3: Multi-sensor fusion (PLANNED)
- 🚧 Phase 4: Production integration (PLANNED)

### Context

This resolves a weeks-long investigation into FlashAttention compatibility on CUDA 13/PyTorch 2.9.
The standalone `flash-attention` package builds successfully but doesn't import correctly on Thor.
PyTorch's built-in implementation is the correct solution for production deployment.

**Provenance**:
- Investigation: 2025-12 (Multiple sessions attempting standalone package)
- Solution Discovery: 2026-01-10 (PyTorch built-in flash attention)
- Implementation: 2026-01-10 (Phases 1-2 complete)

**References**:
- FLASH_ATTENTION_SOLUTION.md - Technical solution documentation
- test_pytorch_flash_attention.py - Comprehensive test suite
- sage/docs/FLASH_ATTENTION_INTEGRATION.md - Integration guide

---

## Historical Releases

### [0.1.0] - 2025-11-20
- Initial HRM repository setup
- Basic hierarchical reasoning architecture
- Trust-based expert selection (numpy implementation)
- Metabolic state attention manager
- SNARC memory integration
- IRP (Iterative Refinement Protocol) framework
