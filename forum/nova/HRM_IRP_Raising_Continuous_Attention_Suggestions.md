
# HRM: Integrating IRP + Raising into a Continuous Attention Orchestrator (Suggestions Draft)

**Date:** 2026-02-24  
**Goal:** Turn the existing IRP plugin stack + raising/sleep-training pipeline into a **continuous attention orchestrator** that can (a) stay alive, (b) decide what to attend to next, (c) call tools/plugins, (d) load and use a local LLM on demand, and (e) periodically consolidate experience into LoRA checkpoints (“sleep”).

This is a **design and implementation suggestion doc**, not a status report.

---

## 0) Design constraints (keep these explicit)

1. **Always-on ≠ always-inferencing.** The orchestrator can be always running while keeping the LLM cold most of the time.
2. **Separation of concerns:**  
   - **Attention kernel**: lightweight, event-driven, deterministic, runs continuously.  
   - **LLM runtime**: heavier, on-demand, sandboxed, can be restarted independently.  
   - **Sleep trainer**: isolated training subprocess (already validated as necessary on CUDA/Jetson-class environments).
3. **Artifacts are the truth.** Sessions, buffers, checkpoints, and manifests must be the durable substrate, not RAM state.
4. **Cross-machine discipline:** enforce track/machine separation via guardrails (Sprout vs Thor vs McNugget) and explicit manifests.

---

## 1) Target architecture overview

### A. The “continuous attention” loop (kernel)
A single long-running process that:
- Watches **events** (timers, user inputs, file changes, sensor inputs, plugin outputs)
- Maintains a **working context** (short horizon + selected memory snippets)
- Dispatches **IRP plugins** under ATP budgets
- Escalates to **LLM calls** only when needed
- Emits **actions** and **experience traces**
- Triggers **sleep** when conditions are met

### B. Two-tier cognition
- **Tier 0 (Always-On):** low-cost attention, routing, resource control, anomaly detection, basic heuristics.
- **Tier 1 (On-Demand):** local LLM inference + structured tool-use loops for deep reasoning.

This mirrors how you avoid keeping a big model “hot” constantly while still being “alive.”

---

## 2) Core components to add / formalize

### 2.1 Attention Kernel (new module)
**Responsibilities**
- Event loop + scheduler
- ATP budgeting and trust-weight routing
- Context assembly (prompt packer)
- Tool/IRP dispatch controller
- Health supervision and crash recovery
- State machine for modes: `idle`, `observe`, `deliberate`, `act`, `sleep`

**Implementation sketch**
- `sage/attention/kernel.py`
- `sage/attention/state_machine.py`
- `sage/attention/event_bus.py` (or minimal queue)

**Key output artifacts**
- `attention_tick.jsonl` (one record per tick: input events, decisions, budgets)
- `action_log.jsonl` (already exists conceptually—make it the canonical action stream)
- `context_manifest.json` (what went into the prompt / what was recalled)

### 2.2 LLM Runtime Service (separate process)
**Responsibilities**
- Load/unload models
- Maintain KV cache (optional) or stay stateless (initially stateless is easier)
- Provide an API: `generate(prompt, constraints, tool_schema) -> response`
- Support multiple backends:
  - Jetson: local runtime (transformers/gguf/whatever you run)
  - Mac: Ollama path already exists for IRP integration (use that as an adapter)

**Implementation sketch**
- `sage/llm/runtime_server.py` (HTTP or unix socket)
- `sage/llm/backends/{ollama,transformers,llamacpp}.py`
- `sage/llm/model_manifest.json` (what model(s) exist on this host)

**Why separate process?**
- Isolation from CUDA state issues
- Restartability
- Clear resource boundary
- Simplifies “cold start” vs “hot path” decisions

### 2.3 IRP Plugin Layer (existing, but needs contracts)
**Formalize plugin interface**
Each IRP plugin should expose:
- `capabilities()` (what inputs/outputs)
- `estimate_cost(context)` (ATP estimate)
- `run(context, budget)` (async)
- `telemetry()` (latency, token usage, success rate, trust hints)

**Add a schema**
- Standard input bundle: `IRPContext { goal, working_memory, sensory_inputs, constraints }`
- Standard output: `IRPResult { claims, evidence, actions_suggested, confidence, deltas }`

### 2.4 Experience Capture + Sleep Trigger (tie-in to raising)
Experience capture should be **continuous** (not just “after sessions”):
- Every action/tool call becomes a candidate experience
- Every contradiction or repair becomes a candidate experience
- Every “novelty spike” becomes a candidate experience

Sleep trigger conditions should be explicit and cheap:
- `experience_buffer.size >= N`
- `salience_sum >= S`
- `time_since_last_sleep >= T`
- `idle_window >= W` (don’t sleep while actively serving)

The sleep trainer remains a subprocess with a strict manifest in/out.

---

## 3) Control flow: the continuous attention cycle

### 3.1 Tick types
Define 3 tick types:

1. **Reactive tick** (fast): route immediate events to lightweight plugins; minimal prompting.
2. **Deliberative tick** (medium): assemble context and ask LLM to choose next actions (tool selection).
3. **Sleep tick** (slow/offline): train LoRA + produce checkpoint + update manifest.

### 3.2 Minimal state machine
Suggested modes:

- `IDLE` — listen/watch; run lightweight observers
- `FOCUS` — gather context; allocate ATP to plugins
- `THINK` — invoke LLM if needed
- `ACT` — execute tool actions; capture experience
- `SLEEP` — consolidate experiences via LoRA subprocess
- `RECOVER` — restart subsystems, roll back checkpoints, degrade gracefully

This mode machine should be persisted (so a reboot continues in a coherent state).

---

## 4) How IRP and Raising interlock (the key integration)

### 4.1 IRP produces “experience atoms” continuously
Instead of raising being “a special mode,” treat it as the **learning substrate** that is always collecting:

- Observation → interpretation → action → outcome → repair

Each IRP output should be normalized into an “experience atom”:

```json
{
  "ts": "...",
  "goal": "...",
  "context_manifest": {...},
  "actions": [...],
  "outcome": {...},
  "self_eval": {...},
  "salience": 0.0,
  "tags": ["tool_use", "format_failure", "repair", "novelty"]
}
```

### 4.2 SNARC/salience scoring becomes an always-on filter
Run salience scoring as a cheap pass:
- Score atoms
- Keep top-K per time window
- Optionally downweight “runner artifacts” (see regression issues)

### 4.3 Sleep turns selected atoms into training episodes
During sleep:
- Convert atoms into training examples (ChatML or your target format)
- Weight loss by salience (already in your sleep-training concept)
- Save LoRA checkpoint + manifest entry
- Emit a **post-sleep evaluation task**

### 4.4 Post-sleep evaluation must be automatic
After a sleep cycle:
- Run a small fixed test suite (“behavioral probes”)
- Compare to previous checkpoint
- If regression exceeds threshold, mark checkpoint “candidate” not “current”

This is how you avoid silent drift.

---

## 5) “Always-on” local LLM on demand: practical scaffolding

### 5.1 Cold start strategy
The kernel decides whether to spin up the LLM runtime based on:
- urgency (user waiting)
- complexity estimate (plugins disagree, low confidence)
- novelty spike (new domain)
- explicit request (“think deeper”)

### 5.2 Hot/cold lifecycle management
- Keep LLM runtime **off** by default on constrained hosts
- Warm for a fixed window (e.g., 5–15 minutes after use)
- Shut down when idle to reclaim GPU memory

### 5.3 Prompt packing and “context manifests”
Make prompt assembly deterministic and auditable:
- `prompt.md` (rendered)
- `context_manifest.json` (structured sources)
- `tool_schema.json` (function signatures)

This makes failures diagnosable (“it hallucinated because we fed it X”).

---

## 6) Trust + ATP: how to make it real (incrementally)

### 6.1 Trust signals
Start with hard, measurable signals:
- success/failure rate
- latency vs estimate
- repair burden introduced
- contradiction rate
- user rejection rate

Only later add “semantic” trust signals.

### 6.2 ATP enforcement
Today, ATP can remain abstract, but enforceable:
- deny plugin calls that exceed remaining budget
- degrade to cheaper plugins
- shorten LLM context / reduce max tokens
- choose a smaller model variant if available

---

## 7) Minimal “integration plan” (doable in phases)

### Phase 1: Make the attention kernel real
- Implement the kernel + state machine
- Define the canonical logs/manifests
- Integrate existing IRP plugin calls
- No training automation changes yet

**DoD:** kernel runs 24/7; can process events; can call plugins; produces auditable logs.

### Phase 2: Add LLM runtime as a service
- Backend adapters (Ollama on McNugget; Jetson backend on Sprout/Thor)
- Tool-use loop controller (function calling scaffolding)
- Health supervisor (restart LLM process on crash)

**DoD:** kernel can “ask the LLM” and use tool calls safely.

### Phase 3: Wire continuous experience capture → sleep trigger
- Experience atoms produced from actions and LLM/tool loops
- Salience scorer + buffer
- Sleep trigger policy
- Sleep subprocess with strict manifest in/out

**DoD:** sleep runs automatically when conditions met; produces checkpoint + manifest; no CUDA bleed.

### Phase 4: Add automatic evaluation + rollback
- Behavioral probe suite
- Regression gates
- Checkpoint promotion policy (`candidate` → `current`)

**DoD:** no silent regressions; training is bounded by measurable improvement.

### Phase 5: Cross-machine federation (optional)
- Share only manifests + selected experiences (not raw everything)
- Compare checkpoints across machines
- Promote “upstream” candidates carefully

**DoD:** distributed raising without uncontrolled drift.

---

## 8) Specific files/artifacts to standardize

Suggested canonical layout (per host):

```
model-zoo/
  host/
    identity.json
    model_manifest.json
    checkpoints/
      sleep/
        cycle_001/
          adapter_model.safetensors
          lora_config.json
          train_metrics.json
          manifest_entry.json
    buffers/
      experience_buffer.jsonl
      salience_index.json
    logs/
      attention_tick.jsonl
      action_log.jsonl
      llm_calls.jsonl
      plugin_telemetry.jsonl
    status/
      current.json
      last_good_checkpoint.json
```

Key: everything important is **append-only logs + small manifests**.

---

## 9) Hard problems (name them now)

1. **Runner artifacts vs cognition:** scripts can create “behavior” that looks like model change. Instrument runner invariants.
2. **Format constraints vs LoRA priors:** mixing structured output training with strict format constraints can destabilize; separate policy conditioning from content training.
3. **Catastrophic drift across cycles:** requires evaluation gates and rollback.
4. **Resource isolation:** training/inference must not share CUDA state implicitly; keep subprocess boundaries.
5. **Identity persistence vs confabulation:** identity files can amplify confabulation unless they’re treated as constraints + evidence, not prompts.

---

## 10) The honest claim (what you can say when this is built)

When the above is implemented, HRM/SAGE can honestly claim:
- There is a continuously running attention kernel.
- It can invoke plugins and tools under a resource budget.
- It can load a local LLM on demand and scaffold tool use.
- It can consolidate selected experiences into LoRA checkpoints during scheduled sleep.
- It keeps auditable logs and can roll back regressions.

That’s the “always-on orchestrator + raising pipeline” reality, without mythology.
