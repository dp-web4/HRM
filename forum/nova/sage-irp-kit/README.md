# SAGE IRP Starter Kit

A compact scaffold for a **fractal / hierarchical IRP** controller suitable for SAGE.

## Layers (tiers)

0. **SubstrateReasoner** – initial draft + retrieval proposals.  
1. **ReflectiveCoherence** – contradiction checks, reasoning audits.  
2. **MetaIntent** – epistemic calibration, clarify/ask routing, boilerplate suppression.  
3. **MemoryIntegrator** – episodic/semantic staging (triples).  
4. **SynchronyLayer** – optional peer consult / consensus merge.  
5. **Witness** – self-audit; refuse on low-confidence/high-risk.

## Quick start

```bash
python sage_irp.py
```



You’ll see a toy response shaped as:

```
Answer: <concise>
Reason: <short evidence-first rationale>
Epistemic: likely (~0.7)
Next: proceed
```

## Swap these stubs

- Replace `DummyLLM` with your **qwen2.5-0.5b-epiprag** bindings (draft, summarize, inconsistency).
- Replace `InMemoryBackend` with your **semantic graph + vector store + LCT ledger**.
- Implement real `retrieve`, `peer_consult`, and tool exec in `SAGEController._execute`.

## Policy example

```
risk_aversion: 0.45
max_actions_per_turn: 3
clarify_threshold: 0.4
boilerplate_blocklist:
  - "^As an AI"
  - "\\bI cannot\\b"
  - "\\bI am unable\\b"
```

## Notes

- Keep tiers **fast, single-pass**; edge-friendly for Jetson.
- **MetaIntent** is where your calibration head + DPO preferences live.
- **Witness** centralizes refusal on **risk × low confidence** (non-performative).
- **SynchronyLayer** is optional on edge; enable when peers are reachable.