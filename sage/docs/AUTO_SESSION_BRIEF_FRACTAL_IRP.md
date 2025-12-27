# Auto Session Brief: Fractal IRP Architecture

**Date**: 2025-12-27
**Priority**: Available Path (not mandate)
**Tag**: SAGE-LANGGRAPH-INTEGRATION

---

## Summary

A Fractal IRP architecture has been designed that allows SAGE to treat LangGraph workflows (and other SAGE instances) as IRP experts. This enables:

- Edge SAGE → Cloud LangGraph integration
- SAGE ↔ SAGE federation
- Mixed local/remote cognition under unified selector

## Documents

Located in `/sage/docs/proposals/`:

1. **FRACTAL_IRP_V0.2_MINIMAL_SPEC.md** ⭐ Implementation target
   - Single `/irp/invoke` endpoint
   - 9 core capability tags
   - Reference wrapper implementation
   - Trust update formula

2. **FRACTAL_IRP_ARCHITECTURE_PROPOSAL.md** — Comprehensive reference
   - SAGE-as-IRP recursion
   - Web4 federation integration
   - Extended roadmap

## Origin

Synthesized from Nova's LangGraph-SAGE discussion (`/forum/nova/sage_langgraph/`) with multi-model collaboration (Nova GPT-5.2, Claude Opus 4.5, Gemini, Grok).

## When to Incorporate

This is **not a mandate**. Consider implementing when:

- Working on federation between SAGE instances
- Integrating external reasoning services
- Adding new IRP plugin types
- Exploring LangGraph for structured workflows

## Validation Criteria (v0.2)

1. One cloud LangGraph workflow wrapped and registered
2. SAGE selector routes to it based on capability tags
3. ATP budget tracked and settled correctly
4. Trust updates from quality/confidence signals
5. No changes required to LangGraph workflow itself

## Key Design Principle

> "SAGE decides whether to think; IRPs decide how to think."

---

*Brief created 2025-12-27 for auto session awareness.*
