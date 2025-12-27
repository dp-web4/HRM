# SAGE Architecture Proposals

This directory contains formal architectural proposals for SAGE development.

## Active Proposals

### [FRACTAL_IRP_V0.2_MINIMAL_SPEC.md](./FRACTAL_IRP_V0.2_MINIMAL_SPEC.md) ⭐ Start Here

**Status**: Implementation-Ready
**Date**: 2025-12-27
**Authors**: Nova (GPT-5.2), Claude Opus 4.5, Dennis (dp-web4)

**Summary**: Minimal integration spec that proves fractal IRPs work end-to-end. This is the implementation target.

**Key Features**:
- Single `/irp/invoke` endpoint
- 9 core capability tags
- Accounting vs Trust hard separation (quality ≠ confidence ≠ cost)
- LangGraph wrapper reference implementation
- ATP settlement thresholds

**Validation Criteria**: One cloud LangGraph wrapped and working with SAGE selector.

---

### [FRACTAL_IRP_ARCHITECTURE_PROPOSAL.md](./FRACTAL_IRP_ARCHITECTURE_PROPOSAL.md)

**Status**: Comprehensive Reference (v1.0-draft)
**Date**: 2025-12-27
**Authors**: Dennis (dp-web4), Nova (GPT-5.2), Claude Opus 4.5

**Summary**: Full architectural vision for Fractal IRP including SAGE-as-IRP recursion, Web4 federation, extended descriptor schema, and 10-week adoption roadmap.

**Key Contributions**:
1. SAGE-as-IRP: Any SAGE instance can be wrapped as an IRP expert
2. LangGraph-as-IRP: Existing LangGraph deployments integrate without rewrite
3. Web4 Federation Integration: LCT identity, ATP settlement, trust propagation
4. Scale-Invariant Routing: Same SNARC × epistemic × ATP × capability-tags at all levels

**Origin**: Synthesized from Nova's sage-langgraph discussion (`/forum/nova/sage_langgraph/`) combined with comprehensive architectural review.

**Note**: Start with v0.2 for implementation. Use this document for architectural context and future extensions.

---

## Proposal Lifecycle

1. **Draft**: Initial proposal creation
2. **Review**: Multi-model peer review (Nova, Grok, Gemini)
3. **Revision**: Address review feedback
4. **Accepted**: Approved for implementation
5. **Implemented**: Code complete, proposal archived

## Review Protocol

Per the Confidence Claim Review Trigger protocol, proposals claiming significant architectural advancement should be reviewed by external models before acceptance.
