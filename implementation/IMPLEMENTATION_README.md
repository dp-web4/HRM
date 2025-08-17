
# HRM Integration and System Philosophy

This repository contains experimental implementations of components intended to support the **SAGE architecture** — a framework for learned coherence in AI. The approach parallels system design in engineering domains: modular pieces, carefully integrated, form a system greater than the sum of its parts.

## Philosophy
- **Architectural role**: focus is on integration and orchestration, not on re-inventing submodules.
- **Grafting known pieces**: HRM, Sidecar Memory, and Totality/SubThoughts are external projects; here they are treated as subsystems to graft into a coherent whole.
- **System-level coherence**: success is not "a function runs" but "state evolves in a coherent way" across modules.

## Principles
- **Stay on GPU**: Treat GPU as the shared bus for cognition. Modules exchange data directly on-device via mailboxes.
- **CPU as arbiter**: CPU coordinates, persists, and arbitrates — but does not haul data back and forth unless necessary.
- **Trust but verify**: Always compare state *before vs. after*. Victory requires measurable change, not just a running stub.
- **Clear interfaces**: Every component should expose an interface that can be audited without digging into internals.

## Roles
- **Vision**: Define architecture, integration points, and success criteria.
- **AI Assistants (Claude, GPT)**: Implement plumbing, propose scaffolding, generate tests.
- **Oversight**: Ensure declared successes match real state changes.

## Context
Just as in automotive design: engines, transmissions, and suspensions can be sourced or adapted, but their integration into a working vehicle is the true act of design. Here, HRM provides reasoning loops, Sidecar offers memory persistence, and Totality explores sub-thought orchestration. SAGE ties these into a coherent intelligence loop.

## Documents
- `GPU_MAILBOX.md` — technical design for GPU-resident mailboxes.
- `STUB_ALERT_APPENDIX.md` — overview of current stubs and placeholders.
- `COMPREHENSIVE_README.md` — system-wide instructions, setup, and tests.

This repo is exploratory: scaffolding, stubs, and partial implementations are expected. The value lies in testing hypotheses, iterating rapidly, and holding rigor around coherence and state evolution.
