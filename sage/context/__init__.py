"""SAGE context layer — the scaffolding that shapes the model's world.

The context window IS the model's epistemic horizon on a forward pass.
Everything here serves one question: what goes in the window?

Submodules:
    mrh — Markov Relevant Horizon: typed, swappable blocks that
          compose the per-invocation context.

See: shared-context/forum/phase-5-mrh-context-architecture-design.md
"""
