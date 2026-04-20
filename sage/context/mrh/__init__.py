"""MRH context architecture — typed blocks that compose a forward pass's world.

The context window is the model's epistemic horizon. Every block is one
slot in that horizon. Blocks are swappable; the composition is typed.

Design principles (from fleet consensus, Phase 5):

  1. LENS, NOT DESCRIPTION (Sprout). Block content shapes what the model
     sees; it does not describe the model to itself. "You are playing
     a game. Your goal: reach the gem." (lens) vs "You are SAGE, a
     situation-aware governance engine..." (description). Descriptions
     crystallize on small models; lenses orient.

  2. POPULATE, THEN RENDER (Thor). Block population — any I/O, API
     calls, database queries — happens in the dispatcher. Block
     render() and summarize() are pure string formatting with no
     side effects. This keeps render latency predictable.

  3. ATP/ADP-SHAPED (Web4). render() is the ATP form (full fidelity).
     summarize() is the ADP form (compressed). The composer allocates
     context budget across blocks using priority.

  4. NEVER OMIT (fleet consensus). Under context pressure, blocks
     compress but don't vanish. Even an empty metabolic block emits
     "Status: nominal" — the model should never wonder whether a
     slot is absent vs genuinely empty.

  5. IRREDUCIBLE SENSORS, EFFECTORS, TASK. These three answer
     "what do I see", "what can I do", "why am I invoked right now".
     The model can't reason without any of them.

Public API:

    from sage.context.mrh import (
        MRHBlock, MRHContext,
        IdentityBlock, SensorsBlock, EffectorsBlock,
        MechanicsBlock, ExperientialCacheBlock,
        MetabolicBlock, TaskBlock,
    )

    ctx = MRHContext(
        identity=IdentityBlock(mode="solo_gladiator"),
        sensors=SensorsBlock(...),
        ...
    )
    system_prompt, user_turn = ctx.compose(max_tokens=30000)
"""
from sage.context.mrh.base import MRHBlock, MRHContext, BlockRenderMode
from sage.context.mrh.identity import IdentityBlock
from sage.context.mrh.sensors import SensorsBlock
from sage.context.mrh.effectors import EffectorsBlock
from sage.context.mrh.mechanics import MechanicsBlock
from sage.context.mrh.experiential import ExperientialCacheBlock
from sage.context.mrh.metabolic import MetabolicBlock
from sage.context.mrh.task import TaskBlock

__all__ = [
    "MRHBlock", "MRHContext", "BlockRenderMode",
    "IdentityBlock", "SensorsBlock", "EffectorsBlock",
    "MechanicsBlock", "ExperientialCacheBlock",
    "MetabolicBlock", "TaskBlock",
]
