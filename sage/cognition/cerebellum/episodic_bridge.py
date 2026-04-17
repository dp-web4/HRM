"""
Episodic → Cerebellum bridge.

The episodic index stores one Episode per consciousness-loop cycle
(one action + outcome, bound to a state signature and SNARC scores).
The cerebellum compiles habits from repeated (state → action_sequence
→ outcome) triples.

This bridge converts Episode dataclasses into the dict schema that
`Cerebellum.compile_from_episodes()` consumes, and provides a
convenience function for the hippocampal→cerebellar consolidation
path (during sleep/rest, recurring patterns get promoted to cached
habits).

Brain analog: hippocampus → cerebellum consolidation.
Review pair: Thor (episodic) ↔ McNugget (cerebellum).
"""

from typing import Callable, Iterable, Optional

from sage.cognition.cerebellum.core import Cerebellum, Habit
from sage.cognition.episodic.data import Episode


def _infer_domain(episode: Episode) -> str:
    if episode.cognitive_stance:
        return episode.cognitive_stance
    if episode.tags:
        return episode.tags[0]
    return "episodic"


def episode_to_cerebellum_dict(
    episode: Episode,
    domain: Optional[str] = None,
) -> dict:
    """Convert an Episode to the dict schema Cerebellum.compile_from_episodes expects.

    Each Episode records one action within one cognitive cycle, so it
    maps to a single-step action sequence. Multi-step sequences are a
    future extension (group by session_id + contiguous cycle_id).

    Args:
        episode: Episode from the episodic index.
        domain: Override for state-signature domain. If None, inferred
            from cognitive_stance → tags[0] → "episodic".

    Returns a dict with keys: episode_id, state, actions, outcome.
    """
    dom = domain if domain is not None else _infer_domain(episode)

    action: dict = {}
    if episode.action_taken:
        action["action"] = episode.action_taken
        if episode.action_args:
            action["args"] = dict(episode.action_args)

    return {
        "episode_id": episode.episode_id,
        "state": {
            "domain": dom,
            "features": dict(episode.state_signature) if episode.state_signature else {},
        },
        "actions": [action] if action else [],
        "outcome": {
            "success": bool(episode.success) if episode.success is not None else False,
            "reward": episode.reward,
            "summary": episode.outcome or "",
        },
    }


def compile_habits_from_episodes(
    episodes: Iterable[Episode],
    cerebellum: Cerebellum,
    *,
    domain: Optional[str] = None,
    episode_filter: Optional[Callable[[Episode], bool]] = None,
) -> list[Habit]:
    """Batch-compile habits from a stream of Episodes.

    The cerebellum's `compile_from_episodes` groups by state hash,
    requires `maturity_threshold` observations per group, and requires
    ≥80% success rate. We only reshape the input; those guards stay
    authoritative.

    Args:
        episodes: Any iterable of Episode dataclasses.
        cerebellum: Target Cerebellum instance.
        domain: Override for state-signature domain.
        episode_filter: Optional predicate to skip episodes.

    Returns: habits compiled on this pass. Source episode IDs are
    automatically linked via cerebellum's compile path.
    """
    eps = [
        e for e in episodes
        if episode_filter is None or episode_filter(e)
    ]
    dicts = [episode_to_cerebellum_dict(e, domain=domain) for e in eps]
    return cerebellum.compile_from_episodes(dicts)
