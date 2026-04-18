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

Two consolidation grains are supported:

- single-episode: each Episode → one single-step habit candidate
  (one cycle, one action). Use `compile_habits_from_episodes`.
- trajectory: contiguous Episodes within a session collapse into one
  multi-step habit candidate keyed on the *initial* state, with the
  full ordered action sequence and the final episode's outcome. Use
  `compile_habits_from_trajectories`.

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


# ─── Multi-step trajectory path ──────────────────────────────────────


def group_episodes_into_trajectories(
    episodes: Iterable[Episode],
    *,
    max_gap: int = 1,
) -> list[list[Episode]]:
    """Group episodes into contiguous within-session trajectories.

    A trajectory is an ordered list of Episodes that share a session_id
    and have monotonically non-decreasing cycle_id values where each
    successive cycle_id is within `max_gap` of the previous one. Larger
    gaps split into separate trajectories. Episodes with empty
    session_id are each treated as their own single-element trajectory
    (no session anchor → no contiguity claim).

    Output trajectories are sorted by their initial (session_id,
    cycle_id) for determinism. Within each trajectory, episodes are
    ordered by cycle_id ascending.

    Args:
        episodes: Any iterable of Episode dataclasses.
        max_gap: Maximum cycle_id delta between consecutive episodes
            for them to remain in the same trajectory. Default 1
            (strictly contiguous). Use a larger value to tolerate
            small holes (e.g., a skipped logging cycle).

    Returns: list of trajectories, each a list of Episodes.
    """
    if max_gap < 1:
        raise ValueError("max_gap must be >= 1")

    trajectories: list[list[Episode]] = []
    by_session: dict[str, list[Episode]] = {}
    for ep in episodes:
        if not ep.session_id:
            trajectories.append([ep])
            continue
        by_session.setdefault(ep.session_id, []).append(ep)

    for session_id in sorted(by_session.keys()):
        eps = sorted(by_session[session_id], key=lambda e: e.cycle_id)
        current: list[Episode] = []
        prev_cycle: Optional[int] = None
        for ep in eps:
            if prev_cycle is None or (ep.cycle_id - prev_cycle) <= max_gap:
                current.append(ep)
            else:
                trajectories.append(current)
                current = [ep]
            prev_cycle = ep.cycle_id
        if current:
            trajectories.append(current)

    return trajectories


def trajectory_to_cerebellum_dict(
    trajectory: list[Episode],
    *,
    domain: Optional[str] = None,
) -> dict:
    """Collapse a trajectory into a single cerebellum-compile dict.

    The state signature is taken from the trajectory's *initial*
    episode — this is what the cerebellum will match against at
    recall time. The action sequence is the ordered list of
    non-empty actions across the trajectory. The outcome is the
    *final* episode's outcome (success/reward), with a summary that
    reports the trajectory length so multi-step habits are
    identifiable from their outcome_summary.

    Args:
        trajectory: Non-empty list of Episodes ordered by cycle_id.
        domain: Override for state-signature domain. If None, inferred
            from the initial episode's cognitive_stance → tags[0] →
            "episodic".

    Returns: a dict matching the schema Cerebellum.compile_from_episodes
    expects. The episode_id is taken from the *initial* episode (so
    each trajectory has a stable identifier); the full per-episode
    provenance is preserved by `compile_habits_from_trajectories` via
    multiple input dicts sharing source IDs.

    Raises ValueError if trajectory is empty.
    """
    if not trajectory:
        raise ValueError("trajectory must contain at least one episode")

    initial = trajectory[0]
    final = trajectory[-1]
    dom = domain if domain is not None else _infer_domain(initial)

    actions: list[dict] = []
    for ep in trajectory:
        if ep.action_taken:
            step: dict = {"action": ep.action_taken}
            if ep.action_args:
                step["args"] = dict(ep.action_args)
            actions.append(step)

    return {
        "episode_id": initial.episode_id,
        "state": {
            "domain": dom,
            "features": dict(initial.state_signature) if initial.state_signature else {},
        },
        "actions": actions,
        "outcome": {
            "success": bool(final.success) if final.success is not None else False,
            "reward": final.reward,
            "summary": (
                f"trajectory[{len(trajectory)}] {final.outcome or ''}".strip()
            ),
        },
    }


def compile_habits_from_trajectories(
    episodes: Iterable[Episode],
    cerebellum: Cerebellum,
    *,
    domain: Optional[str] = None,
    episode_filter: Optional[Callable[[Episode], bool]] = None,
    max_gap: int = 1,
) -> list[Habit]:
    """Batch-compile multi-step habits from a stream of Episodes.

    Episodes are grouped into trajectories (see
    `group_episodes_into_trajectories`), each trajectory is collapsed
    into a single state→sequence→outcome dict, and the resulting
    dicts are fed to the cerebellum's existing batch compile path.
    The cerebellum's maturity and success-rate guards remain
    authoritative — and they count *trajectories*, not episodes.
    A single 3-cycle trajectory contributes one observation, not
    three. This keeps `maturity_threshold` semantically aligned
    between single-episode and trajectory consolidation paths.

    Source-episode provenance carries the initial episode_id of each
    contributing trajectory (the cerebellum's compile path collects
    one ID per input dict). Per-step episode IDs are not surfaced on
    the Habit; if needed for richer provenance, walk back from the
    initial episode via the EpisodicIndex using session_id +
    contiguous cycle_id.

    Args:
        episodes: Any iterable of Episode dataclasses.
        cerebellum: Target Cerebellum instance.
        domain: Override for state-signature domain.
        episode_filter: Optional predicate applied per Episode before
            grouping.
        max_gap: Maximum cycle_id delta inside a trajectory.

    Returns: habits compiled on this pass.
    """
    eps = [
        e for e in episodes
        if episode_filter is None or episode_filter(e)
    ]
    trajectories = group_episodes_into_trajectories(eps, max_gap=max_gap)

    dicts = [
        trajectory_to_cerebellum_dict(traj, domain=domain)
        for traj in trajectories
        if traj
    ]
    return cerebellum.compile_from_episodes(dicts)
