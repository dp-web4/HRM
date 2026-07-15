"""
Imagination IRP Plugin - Predictive rollout organ for SAGE consciousness loop
Version: 1.0 (2026-07-14)

Four Invariants:
1. State space: Candidate effects/actions + their imagined rollouts
   (predicted next-states under a world-model)
2. Noise model: World-model uncertainty (per-prediction confidence; a
   copy-prior model is maximally uncertain about change)
3. Energy metric: 1 - best imagined outcome score (refinement = deepening
   and widening rollouts; finding better branches lowers energy)
4. Coherence contribution: Predicted-vs-actual residual on the NEXT cycle
   (surprise). Low residual = the organ's picture of the world is coherent
   with what happened; feeds trust and SNARC salience.

Role in the loop: imagination ADVISES, it never vetoes. It rolls each
proposed effect through the world-model, scores the imagined outcome, and
annotates/ranks the candidates BEFORE the conscience gate (PolicyGate).
Filtering remains PolicyGate's job; dispatch remains the effectors' job.

The substrate is swappable by design. A copy-prior world-model — one that
predicts "no change" — makes imagination unable to picture what an action
does, so anything downstream that scores imagined outcomes reads as
failing when the deficit is actually upstream. This plugin therefore:
  - ships CopyPriorWorldModel as the DEFAULT and names it what it is
    (an honest baseline, not a capability), and
  - accepts any better substrate (a learned dynamics model, an LLM asked
    to picture the outcome) through the same WorldModelAdapter socket,
    via config, without the loop knowing the difference.

Fractal self-similarity (same shape as PolicyGate): the orchestrator sees
just another IRP plugin; internally the plugin runs an imagine → score →
prune cycle; and the world-model it calls may itself be an iterative
refiner. Same pattern at three scales.
"""

import time
import importlib
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from ..base import IRPPlugin, IRPState
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from base import IRPPlugin, IRPState


# ============================================================================
# World-model substrate (the swappable socket)
# ============================================================================

@dataclass
class Prediction:
    """One imagined transition: what the world looks like after an action."""
    next_state: Any                 # Predicted observation/state (plugin-agnostic)
    confidence: float = 0.5         # [0,1] — how much the model trusts this picture
    meta: Dict[str, Any] = field(default_factory=dict)


class WorldModelAdapter:
    """
    Contract for imagination substrates.

    Any model that can answer "if I do THIS in THIS state, what happens?"
    can back the imagination organ: a copy-prior stub, a learned dynamics
    model, an LLM asked to picture the outcome. The loop never sees which.
    """

    name = "world_model"

    def predict(self, state: Any, action: Any, ctx: Dict[str, Any]) -> Prediction:
        raise NotImplementedError


class CopyPriorWorldModel(WorldModelAdapter):
    """
    Predicts that nothing changes.

    This is the honest floor, not a capability: it makes every action look
    the same to the outcome scorer, which is exactly the deficit that makes
    goal inference fail for upstream reasons. It exists so the organ can be
    wired and exercised end-to-end before a real dynamics substrate lands,
    and so A/B swaps have a named baseline.
    """

    name = "copy_prior"

    def __init__(self, confidence: float = 0.3):
        # Low confidence by design: "I cannot picture change."
        self.confidence = confidence

    def predict(self, state: Any, action: Any, ctx: Dict[str, Any]) -> Prediction:
        return Prediction(next_state=state, confidence=self.confidence,
                          meta={'substrate': self.name})


class CallableWorldModel(WorldModelAdapter):
    """
    Wraps any callable(state, action, ctx) -> Prediction | (next_state, conf).

    This is where a learned dynamics checkpoint plugs in without
    subclassing.
    """

    def __init__(self, fn: Callable, name: str = "callable"):
        self.fn = fn
        self.name = name

    def predict(self, state: Any, action: Any, ctx: Dict[str, Any]) -> Prediction:
        out = self.fn(state, action, ctx)
        if isinstance(out, Prediction):
            return out
        if isinstance(out, tuple) and len(out) == 2:
            return Prediction(next_state=out[0], confidence=float(out[1]),
                              meta={'substrate': self.name})
        return Prediction(next_state=out, confidence=0.5,
                          meta={'substrate': self.name})


def load_world_model(spec: Any) -> WorldModelAdapter:
    """
    Resolve a config value into a WorldModelAdapter.

    Accepts: an adapter instance, a callable, or an import path string
    "package.module:factory" whose factory returns an adapter or callable.
    Falls back to CopyPriorWorldModel on any failure — the organ must wire
    even when the good substrate is absent, and the fallback is honest
    about what it is.
    """
    if isinstance(spec, WorldModelAdapter):
        return spec
    if callable(spec):
        return CallableWorldModel(spec)
    if isinstance(spec, str) and ':' in spec:
        try:
            module_path, factory_name = spec.split(':', 1)
            factory = getattr(importlib.import_module(module_path), factory_name)
            built = factory()
            if isinstance(built, WorldModelAdapter):
                return built
            if callable(built):
                return CallableWorldModel(built, name=spec)
        except Exception as e:
            print(f"[Imagination] world_model '{spec}' failed to load ({e}); "
                  f"falling back to copy_prior baseline")
    return CopyPriorWorldModel()


# ============================================================================
# Default outcome scorer (deliberately generic)
# ============================================================================

def default_outcome_scorer(candidate: Any, prediction: Prediction,
                           ctx: Dict[str, Any]) -> float:
    """
    Score an imagined outcome in [0,1] with NO domain semantics.

    Default = the model's own confidence in the pictured change. Under
    copy-prior this is flat across candidates (correctly conveying "I
    can't tell these apart"); a real dynamics substrate differentiates.
    Domain scorers (goal progress, value heads) are injected via config —
    they belong to the instance's context, not to this organ.
    """
    return max(0.0, min(1.0, prediction.confidence))


# ============================================================================
# Imagination IRP
# ============================================================================

class ImaginationIRP(IRPPlugin):
    """
    Rolls candidate actions/effects through a world-model and ranks them
    by imagined outcome. Iterative refinement = imagining more candidates
    (width) and further steps (depth) until energy converges or budget ends.
    """

    def __init__(self, config: Dict[str, Any]):
        config.setdefault('entity_id', 'imagination')
        config.setdefault('max_iterations', 8)
        super().__init__(config)
        self.world_model = load_world_model(config.get('world_model'))
        self.outcome_scorer: Callable = config.get(
            'outcome_scorer', default_outcome_scorer)
        self.rollout_depth = int(config.get('rollout_depth', 1))
        self.batch_size = int(config.get('imagine_batch', 4))
        # Residual bookkeeping: last cycle's best imagined next-state
        self._last_prediction: Optional[Prediction] = None
        self._residual_history: List[float] = []

    # ----- Core IRP contract -----

    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
        """
        x0: list of candidates. Each candidate is any object; if it has a
        `.payload`/`.parameters` attr (Effect-like) that is passed to the
        world-model as the action, else the candidate itself is the action.
        task_ctx: must carry 'observation' (current state); everything else
        is passed through to the substrate and scorer.
        """
        candidates = list(x0) if x0 else []
        return IRPState(
            x={
                'candidates': candidates,
                'pending': list(range(len(candidates))),
                'rollouts': {},        # idx -> {'prediction': Prediction, 'score': float}
                'observation': task_ctx.get('observation'),
                'ctx': task_ctx,
            },
            step_idx=0,
            energy_val=1.0,
        )

    def _action_of(self, candidate: Any) -> Any:
        for attr in ('payload', 'parameters', 'action'):
            if hasattr(candidate, attr):
                return getattr(candidate, attr)
        return candidate

    def step(self, state: IRPState) -> IRPState:
        """Imagine the next batch of pending candidates (one rollout each)."""
        s = state.x
        batch, s['pending'] = s['pending'][:self.batch_size], s['pending'][self.batch_size:]
        for idx in batch:
            candidate = s['candidates'][idx]
            obs = s['observation']
            pred = None
            try:
                for _ in range(self.rollout_depth):
                    pred = self.world_model.predict(
                        pred.next_state if pred else obs,
                        self._action_of(candidate), s['ctx'])
            except Exception as e:
                pred = Prediction(next_state=obs, confidence=0.0,
                                  meta={'error': str(e)[:80]})
            try:
                score = float(self.outcome_scorer(candidate, pred, s['ctx']))
            except Exception:
                score = 0.0
            s['rollouts'][idx] = {'prediction': pred, 'score': score}
        state.step_idx += 1
        state.energy_val = self.energy(state)
        return state

    def energy(self, state: IRPState) -> float:
        """1 - best imagined score so far; monotone non-increasing."""
        scores = [r['score'] for r in state.x['rollouts'].values()]
        return 1.0 - max(scores) if scores else 1.0

    def halt(self, history: List[IRPState]) -> bool:
        if not history:
            return False
        if not history[-1].x['pending']:
            return True
        return super().halt(history) if hasattr(super(), 'halt') else False

    def refine(self, x0: Any, task_ctx: Dict[str, Any]):
        """Convenience runner: imagine everything, return (state, history)."""
        state = self.init_state(x0, task_ctx)
        history = [state]
        max_iter = self.config.get('max_iterations', 8)
        while state.x['pending'] and state.step_idx < max_iter:
            state = self.step(state)
            history.append(state)
        # Remember the best branch for next-cycle residual scoring
        ranked = self.get_ranked(state)
        if ranked:
            best_idx = ranked[0][0]
            self._last_prediction = state.x['rollouts'][best_idx]['prediction']
        return state, history

    # ----- Consumers -----

    def get_ranked(self, state: IRPState) -> List[tuple]:
        """[(idx, candidate, rollout_dict)] sorted best-first."""
        s = state.x
        done = [(i, s['candidates'][i], s['rollouts'][i]) for i in s['rollouts']]
        return sorted(done, key=lambda t: t[2]['score'], reverse=True)

    def annotate(self, state: IRPState) -> List[Any]:
        """
        Return candidates best-first. Effect-like candidates (with a
        .metadata dict) get an 'imagined' annotation; all candidates are
        preserved — imagination never drops one.
        """
        ranked = self.get_ranked(state)
        seen = set()
        out = []
        for rank, (idx, candidate, roll) in enumerate(ranked):
            meta = getattr(candidate, 'metadata', None)
            if isinstance(meta, dict):
                meta['imagined'] = {
                    'rank': rank,
                    'score': round(roll['score'], 4),
                    'confidence': round(roll['prediction'].confidence, 4),
                    'substrate': self.world_model.name,
                }
            seen.add(idx)
            out.append(candidate)
        # Anything not yet imagined (budget ran out) passes through unranked
        for i, candidate in enumerate(state.x['candidates']):
            if i not in seen:
                out.append(candidate)
        return out

    # ----- Coherence: predicted-vs-actual residual (next cycle) -----

    def score_residual(self, actual: Any) -> Optional[float]:
        """
        Compare last cycle's best imagined next-state against what actually
        arrived. Returns residual in [0,1] (0 = perfect picture) or None if
        no prediction is pending / states aren't comparable. Surprise for
        SNARC; (1 - residual) is the organ's coherence contribution.
        """
        if self._last_prediction is None:
            return None
        predicted = self._last_prediction.next_state
        self._last_prediction = None
        residual = self._compare(predicted, actual)
        if residual is not None:
            self._residual_history.append(residual)
            if len(self._residual_history) > 200:
                self._residual_history = self._residual_history[-200:]
        return residual

    @classmethod
    def _compare(cls, predicted: Any, actual: Any) -> Optional[float]:
        if predicted is None or actual is None:
            return None
        # Dict observations (sensor_name -> data, the embodied shape):
        # compare shared keys, mean the comparable residuals.
        if isinstance(predicted, dict) and isinstance(actual, dict):
            shared = set(predicted) & set(actual)
            residuals = [r for r in (cls._compare(predicted[k], actual[k])
                                     for k in shared) if r is not None]
            return sum(residuals) / len(residuals) if residuals else None
        if np is not None:
            try:
                p, a = np.asarray(predicted), np.asarray(actual)
                if p.shape == a.shape and p.dtype != object:
                    denom = float(max(np.abs(p).max(), np.abs(a).max(), 1e-9))
                    return float(np.clip(np.abs(p - a).mean() / denom, 0.0, 1.0))
            except Exception:
                pass
        try:
            return 0.0 if predicted == actual else 1.0
        except Exception:
            return None

    def coherence(self) -> Optional[float]:
        """Rolling coherence = 1 - mean recent residual."""
        if not self._residual_history:
            return None
        recent = self._residual_history[-20:]
        return 1.0 - (sum(recent) / len(recent))

    def to_snarc_scores(self, state: Optional[IRPState] = None) -> Dict[str, float]:
        """Surprise-dominant SNARC contribution from the residual channel."""
        surprise = self._residual_history[-1] if self._residual_history else 0.0
        novelty = 0.0
        if state is not None and state.x['rollouts']:
            scores = [r['score'] for r in state.x['rollouts'].values()]
            spread = (max(scores) - min(scores)) if len(scores) > 1 else 0.0
            novelty = spread  # differentiated futures = something to learn
        return {
            'surprise': surprise,
            'novelty': novelty,
            'arousal': 0.0,
            'reward': 0.0,
            'conflict': 0.0,
            'total': 0.6 * surprise + 0.4 * novelty,
        }
