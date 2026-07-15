"""
Test ImaginationIRP integration with the SAGE consciousness loop.

Covers the organ's contract in isolation (substrate swap, ranking,
never-vetoes, residual/coherence) plus the loop-facing invariants the
step() seams rely on. Uses minimal mocks, same style as the PolicyGate
integration tests.
"""

import unittest
from dataclasses import dataclass, field
from typing import Any, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from sage.irp.plugins.imagination_irp import (
    ImaginationIRP,
    Prediction,
    CopyPriorWorldModel,
    CallableWorldModel,
    load_world_model,
)


@dataclass
class FakeEffect:
    """Minimal Effect-like candidate: has payload + metadata."""
    effect_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def make_effects(n=3):
    return [FakeEffect(effect_id=f'e{i}', payload={'action': i}) for i in range(n)]


class TestSubstrateSocket(unittest.TestCase):

    def test_01_default_is_copy_prior(self):
        organ = ImaginationIRP({})
        self.assertIsInstance(organ.world_model, CopyPriorWorldModel)

    def test_02_copy_prior_predicts_no_change(self):
        wm = CopyPriorWorldModel()
        state = {'frame': [1, 2, 3]}
        pred = wm.predict(state, action='anything', ctx={})
        self.assertIs(pred.next_state, state)
        self.assertLess(pred.confidence, 0.5)  # honest about not picturing change

    def test_03_callable_substrate_swaps_in(self):
        def dynamics(state, action, ctx):
            return Prediction(next_state={'changed': True}, confidence=0.9)
        organ = ImaginationIRP({'world_model': dynamics})
        self.assertIsInstance(organ.world_model, CallableWorldModel)
        pred = organ.world_model.predict({}, None, {})
        self.assertEqual(pred.confidence, 0.9)

    def test_04_bad_import_path_falls_back_to_baseline(self):
        wm = load_world_model('nonexistent.module:factory')
        self.assertIsInstance(wm, CopyPriorWorldModel)


class TestRankingAndNeverVetoes(unittest.TestCase):

    def test_05_all_candidates_survive(self):
        organ = ImaginationIRP({})
        effects = make_effects(5)
        state, _ = organ.refine(effects, {'observation': {'x': 1}})
        out = organ.annotate(state)
        self.assertEqual({e.effect_id for e in out},
                         {e.effect_id for e in effects})

    def test_06_dynamics_substrate_differentiates_ranking(self):
        # A substrate that prefers higher action ids
        def dynamics(state, action, ctx):
            conf = 0.1 + 0.2 * action.get('action', 0)
            return Prediction(next_state=state, confidence=min(conf, 1.0))
        organ = ImaginationIRP({'world_model': dynamics})
        effects = make_effects(4)
        state, _ = organ.refine(effects, {'observation': {}})
        out = organ.annotate(state)
        self.assertEqual(out[0].effect_id, 'e3')  # best imagined outcome first
        self.assertEqual(out[0].metadata['imagined']['rank'], 0)
        self.assertIn('substrate', out[0].metadata['imagined'])

    def test_07_copy_prior_is_flat_no_false_signal(self):
        organ = ImaginationIRP({})
        effects = make_effects(4)
        state, _ = organ.refine(effects, {'observation': {}})
        scores = {r['score'] for r in state.x['rollouts'].values()}
        self.assertEqual(len(scores), 1)  # cannot tell actions apart — and says so

    def test_08_scorer_exception_scores_zero_not_crash(self):
        def bad_scorer(candidate, prediction, ctx):
            raise RuntimeError('scorer blew up')
        organ = ImaginationIRP({'outcome_scorer': bad_scorer})
        effects = make_effects(2)
        state, _ = organ.refine(effects, {'observation': {}})
        self.assertEqual(len(state.x['rollouts']), 2)
        self.assertTrue(all(r['score'] == 0.0
                            for r in state.x['rollouts'].values()))


class TestEnergyConvergence(unittest.TestCase):

    def test_09_energy_monotone_nonincreasing(self):
        def dynamics(state, action, ctx):
            return Prediction(next_state=state,
                              confidence=0.1 + 0.1 * action.get('action', 0))
        organ = ImaginationIRP({'world_model': dynamics, 'imagine_batch': 4})
        organ.batch_size = 2
        effects = make_effects(6)
        _, history = organ.refine(effects, {'observation': {}})
        energies = [s.energy_val for s in history if s.energy_val is not None]
        for a, b in zip(energies, energies[1:]):
            self.assertLessEqual(b, a + 1e-9)


class TestResidualCoherence(unittest.TestCase):

    def test_10_no_prediction_no_residual(self):
        organ = ImaginationIRP({})
        self.assertIsNone(organ.score_residual({'x': 1}))

    def test_11_perfect_prediction_zero_residual(self):
        organ = ImaginationIRP({})
        obs = np.ones((4, 4))
        organ.refine(make_effects(2), {'observation': obs})
        residual = organ.score_residual(obs)  # world didn't change: copy-prior right
        self.assertIsNotNone(residual)
        self.assertAlmostEqual(residual, 0.0)
        self.assertAlmostEqual(organ.coherence(), 1.0)

    def test_12_change_surprises_copy_prior(self):
        organ = ImaginationIRP({})
        obs = np.zeros((4, 4))
        organ.refine(make_effects(2), {'observation': obs})
        changed = np.ones((4, 4))
        residual = organ.score_residual(changed)
        self.assertIsNotNone(residual)
        self.assertGreater(residual, 0.5)

    def test_13_residual_consumed_once(self):
        organ = ImaginationIRP({})
        obs = np.ones(3)
        organ.refine(make_effects(1), {'observation': obs})
        self.assertIsNotNone(organ.score_residual(obs))
        self.assertIsNone(organ.score_residual(obs))  # prediction spent

    def test_14_snarc_scores_shape(self):
        organ = ImaginationIRP({})
        obs = np.zeros(3)
        organ.refine(make_effects(2), {'observation': obs})
        organ.score_residual(np.ones(3))
        snarc = organ.to_snarc_scores()
        for key in ('surprise', 'novelty', 'total'):
            self.assertIn(key, snarc)
        self.assertGreater(snarc['surprise'], 0.5)


class TestWorldModelErrorIsolation(unittest.TestCase):

    def test_15_substrate_exception_yields_zero_confidence_not_crash(self):
        def broken(state, action, ctx):
            raise RuntimeError('substrate died')
        organ = ImaginationIRP({'world_model': broken})
        effects = make_effects(2)
        state, _ = organ.refine(effects, {'observation': {'x': 1}})
        self.assertEqual(len(state.x['rollouts']), 2)
        for roll in state.x['rollouts'].values():
            self.assertEqual(roll['prediction'].confidence, 0.0)


if __name__ == '__main__':
    unittest.main()
