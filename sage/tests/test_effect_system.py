"""
Unit tests for the SAGE Effect system.

Tests Effect, EffectorRegistry, EffectExtractor, and mock effectors.
Mocks torch if unavailable so tests run on any machine.

Run:  python3 sage/tests/test_effect_system.py
"""

import sys
import os
import time
import unittest
from unittest.mock import MagicMock

# Mock torch if not available (base_effector.py and effector_hub.py import it)
if 'torch' not in sys.modules:
    _mt = MagicMock()
    _mt.device = MagicMock(return_value=MagicMock())
    _mt.Tensor = type('Tensor', (), {})
    _mt.cuda.is_available = MagicMock(return_value=False)
    sys.modules['torch'] = _mt
    sys.modules['yaml'] = MagicMock()  # effector_hub.py imports yaml

# Add sage root to path
_sage_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _sage_root not in sys.path:
    sys.path.insert(0, _sage_root)

from interfaces.effect import Effect, EffectType, EffectStatus
from interfaces.effect_extractor import EffectExtractor
from interfaces.effector_registry import EffectorRegistry
from interfaces.base_effector import EffectorCommand, EffectorResult, EffectorStatus as EfStatus
from interfaces.effectors.mock_effectors import (
    MockFileSystemEffector, MockToolUseEffector,
    MockWebEffector, MockCognitiveEffector,
)


# ============================================================================
# Effect Dataclass Tests
# ============================================================================

class TestEffectCreation(unittest.TestCase):
    """Effect creation and defaults."""

    def test_default_creation(self):
        e = Effect()
        self.assertEqual(e.effect_type, EffectType.TOOL_USE)
        self.assertEqual(e.status, EffectStatus.PROPOSED)
        self.assertEqual(len(e.effect_id), 12)
        self.assertEqual(e.children, [])
        self.assertEqual(e.trust_score, 0.5)
        self.assertEqual(e.atp_cost, 1.0)
        self.assertEqual(e.priority, 0)
        self.assertTrue(e.reversible)

    def test_explicit_creation(self):
        e = Effect(
            effect_type=EffectType.FILE_IO,
            action='write',
            target='/tmp/test.txt',
            source_plugin='language',
            role='developer',
            trust_score=0.8,
            atp_cost=0.5,
            priority=5,
            parameters={'content': 'hello'},
        )
        self.assertEqual(e.effect_type, EffectType.FILE_IO)
        self.assertEqual(e.action, 'write')
        self.assertEqual(e.target, '/tmp/test.txt')
        self.assertEqual(e.trust_score, 0.8)
        self.assertEqual(e.priority, 5)

    def test_unique_ids(self):
        ids = [Effect().effect_id for _ in range(100)]
        self.assertEqual(len(ids), len(set(ids)))

    def test_effect_type_completeness(self):
        values = [et.value for et in EffectType]
        self.assertEqual(len(values), 13)
        self.assertEqual(len(values), len(set(values)))

    def test_effect_status_completeness(self):
        statuses = [es.value for es in EffectStatus]
        self.assertEqual(len(statuses), 9)
        self.assertEqual(len(statuses), len(set(statuses)))


class TestEffectAdapters(unittest.TestCase):
    """Effect adapter methods (to_policy_action, to_effector_command, to_dict)."""

    def setUp(self):
        self.effect = Effect(
            effect_type=EffectType.FILE_IO,
            action='write',
            target='/tmp/test.txt',
            source_plugin='language',
            role='developer',
            trust_score=0.8,
            atp_cost=0.5,
            priority=5,
            timeout=10.0,
            parameters={'content': 'hello', 'encoding': 'utf-8'},
        )

    def test_to_policy_action(self):
        pa = self.effect.to_policy_action()
        self.assertEqual(pa['action_id'], self.effect.effect_id)
        self.assertEqual(pa['action_type'], 'write')
        self.assertEqual(pa['target'], '/tmp/test.txt')
        self.assertEqual(pa['role'], 'developer')
        self.assertEqual(pa['trust_score'], 0.8)
        self.assertIn('content', pa['parameters'])

    def test_to_effector_command(self):
        cmd = self.effect.to_effector_command('fs_0')
        self.assertEqual(cmd.effector_id, 'fs_0')
        self.assertEqual(cmd.effector_type, 'file_io')
        self.assertEqual(cmd.action, 'write')
        self.assertEqual(cmd.parameters, self.effect.parameters)
        self.assertEqual(cmd.priority, 5)
        self.assertEqual(cmd.timeout, 10.0)
        self.assertEqual(cmd.metadata['effect_id'], self.effect.effect_id)
        self.assertEqual(cmd.metadata['target'], '/tmp/test.txt')
        self.assertEqual(cmd.metadata['source_plugin'], 'language')

    def test_to_dict(self):
        d = self.effect.to_dict()
        self.assertEqual(d['effect_type'], 'file_io')
        self.assertEqual(d['status'], 'proposed')
        self.assertEqual(d['source_plugin'], 'language')
        self.assertEqual(d['trust_score'], 0.8)
        self.assertNotIn('children', d)

    def test_to_dict_with_children(self):
        parent = Effect(
            effect_type=EffectType.COMPOSITE,
            action='deploy',
            children=[
                Effect(effect_type=EffectType.FILE_IO, action='write'),
                Effect(effect_type=EffectType.MESSAGE, action='send'),
            ],
        )
        d = parent.to_dict()
        self.assertIn('children', d)
        self.assertEqual(len(d['children']), 2)
        self.assertEqual(d['children'][0]['effect_type'], 'file_io')
        self.assertEqual(d['children'][1]['effect_type'], 'message')


class TestEffectLifecycle(unittest.TestCase):
    """Effect lifecycle state transitions."""

    def test_approve(self):
        e = Effect()
        e.approve(rule_id='R1', reason='trusted')
        self.assertEqual(e.status, EffectStatus.APPROVED)
        self.assertEqual(e.policy_decision, 'allow')
        self.assertEqual(e.policy_rule_id, 'R1')
        self.assertEqual(e.policy_reason, 'trusted')

    def test_warn(self):
        e = Effect()
        e.warn(reason='rate limit')
        self.assertEqual(e.status, EffectStatus.WARNED)
        self.assertEqual(e.policy_decision, 'warn')

    def test_deny(self):
        e = Effect()
        e.deny(reason='untrusted')
        self.assertEqual(e.status, EffectStatus.DENIED)
        self.assertEqual(e.policy_decision, 'deny')

    def test_complete(self):
        e = Effect()
        e.complete(result={'bytes_written': 42})
        self.assertEqual(e.status, EffectStatus.COMPLETED)
        self.assertIsNotNone(e.completed_at)
        self.assertEqual(e.execution_result['bytes_written'], 42)

    def test_fail(self):
        e = Effect()
        e.fail(error='timeout')
        self.assertEqual(e.status, EffectStatus.FAILED)
        self.assertIsNotNone(e.completed_at)
        self.assertEqual(e.execution_error, 'timeout')

    def test_complete_without_result(self):
        e = Effect()
        e.complete()
        self.assertEqual(e.status, EffectStatus.COMPLETED)
        self.assertIsNone(e.execution_result)


class TestCompositeEffect(unittest.TestCase):
    """COMPOSITE effect with children."""

    def test_composite_children(self):
        parent = Effect(
            effect_type=EffectType.COMPOSITE,
            action='deploy',
            target='staging',
            children=[
                Effect(effect_type=EffectType.FILE_IO, action='write', target='app.py'),
                Effect(effect_type=EffectType.API_CALL, action='post', target='https://api.example.com'),
                Effect(effect_type=EffectType.MESSAGE, action='send', target='#deployments'),
            ],
        )
        self.assertEqual(len(parent.children), 3)
        self.assertEqual(parent.children[0].effect_type, EffectType.FILE_IO)
        self.assertEqual(parent.children[1].effect_type, EffectType.API_CALL)
        self.assertEqual(parent.children[2].effect_type, EffectType.MESSAGE)

    def test_composite_dict_serialization(self):
        parent = Effect(
            effect_type=EffectType.COMPOSITE,
            action='deploy',
            children=[
                Effect(effect_type=EffectType.FILE_IO, action='write'),
            ],
        )
        d = parent.to_dict()
        self.assertIn('children', d)
        self.assertEqual(len(d['children']), 1)

    def test_nested_composite(self):
        """COMPOSITE containing another COMPOSITE."""
        inner = Effect(
            effect_type=EffectType.COMPOSITE,
            action='build',
            children=[
                Effect(effect_type=EffectType.FILE_IO, action='write'),
            ],
        )
        outer = Effect(
            effect_type=EffectType.COMPOSITE,
            action='deploy',
            children=[inner, Effect(effect_type=EffectType.MESSAGE, action='send')],
        )
        self.assertEqual(len(outer.children), 2)
        self.assertEqual(len(outer.children[0].children), 1)


# ============================================================================
# EffectorRegistry Tests
# ============================================================================

class TestEffectorRegistry(unittest.TestCase):
    """EffectorRegistry type routing, metabolic gating, ATP budgeting."""

    def setUp(self):
        self.registry = EffectorRegistry()
        self.mock_fs = MockFileSystemEffector({'effector_id': 'fs_0', 'effector_type': 'file_io'})
        self.mock_tool = MockToolUseEffector({'effector_id': 'tool_0', 'effector_type': 'tool_use'})
        self.mock_web = MockWebEffector({'effector_id': 'web_0', 'effector_type': 'web'})
        self.mock_cog = MockCognitiveEffector({'effector_id': 'cog_0', 'effector_type': 'cognitive'})

        self.registry.register_effector(self.mock_fs, handles=[EffectType.FILE_IO])
        self.registry.register_effector(self.mock_tool, handles=[EffectType.TOOL_USE])
        self.registry.register_effector(self.mock_web, handles=[EffectType.WEB, EffectType.API_CALL])
        self.registry.register_effector(
            self.mock_cog,
            handles=[EffectType.MEMORY_WRITE, EffectType.TRUST_UPDATE, EffectType.STATE_CHANGE],
        )

    def test_type_routes_registered(self):
        routes = self.registry.get_routes()
        self.assertIn('file_io', routes)
        self.assertIn('tool_use', routes)
        self.assertIn('web', routes)
        self.assertIn('api_call', routes)
        self.assertIn('memory_write', routes)
        self.assertEqual(routes['file_io'], ['fs_0'])
        self.assertEqual(routes['web'], ['web_0'])

    def test_dispatch_file_io(self):
        effect = Effect(effect_type=EffectType.FILE_IO, action='write',
                        target='/tmp/test.txt', parameters={'content': 'hello'})
        result = self.registry.dispatch_effect(effect, metabolic_state='wake')
        self.assertTrue(result.is_success())
        self.assertEqual(effect.status, EffectStatus.COMPLETED)
        self.assertEqual(len(self.mock_fs.operation_log), 1)
        self.assertEqual(self.mock_fs.operation_log[0]['action'], 'write')

    def test_dispatch_tool_use(self):
        effect = Effect(effect_type=EffectType.TOOL_USE, action='invoke', target='calc')
        result = self.registry.dispatch_effect(effect, metabolic_state='wake')
        self.assertTrue(result.is_success())
        self.assertEqual(len(self.mock_tool.operation_log), 1)

    def test_dispatch_web(self):
        effect = Effect(effect_type=EffectType.WEB, action='get', target='https://example.com')
        result = self.registry.dispatch_effect(effect, metabolic_state='wake')
        self.assertTrue(result.is_success())
        self.assertEqual(len(self.mock_web.operation_log), 1)

    def test_dispatch_cognitive(self):
        effect = Effect(effect_type=EffectType.MEMORY_WRITE, action='consolidate',
                        target='experience_buffer')
        result = self.registry.dispatch_effect(effect, metabolic_state='wake')
        self.assertTrue(result.is_success())
        self.assertEqual(len(self.mock_cog.operation_log), 1)

    def test_no_effector_for_type(self):
        effect = Effect(effect_type=EffectType.DATABASE, action='query', target='users')
        result = self.registry.dispatch_effect(effect, metabolic_state='wake')
        self.assertFalse(result.is_success())
        self.assertEqual(effect.status, EffectStatus.FAILED)

    # --- Metabolic Gating ---

    def test_dream_blocks_physical(self):
        effect = Effect(effect_type=EffectType.FILE_IO, action='write', target='/tmp/x')
        result = self.registry.dispatch_effect(effect, metabolic_state='dream')
        self.assertFalse(result.is_success())
        self.assertEqual(effect.status, EffectStatus.DENIED)

    def test_dream_allows_cognitive(self):
        effect = Effect(effect_type=EffectType.MEMORY_WRITE, action='consolidate',
                        target='buffer')
        result = self.registry.dispatch_effect(effect, metabolic_state='dream')
        self.assertTrue(result.is_success())

    def test_rest_blocks_file_io(self):
        effect = Effect(effect_type=EffectType.FILE_IO, action='write', target='/tmp/x')
        result = self.registry.dispatch_effect(effect, metabolic_state='rest')
        self.assertFalse(result.is_success())
        self.assertEqual(effect.status, EffectStatus.DENIED)

    def test_rest_allows_message(self):
        # Register cognitive mock to also handle MESSAGE
        self.registry.register_effector(
            self.mock_cog, handles=[EffectType.MESSAGE], effector_id='msg_0'
        )
        effect = Effect(effect_type=EffectType.MESSAGE, action='send', target='#channel')
        result = self.registry.dispatch_effect(effect, metabolic_state='rest')
        self.assertTrue(result.is_success())

    def test_crisis_allows_motor(self):
        self.registry.register_effector(
            self.mock_cog, handles=[EffectType.MOTOR], effector_id='motor_0'
        )
        effect = Effect(effect_type=EffectType.MOTOR, action='move', target='arm')
        result = self.registry.dispatch_effect(effect, metabolic_state='crisis')
        self.assertTrue(result.is_success())

    def test_crisis_blocks_database(self):
        effect = Effect(effect_type=EffectType.DATABASE, action='query', target='users')
        result = self.registry.dispatch_effect(effect, metabolic_state='crisis')
        self.assertFalse(result.is_success())

    def test_metabolic_gates_serialization(self):
        gates = self.registry.get_metabolic_gates()
        self.assertIn('wake', gates)
        self.assertIn('dream', gates)
        self.assertEqual(len(gates['wake']), 13)  # All types in wake
        self.assertEqual(len(gates['dream']), 3)   # Only cognitive in dream

    def test_custom_metabolic_gate(self):
        self.registry.set_metabolic_gate('custom', {EffectType.FILE_IO})
        effect = Effect(effect_type=EffectType.FILE_IO, action='write', target='/tmp/x')
        result = self.registry.dispatch_effect(effect, metabolic_state='custom')
        self.assertTrue(result.is_success())

    # --- ATP Budgeting ---

    def test_atp_budget_limits_dispatch(self):
        effects = [
            Effect(effect_type=EffectType.FILE_IO, action='write', target=f'/tmp/{i}.txt',
                   atp_cost=3.0, priority=10 - i)
            for i in range(5)
        ]
        results = self.registry.dispatch_effects(effects, metabolic_state='wake', atp_budget=7.0)
        self.assertEqual(len(results), 5)
        succeeded = sum(1 for r in results if r.is_success())
        self.assertEqual(succeeded, 2)

    def test_atp_budget_respects_priority(self):
        effects = [
            Effect(effect_type=EffectType.FILE_IO, action='write', target='low.txt',
                   atp_cost=5.0, priority=1),
            Effect(effect_type=EffectType.FILE_IO, action='write', target='high.txt',
                   atp_cost=5.0, priority=10),
        ]
        results = self.registry.dispatch_effects(effects, metabolic_state='wake', atp_budget=6.0)
        high = [e for e in effects if e.target == 'high.txt'][0]
        low = [e for e in effects if e.target == 'low.txt'][0]
        self.assertEqual(high.status, EffectStatus.COMPLETED)
        self.assertEqual(low.status, EffectStatus.FAILED)

    def test_zero_atp_budget(self):
        effects = [
            Effect(effect_type=EffectType.FILE_IO, action='write', target='/tmp/x',
                   atp_cost=0.1),
        ]
        results = self.registry.dispatch_effects(effects, metabolic_state='wake', atp_budget=0.0)
        self.assertFalse(results[0].is_success())

    # --- Disabled Effector ---

    def test_disabled_effector_not_selected(self):
        self.mock_fs.disable()
        effect = Effect(effect_type=EffectType.FILE_IO, action='write', target='/tmp/x')
        result = self.registry.dispatch_effect(effect, metabolic_state='wake')
        self.assertFalse(result.is_success())


# ============================================================================
# EffectExtractor Tests
# ============================================================================

class TestEffectExtractor(unittest.TestCase):
    """EffectExtractor bridge from PluginResult to List[Effect]."""

    def setUp(self):
        self.extractor = EffectExtractor()

    def _make_result(self, data):
        r = MagicMock()
        r.final_state = MagicMock()
        r.final_state.x = data
        return r

    # --- Language ---

    def test_language_with_response(self):
        result = self._make_result({'response': 'Hello world'})
        effects = self.extractor.extract('language', result)
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0].effect_type, EffectType.AUDIO)
        self.assertEqual(effects[0].action, 'speak')
        self.assertEqual(effects[0].parameters['text'], 'Hello world')
        self.assertEqual(effects[0].source_plugin, 'language')

    def test_language_with_text_key(self):
        result = self._make_result({'text': 'Greetings'})
        effects = self.extractor.extract('language', result)
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0].parameters['text'], 'Greetings')

    def test_language_empty_dict(self):
        effects = self.extractor.extract('language', self._make_result({}))
        self.assertEqual(len(effects), 0)

    def test_language_non_dict(self):
        effects = self.extractor.extract('language', self._make_result('plain_string'))
        self.assertEqual(len(effects), 0)

    # --- Vision ---

    def test_vision_with_data(self):
        effects = self.extractor.extract('vision', self._make_result({'frame': 'tensor'}))
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0].effect_type, EffectType.VISUAL)
        self.assertEqual(effects[0].action, 'display')

    def test_vision_none(self):
        effects = self.extractor.extract('vision', self._make_result(None))
        self.assertEqual(len(effects), 0)

    # --- Control ---

    def test_control_trajectory(self):
        trajectory = [{'x': 0.1, 'y': 0.2}, {'x': 0.3, 'y': 0.4}]
        effects = self.extractor.extract('control', self._make_result({'trajectory': trajectory}))
        self.assertEqual(len(effects), 2)
        for e in effects:
            self.assertEqual(e.effect_type, EffectType.MOTOR)
            self.assertEqual(e.action, 'move')
            self.assertFalse(e.reversible)

    def test_control_priority_ordering(self):
        trajectory = [{'x': i} for i in range(5)]
        effects = self.extractor.extract('control', self._make_result({'trajectory': trajectory}))
        for i in range(len(effects) - 1):
            self.assertGreater(effects[i].priority, effects[i + 1].priority)

    def test_control_empty(self):
        effects = self.extractor.extract('control', self._make_result({'trajectory': []}))
        self.assertEqual(len(effects), 0)

    # --- Memory ---

    def test_memory_with_patterns(self):
        effects = self.extractor.extract(
            'memory', self._make_result({'consolidated': True, 'patterns': ['p1']})
        )
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0].effect_type, EffectType.MEMORY_WRITE)
        self.assertEqual(effects[0].action, 'consolidate')

    def test_memory_without_patterns(self):
        effects = self.extractor.extract('memory', self._make_result({'other': True}))
        self.assertEqual(len(effects), 0)

    # --- TTS ---

    def test_tts_dict_text(self):
        effects = self.extractor.extract('tts', self._make_result({'text': 'Say this'}))
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0].effect_type, EffectType.AUDIO)
        self.assertEqual(effects[0].action, 'speak')

    def test_tts_raw_data(self):
        effects = self.extractor.extract('tts', self._make_result('raw_audio'))
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0].action, 'play')

    # --- General ---

    def test_unknown_plugin(self):
        effects = self.extractor.extract('unknown', self._make_result({'data': 1}))
        self.assertEqual(len(effects), 0)

    def test_none_result(self):
        effects = self.extractor.extract('language', None)
        self.assertEqual(len(effects), 0)

    def test_custom_extractor(self):
        def custom(result, ctx):
            return [Effect(effect_type=EffectType.DATABASE, action='query', target='users')]

        self.extractor.register('custom_db', custom)
        effects = self.extractor.extract('custom_db', self._make_result({}))
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0].effect_type, EffectType.DATABASE)

    def test_trust_from_context(self):
        result = self._make_result({'response': 'hello'})
        effects = self.extractor.extract('language', result, {'trust_weights': {'language': 0.9}})
        self.assertEqual(effects[0].trust_score, 0.9)

    def test_default_trust_without_context(self):
        result = self._make_result({'response': 'hello'})
        effects = self.extractor.extract('language', result)
        self.assertEqual(effects[0].trust_score, 0.5)


# ============================================================================
# Mock Effector Tests
# ============================================================================

class TestMockEffectors(unittest.TestCase):
    """Mock effector implementations."""

    def test_filesystem_execute(self):
        fs = MockFileSystemEffector({'effector_id': 'fs_0', 'effector_type': 'file_io'})
        cmd = EffectorCommand(
            effector_id='fs_0', effector_type='file_io', action='write',
            parameters={'content': 'hello'}, metadata={'target': '/tmp/test.txt'},
        )
        result = fs.execute(cmd)
        self.assertTrue(result.is_success())
        self.assertEqual(len(fs.operation_log), 1)
        self.assertEqual(fs.operation_log[0]['action'], 'write')

    def test_filesystem_invalid_action(self):
        fs = MockFileSystemEffector({'effector_id': 'fs_0'})
        valid, msg = fs.validate_command(EffectorCommand(
            effector_id='fs_0', effector_type='file_io', action='destroy'
        ))
        self.assertFalse(valid)

    def test_tool_use_execute(self):
        tool = MockToolUseEffector({'effector_id': 'tool_0', 'effector_type': 'tool_use'})
        cmd = EffectorCommand(
            effector_id='tool_0', effector_type='tool_use', action='invoke',
            metadata={'target': 'calculator'},
        )
        result = tool.execute(cmd)
        self.assertTrue(result.is_success())
        self.assertEqual(tool.operation_log[0]['tool'], 'calculator')

    def test_web_execute(self):
        web = MockWebEffector({'effector_id': 'web_0', 'effector_type': 'web'})
        cmd = EffectorCommand(
            effector_id='web_0', effector_type='web', action='get',
            metadata={'target': 'https://example.com'},
        )
        result = web.execute(cmd)
        self.assertTrue(result.is_success())
        self.assertEqual(web.operation_log[0]['method'], 'GET')

    def test_cognitive_execute(self):
        cog = MockCognitiveEffector({'effector_id': 'cog_0', 'effector_type': 'cognitive'})
        cmd = EffectorCommand(
            effector_id='cog_0', effector_type='cognitive', action='consolidate',
            metadata={'target': 'experience_buffer'},
        )
        result = cog.execute(cmd)
        self.assertTrue(result.is_success())
        self.assertEqual(cog.operation_log[0]['action'], 'consolidate')

    def test_disabled_effector(self):
        fs = MockFileSystemEffector({'effector_id': 'fs_0'})
        fs.disable()
        cmd = EffectorCommand(
            effector_id='fs_0', effector_type='file_io', action='write',
            metadata={'target': '/tmp/test.txt'},
        )
        result = fs.execute(cmd)
        self.assertFalse(result.is_success())

    def test_effector_stats(self):
        fs = MockFileSystemEffector({'effector_id': 'fs_0', 'effector_type': 'file_io'})
        for i in range(5):
            cmd = EffectorCommand(
                effector_id='fs_0', effector_type='file_io', action='read',
                metadata={'target': f'/tmp/{i}.txt'},
            )
            fs.execute(cmd)
        self.assertEqual(fs.execute_count, 5)
        self.assertEqual(fs.success_count, 5)

    def test_effector_info(self):
        fs = MockFileSystemEffector({'effector_id': 'fs_0', 'effector_type': 'file_io'})
        info = fs.get_info()
        self.assertEqual(info['effector_id'], 'fs_0')
        self.assertTrue(info['mock'])
        self.assertIn('write', info['supported_actions'])


# ============================================================================
# Run
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SAGE Effect System — Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
