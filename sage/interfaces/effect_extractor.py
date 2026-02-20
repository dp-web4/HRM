"""
Effect Extractor — Bridge from PluginResult to List[Effect].

Each IRP plugin produces results in its own format (tensor for vision,
dict for PolicyGate, trajectory for control). The extractor derives
proposed Effects from each plugin type's output.

Translation layer between the plugin world (IRP states) and the
effector world (Effects).

Version: 1.0 (2026-02-19)
"""

import uuid
from typing import Dict, Any, List, Callable, Optional

from .effect import Effect, EffectType


class EffectExtractor:
    """
    Extracts proposed Effects from plugin results.

    Maintains a registry of plugin-specific extraction functions.
    New plugins register their own extractors via register().
    """

    def __init__(self):
        self._extractors: Dict[str, Callable] = {}
        self._register_defaults()

    def register(self, plugin_name: str, extractor_fn: Callable):
        """Register an extraction function for a plugin type."""
        self._extractors[plugin_name] = extractor_fn

    def extract(self, plugin_name: str, plugin_result: Any,
                context: Optional[Dict[str, Any]] = None) -> List[Effect]:
        """
        Extract Effects from a plugin result.

        Args:
            plugin_name: Name of the source plugin
            plugin_result: PluginResult from orchestrator
            context: Current context (metabolic state, trust weights, etc.)

        Returns:
            List of proposed Effects (status=PROPOSED)
        """
        ctx = context or {}
        if plugin_name in self._extractors:
            try:
                return self._extractors[plugin_name](plugin_result, ctx)
            except Exception:
                return []
        return []

    def _register_defaults(self):
        """Register default extractors for known plugin types."""
        self._extractors['language'] = _extract_language
        self._extractors['vision'] = _extract_vision
        self._extractors['control'] = _extract_control
        self._extractors['memory'] = _extract_memory
        self._extractors['tts'] = _extract_tts


# ============================================================================
# Default extraction functions
# ============================================================================

def _get_state_data(result) -> Optional[Any]:
    """Safely get final_state.x from a PluginResult."""
    if result is None:
        return None
    fs = getattr(result, 'final_state', None)
    if fs is None:
        return None
    return getattr(fs, 'x', None)


def _get_trust(ctx: dict, plugin: str) -> float:
    """Get trust score for a plugin from context."""
    return ctx.get('trust_weights', {}).get(plugin, 0.5)


def _extract_language(result, ctx: dict) -> List[Effect]:
    """Language plugin may propose speech or message effects."""
    data = _get_state_data(result)
    if not isinstance(data, dict):
        return []

    effects = []
    response = data.get('response', data.get('text', ''))
    if not response:
        return effects

    # If this is a gateway message response, create a MESSAGE effect
    message_id = data.get('message_id')
    if message_id:
        effects.append(Effect(
            effect_type=EffectType.MESSAGE,
            action='respond',
            target=data.get('sender', 'unknown'),
            parameters={
                'message_id': message_id,
                'response': str(response),
                'conversation_id': data.get('conversation_id', ''),
                'action': 'respond',
            },
            data={'response': str(response)},
            source_plugin='language',
            trust_score=_get_trust(ctx, 'language'),
            atp_cost=0.5,
            priority=10,  # High priority — someone is waiting
        ))
    else:
        # Non-message language output → speech effect
        effects.append(Effect(
            effect_type=EffectType.AUDIO,
            action='speak',
            target='default_speaker',
            parameters={'text': str(response)},
            source_plugin='language',
            trust_score=_get_trust(ctx, 'language'),
            atp_cost=0.5,
            priority=5,
        ))
    return effects


def _extract_vision(result, ctx: dict) -> List[Effect]:
    """Vision plugin may propose display effects."""
    data = _get_state_data(result)
    if data is None:
        return []

    return [Effect(
        effect_type=EffectType.VISUAL,
        action='display',
        target='default_display',
        data=data,
        source_plugin='vision',
        trust_score=_get_trust(ctx, 'vision'),
        atp_cost=0.3,
        priority=3,
    )]


def _extract_control(result, ctx: dict) -> List[Effect]:
    """Control plugin proposes motor effects from trajectory."""
    data = _get_state_data(result)
    if not isinstance(data, dict):
        return []

    trajectory = data.get('trajectory', data.get('actions', []))
    if not trajectory:
        return []

    effects = []
    for i, waypoint in enumerate(trajectory):
        params = waypoint if isinstance(waypoint, dict) else {'position': waypoint}
        effects.append(Effect(
            effect_type=EffectType.MOTOR,
            action='move',
            target='robot_arm',
            parameters=params,
            source_plugin='control',
            trust_score=_get_trust(ctx, 'control'),
            atp_cost=0.3,
            priority=10 - i,  # Earlier waypoints higher priority
            reversible=False,
        ))
    return effects


def _extract_memory(result, ctx: dict) -> List[Effect]:
    """Memory plugin proposes memory write effects."""
    data = _get_state_data(result)
    if not isinstance(data, dict):
        return []

    effects = []
    if 'consolidated' in data or 'patterns' in data:
        effects.append(Effect(
            effect_type=EffectType.MEMORY_WRITE,
            action='consolidate',
            target='experience_buffer',
            parameters=data,
            source_plugin='memory',
            trust_score=_get_trust(ctx, 'memory'),
            atp_cost=0.2,
            priority=2,
        ))
    return effects


def _extract_tts(result, ctx: dict) -> List[Effect]:
    """TTS plugin proposes audio effects."""
    data = _get_state_data(result)

    effects = []
    if data is not None:
        # Could be tensor (raw audio) or dict with text
        if isinstance(data, dict):
            text = data.get('text', '')
            if text:
                effects.append(Effect(
                    effect_type=EffectType.AUDIO,
                    action='speak',
                    target='default_speaker',
                    parameters={'text': str(text)},
                    source_plugin='tts',
                    trust_score=_get_trust(ctx, 'tts'),
                    atp_cost=0.5,
                    priority=5,
                ))
        else:
            # Assume tensor/audio data
            effects.append(Effect(
                effect_type=EffectType.AUDIO,
                action='play',
                target='default_speaker',
                data=data,
                source_plugin='tts',
                trust_score=_get_trust(ctx, 'tts'),
                atp_cost=0.5,
                priority=5,
            ))
    return effects
