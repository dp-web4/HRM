"""
Effector Registry — Effect-type-aware dispatch layer.

Extends EffectorHub with:
1. EffectType -> List[effector_id] mapping (which effectors handle which effects)
2. Capability-based routing (select best available effector for an effect)
3. ATP budgeting of effects before dispatch
4. Metabolic state gating (disable effect categories per state)

Version: 1.0 (2026-02-19)
"""

import time
from typing import Dict, List, Optional, Any, Set

from .base_effector import BaseEffector, EffectorResult, EffectorStatus
from .effector_hub import EffectorHub
from .effect import Effect, EffectType, EffectStatus as EStatus


class EffectorRegistry(EffectorHub):
    """
    Extends EffectorHub with effect-type-aware dispatch.

    EffectorHub: effector_id -> BaseEffector (direct dispatch)
    EffectorRegistry adds: EffectType -> [effector_id, ...] (routing table)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._type_routes: Dict[EffectType, List[str]] = {}
        self._metabolic_gates: Dict[str, Set[EffectType]] = self._default_metabolic_gates()

    def register_effector(self, effector: BaseEffector,
                          handles: Optional[List[EffectType]] = None,
                          effector_id: Optional[str] = None):
        """
        Register effector with effect type routing.

        Args:
            effector: The effector instance
            handles: List of EffectTypes this effector can handle
            effector_id: Optional custom ID (uses effector.effector_id if None)
        """
        super().register_effector(effector, effector_id)
        eid = effector_id or effector.effector_id

        if handles:
            for effect_type in handles:
                if effect_type not in self._type_routes:
                    self._type_routes[effect_type] = []
                if eid not in self._type_routes[effect_type]:
                    self._type_routes[effect_type].append(eid)

    def dispatch_effect(self, effect: Effect,
                        metabolic_state: str = "wake") -> EffectorResult:
        """
        Dispatch a single Effect to the appropriate Effector.

        Pipeline: metabolic gate -> type route -> select available ->
                  adapt via to_effector_command -> execute -> update status
        """
        # 1. Metabolic gate
        allowed = self._metabolic_gates.get(metabolic_state, set())
        if effect.effect_type not in allowed:
            effect.status = EStatus.DENIED
            effect.policy_reason = (
                f"Effect type {effect.effect_type.value} "
                f"not allowed in {metabolic_state}"
            )
            return EffectorResult(
                effector_id="registry",
                status=EffectorStatus.SAFETY_VIOLATION,
                message=f"Metabolic state '{metabolic_state}' blocks "
                        f"{effect.effect_type.value}",
            )

        # 2. Find capable effectors
        candidates = self._type_routes.get(effect.effect_type, [])
        if not candidates:
            effect.fail(f"No effector registered for {effect.effect_type.value}")
            return EffectorResult(
                effector_id="registry",
                status=EffectorStatus.HARDWARE_UNAVAILABLE,
                message=f"No effector for {effect.effect_type.value}",
            )

        # 3. Select first available
        selected_id = None
        for eid in candidates:
            if eid in self.effectors and self.effectors[eid].is_available():
                selected_id = eid
                break

        if not selected_id:
            effect.fail(f"No available effector for {effect.effect_type.value}")
            return EffectorResult(
                effector_id="registry",
                status=EffectorStatus.HARDWARE_UNAVAILABLE,
                message=f"No available effector for {effect.effect_type.value}",
            )

        # 4. Convert and dispatch
        effect.status = EStatus.DISPATCHED
        command = effect.to_effector_command(selected_id)

        # 5. Execute through parent hub
        effect.status = EStatus.EXECUTING
        result = super().execute(command)

        # 6. Update effect from result
        if result.is_success():
            effect.complete(result.metadata)
        else:
            effect.fail(result.message)

        return result

    def dispatch_effects(self, effects: List[Effect],
                         metabolic_state: str = "wake",
                         atp_budget: float = 10.0) -> List[EffectorResult]:
        """
        Dispatch multiple effects with ATP budgeting.

        Effects sorted by priority (descending), dispatched until budget exhausted.
        """
        sorted_effects = sorted(effects, key=lambda e: -e.priority)
        results = []
        remaining = atp_budget

        for effect in sorted_effects:
            if effect.atp_cost > remaining:
                effect.fail(
                    f"Insufficient ATP (need {effect.atp_cost:.1f}, "
                    f"have {remaining:.1f})"
                )
                results.append(EffectorResult(
                    effector_id="registry",
                    status=EffectorStatus.FAILED,
                    message=f"ATP budget exhausted for {effect.effect_id}",
                ))
                continue

            result = self.dispatch_effect(effect, metabolic_state)
            results.append(result)

            if result.is_success():
                remaining -= effect.atp_cost

        return results

    def get_routes(self) -> Dict[str, List[str]]:
        """Get the type routing table (string keys for serialization)."""
        return {k.value: v for k, v in self._type_routes.items()}

    def get_metabolic_gates(self) -> Dict[str, List[str]]:
        """Get the metabolic gating rules (string values for serialization)."""
        return {
            state: [et.value for et in types]
            for state, types in self._metabolic_gates.items()
        }

    def set_metabolic_gate(self, state: str, allowed: Set[EffectType]):
        """Override metabolic gate for a specific state."""
        self._metabolic_gates[state] = allowed

    @staticmethod
    def _default_metabolic_gates() -> Dict[str, Set[EffectType]]:
        """Default metabolic gating rules."""
        all_types = set(EffectType)
        cognitive = {
            EffectType.MEMORY_WRITE,
            EffectType.TRUST_UPDATE,
            EffectType.STATE_CHANGE,
        }
        return {
            'wake': all_types,
            'focus': all_types,
            'rest': cognitive | {EffectType.MESSAGE},
            'dream': cognitive,  # Only internal effects during consolidation
            'crisis': {
                EffectType.MOTOR, EffectType.MESSAGE,
                EffectType.STATE_CHANGE, EffectType.MEMORY_WRITE,
            },
        }
