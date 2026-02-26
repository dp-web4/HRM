"""
SAGE - Situation-Aware Governance Engine

Unified entry point for creating and running SAGE consciousness systems.

Usage:
    from sage import SAGE

    # Create with defaults (mock sensors, algorithmic SNARC)
    sage = SAGE.create()

    # Create with real LLM and PolicyGate
    sage = SAGE.create(use_real_llm=True, use_policy_gate=True)

    # Run the consciousness loop
    await sage.run(max_cycles=100)

This facade auto-wires the best available components from the SAGE codebase:
- Metabolic controller (5 states: WAKE/FOCUS/REST/DREAM/CRISIS)
- IRP plugin system (33 plugins with ATP budgeting)
- SNARC salience scoring (5D: Surprise/Novelty/Arousal/Reward/Conflict)
- PolicyGate accountability (optional)
- Effector system (FileSystem, Web, Tool, Network)
- Sleep consolidation pipeline (experience → LoRA training)
"""

from typing import Optional, Dict, Any
import asyncio


class SAGE:
    """
    Unified SAGE facade - single entry point for consciousness system.

    Automatically wires:
    - SAGEConsciousness (full 9-step loop)
    - MetabolicController (5-state management)
    - IRP plugins (33 plugins with ATP budget)
    - SNARC salience (algorithmic 5D scoring)
    - PolicyGate (optional accountability checkpoint)
    - EffectorRegistry (real effect dispatch)
    """

    def __init__(self, consciousness_loop, config: Optional[Dict[str, Any]] = None):
        """
        Internal constructor. Use SAGE.create() instead.

        Args:
            consciousness_loop: Underlying consciousness implementation
            config: Configuration dictionary
        """
        self._loop = consciousness_loop
        self._config = config or {}

    @staticmethod
    def create(
        config: Optional[Dict[str, Any]] = None,
        use_real_llm: bool = False,
        use_real_sensors: bool = False,
        use_policy_gate: bool = False,
        use_neural_snarc: bool = False
    ) -> 'SAGE':
        """
        Create a SAGE consciousness system with automatic component wiring.

        Args:
            config: Configuration dictionary (optional)
                - 'lct_identity': Hardware-bound identity config
                - 'metabolic_params': ATP rates, state thresholds
                - 'plugin_config': IRP plugin settings
                - 'snarc_weights': 5D salience weights
                - 'llm_config': LLM backend settings

            use_real_llm: Use real LLM (Ollama/Transformers) vs mock
            use_real_sensors: Use real sensors vs mock observations
            use_policy_gate: Enable PolicyGate at step 8.5
            use_neural_snarc: Use learned SNARC model vs algorithmic

        Returns:
            SAGE instance ready to run

        Example:
            # Minimal - mock everything
            sage = SAGE.create()

            # Production-like - real LLM + PolicyGate
            sage = SAGE.create(use_real_llm=True, use_policy_gate=True)

            # Full config
            sage = SAGE.create(config={
                'lct_identity': {'hardware_id': 'thor-001'},
                'metabolic_params': {'base_atp': 1000.0}
            }, use_real_llm=True, use_policy_gate=True)
        """
        config = config or {}

        # Determine which consciousness implementation to use
        if use_real_llm:
            from sage.core.sage_consciousness_real import RealSAGEConsciousness

            # RealSAGEConsciousness uses real LLM + real SNARC
            loop = RealSAGEConsciousness(config)

        else:
            from sage.core.sage_consciousness import SAGEConsciousness

            # SAGEConsciousness is the full 9-step loop with metabolic model
            # It supports PolicyGate, EffectorRegistry, and all IRP plugins
            loop_config = {
                **config,
                'use_policy_gate': use_policy_gate,
                'use_neural_snarc': use_neural_snarc,
                'use_real_sensors': use_real_sensors
            }

            loop = SAGEConsciousness(loop_config)

        return SAGE(loop, config)

    async def run(
        self,
        max_cycles: Optional[int] = None,
        max_duration_seconds: Optional[float] = None,
        stop_on_crisis: bool = False
    ) -> Dict[str, Any]:
        """
        Run the SAGE consciousness loop.

        Args:
            max_cycles: Maximum number of consciousness cycles (None = infinite)
            max_duration_seconds: Maximum runtime in seconds (None = no limit)
            stop_on_crisis: Stop if CRISIS metabolic state is entered

        Returns:
            Statistics about the run:
                - 'cycles_completed': Number of cycles executed
                - 'final_state': Final metabolic state
                - 'total_experiences': Experiences captured
                - 'atp_remaining': Final ATP budget
        """
        import time

        start_time = time.time()
        cycles = 0

        print(f"[SAGE] Starting consciousness loop...")
        print(f"  Max cycles: {max_cycles if max_cycles else 'unlimited'}")
        print(f"  Max duration: {max_duration_seconds if max_duration_seconds else 'unlimited'}s")
        print(f"  Stop on CRISIS: {stop_on_crisis}")

        try:
            while True:
                # Check stop conditions
                if max_cycles and cycles >= max_cycles:
                    print(f"[SAGE] Reached max cycles ({max_cycles})")
                    break

                if max_duration_seconds:
                    elapsed = time.time() - start_time
                    if elapsed >= max_duration_seconds:
                        print(f"[SAGE] Reached max duration ({max_duration_seconds}s)")
                        break

                # Execute one consciousness cycle
                # SAGEConsciousness uses step(), not tick()
                await self._loop.step()
                cycles += 1

                # Check for CRISIS state if requested
                if stop_on_crisis and hasattr(self._loop, 'state'):
                    from sage.core.metabolic_states import MetabolicState
                    if self._loop.state == MetabolicState.CRISIS:
                        print(f"[SAGE] Entered CRISIS state - stopping")
                        break

                # Brief pause between cycles
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print(f"\n[SAGE] Interrupted by user")

        # Gather statistics
        stats = {
            'cycles_completed': cycles,
            'duration_seconds': time.time() - start_time,
            'final_state': str(self._loop.state) if hasattr(self._loop, 'state') else 'unknown'
        }

        # Add component-specific stats if available
        if hasattr(self._loop, 'experience_buffer'):
            stats['total_experiences'] = len(self._loop.experience_buffer.experiences)

        if hasattr(self._loop, 'metabolic_controller'):
            stats['atp_remaining'] = self._loop.metabolic_controller.atp_budget

        print(f"\n[SAGE] Run complete:")
        print(f"  Cycles: {stats['cycles_completed']}")
        print(f"  Duration: {stats['duration_seconds']:.1f}s")
        print(f"  Final state: {stats['final_state']}")

        return stats

    @property
    def state(self):
        """Get current metabolic state"""
        if hasattr(self._loop, 'state'):
            return self._loop.state
        return None

    @property
    def experiences(self):
        """Get captured experiences"""
        if hasattr(self._loop, 'experience_buffer'):
            return self._loop.experience_buffer.experiences
        return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the SAGE system.

        Returns:
            Dictionary with:
                - metabolic_state: Current state
                - atp_budget: Remaining ATP
                - experience_count: Experiences captured
                - plugin_stats: IRP plugin invocation counts
                - snarc_stats: Salience scoring statistics
        """
        stats = {}

        if hasattr(self._loop, 'state'):
            stats['metabolic_state'] = str(self._loop.state)

        if hasattr(self._loop, 'metabolic_controller'):
            stats['atp_budget'] = self._loop.metabolic_controller.atp_budget

        if hasattr(self._loop, 'experience_buffer'):
            stats['experience_count'] = len(self._loop.experience_buffer.experiences)

        if hasattr(self._loop, 'plugin_invocations'):
            stats['plugin_stats'] = self._loop.plugin_invocations

        if hasattr(self._loop, 'snarc_scorer'):
            stats['snarc_stats'] = self._loop.snarc_scorer.get_statistics()

        return stats


# Convenience exports for direct access to components
__all__ = ['SAGE']

# Version
__version__ = '0.1.0-alpha'
