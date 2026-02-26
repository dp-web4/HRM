#!/usr/bin/env python3
"""
Plugin Router for Attention Kernel

Lightweight wrapper around existing HRM IRP orchestrator that:
- Discovers and registers available IRP plugins
- Routes ATP budgets to selected plugins
- Aggregates results for kernel consumption
- Provides simplified interface for Tier 0 kernel

This integrates existing IRP plugins with the new continuous attention kernel.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
import time

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.irp.base import IRPPlugin, IRPState
from sage.irp.orchestrator import HRMOrchestrator, PluginResult


class PluginRouter:
    """
    Routes plugin requests from Attention Kernel to IRP plugins

    Wraps HRMOrchestrator with a simpler interface suitable for
    the continuous attention loop.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize plugin router

        Args:
            config: Router configuration including:
                - enable_vision: Enable vision plugins (default True)
                - enable_language: Enable language plugins (default True)
                - enable_control: Enable control plugins (default False)
                - enable_memory: Enable memory plugins (default True)
                - enable_tts: Enable text-to-speech (default False)
                - device: Compute device (cpu/cuda/jetson)
        """
        config = config or {}

        # Configure HRM orchestrator with sensible defaults for kernel
        orchestrator_config = {
            'total_ATP': config.get('total_ATP', 1000.0),
            'max_workers': config.get('max_workers', 4),
            'trust_update_rate': config.get('trust_update_rate', 0.1),
            'device': config.get('device', 'cpu'),  # Default to CPU for safety

            # Plugin enables
            'enable_vision': config.get('enable_vision', True),
            'enable_language': config.get('enable_language', True),
            'enable_control': config.get('enable_control', False),
            'enable_memory': config.get('enable_memory', True),
            'enable_tts': config.get('enable_tts', False),
        }

        try:
            self.orchestrator = HRMOrchestrator(orchestrator_config)
            self.available = True
            print(f"[PluginRouter] Initialized with {len(self.orchestrator.plugins)} plugins")
        except Exception as e:
            print(f"[PluginRouter] Failed to initialize orchestrator: {e}")
            print("[PluginRouter] Running in no-plugin mode")
            self.orchestrator = None
            self.available = False

    def get_available_plugins(self) -> List[str]:
        """
        Get list of available plugin names

        Returns:
            List of plugin names ready for invocation
        """
        if not self.available or not self.orchestrator:
            return []

        return list(self.orchestrator.plugins.keys())

    def get_plugin_trust_scores(self) -> Dict[str, float]:
        """
        Get current trust scores for all plugins

        Returns:
            Dict mapping plugin names to trust scores (0.0 to 1.0)
        """
        if not self.available or not self.orchestrator:
            return {}

        return self.orchestrator.trust_weights.copy()

    async def run_plugins(
        self,
        context: Dict[str, Any],
        allocations: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Run selected plugins with ATP budget allocations

        Args:
            context: Execution context including:
                - goal: High-level goal/objective
                - observations: Current sensor inputs
                - constraints: Operational constraints
            allocations: ATP budget per plugin

        Returns:
            Aggregated results from plugin executions:
                - results: Dict of plugin_name -> result data
                - trust_scores: Updated trust scores
                - execution_times: Time per plugin
                - total_budget_used: Total ATP consumed
        """
        if not self.available or not self.orchestrator:
            return {
                'results': {},
                'trust_scores': {},
                'execution_times': {},
                'total_budget_used': 0.0,
                'status': 'no_plugins_available'
            }

        # Extract plugin names from allocations
        selected_plugins = list(allocations.keys())

        # Filter for available plugins
        available = set(self.orchestrator.plugins.keys())
        selected_plugins = [p for p in selected_plugins if p in available]

        if not selected_plugins:
            return {
                'results': {},
                'trust_scores': self.get_plugin_trust_scores(),
                'execution_times': {},
                'total_budget_used': 0.0,
                'status': 'no_valid_plugins'
            }

        # Execute plugins
        results = {}
        execution_times = {}
        total_budget_used = 0.0

        for plugin_name in selected_plugins:
            try:
                start_time = time.time()

                # Get plugin
                plugin = self.orchestrator.plugins[plugin_name]

                # Initialize state (simplified - in v2 will handle different modalities properly)
                # For now, just create a simple context state
                initial_state = self._create_plugin_context(plugin, context)

                # Run refinement
                history = await self._run_plugin_refinement(
                    plugin,
                    initial_state,
                    budget=allocations[plugin_name]
                )

                # Extract result
                final_state = history[-1] if history else initial_state

                results[plugin_name] = {
                    'final_state': final_state,
                    'history_length': len(history),
                    'converged': plugin.halt(history) if history else False,
                    'final_energy': final_state.energy_val if final_state.energy_val else 0.0
                }

                execution_time = time.time() - start_time
                execution_times[plugin_name] = execution_time
                total_budget_used += allocations[plugin_name]

            except Exception as e:
                print(f"[PluginRouter] Error executing {plugin_name}: {e}")
                results[plugin_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
                execution_times[plugin_name] = 0.0

        # Update trust scores based on results (simplified)
        for plugin_name in selected_plugins:
            if plugin_name in results and 'error' not in results[plugin_name]:
                # Successful execution - small trust increase
                current_trust = self.orchestrator.trust_weights.get(plugin_name, 1.0)
                self.orchestrator.trust_weights[plugin_name] = min(current_trust * 1.05, 2.0)

        return {
            'results': results,
            'trust_scores': self.get_plugin_trust_scores(),
            'execution_times': execution_times,
            'total_budget_used': total_budget_used,
            'status': 'success'
        }

    def _create_plugin_context(
        self,
        plugin: IRPPlugin,
        context: Dict[str, Any]
    ) -> IRPState:
        """
        Create initial IRPState for a plugin from kernel context

        Args:
            plugin: IRP plugin instance
            context: Kernel context

        Returns:
            Initial IRPState for refinement
        """
        # Simplified context creation - v2 will handle different modalities
        # For now, create a minimal state
        task_ctx = {
            'goal': context.get('goal', 'observe'),
            'constraints': context.get('constraints', {}),
        }

        # Initialize with plugin's init_state if available
        try:
            x0 = context.get('observations', None)
            state = plugin.init_state(x0, task_ctx)
            return state
        except Exception as e:
            # Fallback to generic state
            return IRPState(
                x=context,
                step_idx=0,
                energy_val=1.0,
                meta={'source': 'fallback'}
            )

    async def _run_plugin_refinement(
        self,
        plugin: IRPPlugin,
        initial_state: IRPState,
        budget: float,
        max_steps: int = 10
    ) -> List[IRPState]:
        """
        Run iterative refinement for a plugin

        Args:
            plugin: IRP plugin to execute
            initial_state: Starting state
            budget: ATP budget allocation
            max_steps: Maximum refinement iterations

        Returns:
            History of states from refinement process
        """
        history = [initial_state]
        current_state = initial_state

        # Determine max iterations from budget (simple heuristic)
        # 1 ATP = 1 iteration for now
        budget_steps = min(int(budget), max_steps)

        for step in range(budget_steps):
            try:
                # Execute one refinement step
                next_state = plugin.step(current_state)

                # Compute energy
                energy = plugin.energy(next_state)
                next_state.energy_val = energy
                next_state.step_idx = step + 1

                history.append(next_state)
                current_state = next_state

                # Check halt condition
                if plugin.halt(history):
                    break

            except Exception as e:
                print(f"[PluginRouter] Error in refinement step {step}: {e}")
                break

        return history

    def shutdown(self):
        """Clean shutdown of plugin router"""
        if self.orchestrator and hasattr(self.orchestrator, 'executor'):
            self.orchestrator.executor.shutdown(wait=True)
            print("[PluginRouter] Shutdown complete")
