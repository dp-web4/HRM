"""
HRM Orchestrator for IRP Plugins
Version: 1.0 (2025-08-23)

Asynchronous orchestration of IRP plugins with trust-based resource allocation.
"""

import asyncio
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import time
import json
from dataclasses import dataclass, field

from .base import IRPPlugin, IRPState
from .vision import VisionIRP
from .language import LanguageIRP
from .control import ControlIRP
from .memory import MemoryIRP


@dataclass
class PluginResult:
    """Result from an IRP plugin execution."""
    plugin_name: str
    final_state: IRPState
    history: List[IRPState]
    telemetry: Dict[str, Any]
    budget_used: float
    execution_time: float


class HRMOrchestrator:
    """
    Orchestrates multiple IRP plugins through HRM's hierarchical architecture.
    
    Key features:
    - Asynchronous plugin execution
    - Trust-weighted budget allocation
    - Dynamic resource reallocation
    - Integrated telemetry and monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize orchestrator with plugin configuration.
        
        Config parameters:
            - total_ATP: Total energy budget for computation
            - max_workers: Maximum parallel workers
            - trust_update_rate: Rate of trust weight updates
            - telemetry_interval: Interval for telemetry emission
            - device: Compute device
        """
        self.config = config
        self.total_ATP = config.get('total_ATP', 100.0)
        self.max_workers = config.get('max_workers', 4)
        self.trust_update_rate = config.get('trust_update_rate', 0.1)
        self.telemetry_interval = config.get('telemetry_interval', 10)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize plugins
        self.plugins = self._initialize_plugins()
        
        # Trust weights (learned through experience)
        self.trust_weights = {
            name: 1.0 for name in self.plugins
        }
        
        # Telemetry storage
        self.telemetry_history = []
        
        # Thread pool for parallel execution
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
        # H-module state for integration
        self.h_state = None
        
    def _initialize_plugins(self) -> Dict[str, IRPPlugin]:
        """Initialize all available IRP plugins."""
        plugins = {}
        
        # Vision plugin
        if self.config.get('enable_vision', True):
            vision_config = {
                **self.config.get('vision_config', {}),
                'entity_id': 'vision_irp',
                'device': self.device
            }
            plugins['vision'] = VisionIRP(vision_config)
        
        # Language plugin
        if self.config.get('enable_language', True):
            language_config = {
                **self.config.get('language_config', {}),
                'entity_id': 'language_irp',
                'device': self.device
            }
            plugins['language'] = LanguageIRP(language_config)
        
        # Control plugin
        if self.config.get('enable_control', True):
            control_config = {
                **self.config.get('control_config', {}),
                'entity_id': 'control_irp',
                'device': self.device
            }
            plugins['control'] = ControlIRP(control_config)
        
        # Memory plugin
        if self.config.get('enable_memory', True):
            memory_config = {
                **self.config.get('memory_config', {}),
                'entity_id': 'memory_irp',
                'device': self.device
            }
            plugins['memory'] = MemoryIRP(memory_config)
        
        return plugins
    
    def allocate_budgets(self, available_ATP: float) -> Dict[str, float]:
        """
        Allocate ATP budget across plugins based on trust weights.
        
        Args:
            available_ATP: Total ATP available
            
        Returns:
            Budget allocation per plugin
        """
        # Normalize trust weights
        total_trust = sum(self.trust_weights.values())
        if total_trust == 0:
            # Equal allocation if no trust established
            equal_budget = available_ATP / len(self.plugins)
            return {name: equal_budget for name in self.plugins}
        
        # Proportional allocation with minimum guarantee
        min_ATP = available_ATP * 0.05  # 5% minimum per plugin
        budgets = {}
        
        for name, plugin in self.plugins.items():
            weight = self.trust_weights[name] / total_trust
            allocated = available_ATP * weight
            budgets[name] = max(min_ATP, allocated)
        
        # Ensure we don't exceed total budget
        total_allocated = sum(budgets.values())
        if total_allocated > available_ATP:
            scale = available_ATP / total_allocated
            budgets = {name: budget * scale for name, budget in budgets.items()}
        
        return budgets
    
    def reallocate_budget(self, freed_ATP: float, active_plugins: List[str]) -> Dict[str, float]:
        """
        Reallocate freed budget to active plugins.
        
        Args:
            freed_ATP: Amount of ATP freed up
            active_plugins: List of still-active plugin names
            
        Returns:
            Additional budget per active plugin
        """
        if not active_plugins:
            return {}
        
        # Get trust weights for active plugins
        active_weights = {
            name: self.trust_weights[name]
            for name in active_plugins
        }
        
        total_weight = sum(active_weights.values())
        if total_weight == 0:
            # Equal distribution if no weights
            equal_share = freed_ATP / len(active_plugins)
            return {name: equal_share for name in active_plugins}
        
        # Proportional distribution
        additional_budgets = {}
        for name, weight in active_weights.items():
            share = weight / total_weight
            additional_budgets[name] = freed_ATP * share
        
        return additional_budgets
    
    def run_plugin(self, plugin_name: str, plugin: IRPPlugin, 
                   input_data: Any, budget: float) -> PluginResult:
        """
        Run a single IRP plugin with budget constraint.
        
        Args:
            plugin_name: Name of the plugin
            plugin: IRP plugin instance
            input_data: Input for the plugin
            budget: ATP budget for this plugin
            
        Returns:
            PluginResult with execution details
        """
        start_time = time.time()
        
        # Configure plugin with budget
        plugin.config['max_ATP'] = budget
        plugin.config['max_iterations'] = int(budget * 10)  # Simplified ATP->iterations
        
        # Extract task context if provided
        task_ctx = {}
        if isinstance(input_data, dict) and 'task_ctx' in input_data:
            task_ctx = input_data['task_ctx']
            input_data = input_data.get('data', input_data)
        
        # Run refinement
        final_state, history = plugin.refine(input_data, task_ctx)
        
        # Compute budget used
        budget_used = len(history) * 0.1  # Simplified: 0.1 ATP per iteration
        
        # Generate telemetry
        telemetry = plugin.emit_telemetry(final_state, history)
        
        execution_time = time.time() - start_time
        
        return PluginResult(
            plugin_name=plugin_name,
            final_state=final_state,
            history=history,
            telemetry=telemetry,
            budget_used=budget_used,
            execution_time=execution_time
        )
    
    async def process_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs asynchronously across plugins.
        
        Args:
            inputs: Dictionary of inputs per plugin
            
        Returns:
            Integrated results from all plugins
        """
        # Allocate initial budgets
        budgets = self.allocate_budgets(self.total_ATP)
        
        # Create futures for parallel execution
        loop = asyncio.get_event_loop()
        futures = {}
        
        for name, plugin in self.plugins.items():
            if name in inputs:
                future = loop.run_in_executor(
                    self.executor,
                    self.run_plugin,
                    name,
                    plugin,
                    inputs[name],
                    budgets[name]
                )
                futures[name] = future
        
        # Collect results as they complete
        results = {}
        remaining_ATP = self.total_ATP
        active_plugins = list(futures.keys())
        
        while futures:
            # Wait for any plugin to complete
            done, pending = await asyncio.wait(
                futures.values(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Process completed plugins
            for future in done:
                # Find which plugin completed
                plugin_name = None
                for name, f in futures.items():
                    if f == future:
                        plugin_name = name
                        break
                
                if plugin_name:
                    # Get result
                    result = await future
                    results[plugin_name] = result
                    
                    # Update remaining budget
                    freed_ATP = budgets[plugin_name] - result.budget_used
                    remaining_ATP -= result.budget_used
                    
                    # Remove from active list
                    del futures[plugin_name]
                    active_plugins.remove(plugin_name)
                    
                    # Reallocate freed budget
                    if freed_ATP > 0 and active_plugins:
                        additional = self.reallocate_budget(freed_ATP, active_plugins)
                        for name, extra in additional.items():
                            if name in budgets:
                                budgets[name] += extra
                    
                    # Store telemetry
                    self.telemetry_history.append(result.telemetry)
        
        # Integrate results
        integrated = self.integrate_results(results)
        
        # Update trust weights
        self.update_trust_weights(results, integrated)
        
        return integrated
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous wrapper for async processing.
        
        Args:
            inputs: Dictionary of inputs per plugin
            
        Returns:
            Integrated results
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_async(inputs))
        finally:
            loop.close()
    
    def integrate_results(self, results: Dict[str, PluginResult]) -> Dict[str, Any]:
        """
        Integrate results from multiple plugins (H-module function).
        
        Args:
            results: Results from each plugin
            
        Returns:
            Integrated output
        """
        integrated = {
            'plugin_outputs': {},
            'system_coherence': 0.0,
            'total_energy': 0.0,
            'total_ATP_used': 0.0,
            'execution_times': {}
        }
        
        # Extract outputs and metrics
        for name, result in results.items():
            # Get plugin-specific output
            if name == 'vision':
                from .vision import VisionIRP
                plugin = self.plugins[name]
                output = plugin.get_semantic_representation(result.final_state)
            elif name == 'language':
                from .language import LanguageIRP
                plugin = self.plugins[name]
                output = plugin.get_understanding(result.final_state)
            elif name == 'control':
                from .control import ControlIRP
                plugin = self.plugins[name]
                output = plugin.get_trajectory(result.final_state)
            elif name == 'memory':
                from .memory import MemoryIRP
                plugin = self.plugins[name]
                output = plugin.get_consolidated_memory(result.final_state)
            else:
                output = {'final_state': result.final_state.x}
            
            integrated['plugin_outputs'][name] = output
            integrated['execution_times'][name] = result.execution_time
            integrated['total_ATP_used'] += result.budget_used
            
            # Accumulate energy
            if result.final_state.energy_val is not None:
                integrated['total_energy'] += result.final_state.energy_val
        
        # Compute system coherence (simplified: inverse of total energy)
        if integrated['total_energy'] != 0:
            integrated['system_coherence'] = 1.0 / (1.0 + abs(integrated['total_energy']))
        else:
            integrated['system_coherence'] = 1.0
        
        # Add trust weights for transparency
        integrated['trust_weights'] = self.trust_weights.copy()
        
        return integrated
    
    def update_trust_weights(self, results: Dict[str, PluginResult], 
                            integrated: Dict[str, Any]):
        """
        Update trust weights based on plugin performance.
        
        Args:
            results: Individual plugin results
            integrated: Integrated system output
        """
        system_coherence = integrated['system_coherence']
        
        for name, result in results.items():
            # Get trust metrics from telemetry
            trust_metrics = result.telemetry.get('trust', {})
            
            # Base trust from convergence quality
            monotonicity = trust_metrics.get('monotonicity_ratio', 0.5)
            
            # Contribution to system coherence
            contribution = trust_metrics.get('contribution_to_H', 0.0)
            system_modifier = self._sigmoid(contribution / (system_coherence + 1e-6))
            
            # Efficiency bonus
            budget_ratio = result.budget_used / self.plugins[name].config.get('max_ATP', 10.0)
            efficiency = 1.0 - min(budget_ratio, 1.0)
            
            # Update with momentum
            old_trust = self.trust_weights[name]
            new_trust = (
                0.7 * old_trust +
                0.2 * monotonicity * system_modifier +
                0.1 * efficiency
            )
            
            # Apply update with learning rate
            self.trust_weights[name] = (
                (1 - self.trust_update_rate) * old_trust +
                self.trust_update_rate * new_trust
            )
            
            # Clamp to valid range
            self.trust_weights[name] = np.clip(self.trust_weights[name], 0.1, 10.0)
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-x))
    
    def get_telemetry_summary(self) -> Dict[str, Any]:
        """
        Get summary of telemetry data.
        
        Returns:
            Summary statistics
        """
        if not self.telemetry_history:
            return {}
        
        summary = {
            'total_runs': len(self.telemetry_history),
            'average_ATP_per_run': np.mean([
                t.get('budget', {}).get('ATP_spent', 0)
                for t in self.telemetry_history
            ]),
            'average_convergence_steps': np.mean([
                t.get('steps', 0)
                for t in self.telemetry_history
            ]),
            'trust_evolution': self.trust_weights.copy(),
            'halt_reasons': {}
        }
        
        # Count halt reasons
        for telemetry in self.telemetry_history:
            reason = telemetry.get('halt_reason', 'unknown')
            summary['halt_reasons'][reason] = summary['halt_reasons'].get(reason, 0) + 1
        
        return summary
    
    def save_state(self, filepath: str):
        """Save orchestrator state to file."""
        state = {
            'trust_weights': self.trust_weights,
            'telemetry_history': self.telemetry_history,
            'config': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self, filepath: str):
        """Load orchestrator state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.trust_weights = state.get('trust_weights', {})
        self.telemetry_history = state.get('telemetry_history', [])
        
        # Update config if needed
        if 'config' in state:
            self.config.update(state['config'])