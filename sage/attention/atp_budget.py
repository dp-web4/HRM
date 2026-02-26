#!/usr/bin/env python3
"""
ATP (Attention Transfer Packet) Budget System

Simple trust-weighted resource allocation for plugins.
Based on existing HRM ATP concept but simplified for kernel v1.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class PluginTrust:
    """Trust metrics for a plugin"""
    name: str
    trust: float  # 0.0 to 1.0
    efficiency: float  # resources used / results produced
    success_rate: float  # successful calls / total calls

    def update_from_result(self, success: bool, resources_used: float, value_produced: float):
        """Update trust based on execution result"""
        # Update success rate (running average)
        self.success_rate = 0.9 * self.success_rate + 0.1 * (1.0 if success else 0.0)

        # Update efficiency
        if value_produced > 0:
            current_eff = resources_used / value_produced
            self.efficiency = 0.9 * self.efficiency + 0.1 * current_eff

        # Trust combines success and efficiency
        self.trust = 0.7 * self.success_rate + 0.3 * (1.0 / max(self.efficiency, 0.1))
        self.trust = min(max(self.trust, 0.0), 1.0)  # Clamp to [0, 1]


class ATPBudget:
    """
    Trust-weighted ATP allocation system

    Allocates computational budget across plugins based on:
    - Current trust scores
    - Historical efficiency
    - Available total budget
    """

    def __init__(self, total_budget: float = 1000.0):
        self.total_budget = total_budget
        self.available = total_budget
        self.plugin_trust: Dict[str, PluginTrust] = {}
        self.allocations: Dict[str, float] = {}

    def register_plugin(self, name: str, initial_trust: float = 0.5):
        """Register a plugin with initial trust"""
        if name not in self.plugin_trust:
            self.plugin_trust[name] = PluginTrust(
                name=name,
                trust=initial_trust,
                efficiency=1.0,
                success_rate=0.5
            )

    def allocate(self, plugin_names: List[str]) -> Dict[str, float]:
        """
        Allocate budget across requested plugins based on trust weights

        Returns dict of {plugin_name: allocated_budget}
        """
        # Ensure all plugins are registered
        for name in plugin_names:
            if name not in self.plugin_trust:
                self.register_plugin(name)

        # Calculate trust-weighted allocation
        total_trust = sum(self.plugin_trust[name].trust for name in plugin_names)

        if total_trust == 0:
            # Equal split if no trust info
            allocation_per_plugin = self.available / len(plugin_names)
            allocations = {name: allocation_per_plugin for name in plugin_names}
        else:
            # Trust-weighted allocation
            allocations = {
                name: (self.plugin_trust[name].trust / total_trust) * self.available
                for name in plugin_names
            }

        self.allocations = allocations
        return allocations

    def consume(self, plugin_name: str, amount: float) -> bool:
        """
        Consume budget for a plugin

        Returns True if consumption succeeded, False if insufficient budget
        """
        if plugin_name not in self.allocations:
            return False

        if self.allocations[plugin_name] >= amount:
            self.allocations[plugin_name] -= amount
            self.available -= amount
            return True
        return False

    def reclaim_unused(self):
        """Reclaim unused budget from all plugins"""
        for plugin_name, remaining in self.allocations.items():
            if remaining > 0:
                self.available += remaining
        self.allocations.clear()

    def report_result(self, plugin_name: str, success: bool, resources_used: float, value_produced: float):
        """Report plugin execution result for trust update"""
        if plugin_name in self.plugin_trust:
            self.plugin_trust[plugin_name].update_from_result(success, resources_used, value_produced)

    def reset(self, total_budget: float = None):
        """Reset budget for new tick"""
        if total_budget is not None:
            self.total_budget = total_budget
        self.available = self.total_budget
        self.allocations.clear()

    def get_trust_scores(self) -> Dict[str, float]:
        """Get current trust scores for all plugins"""
        return {name: pt.trust for name, pt in self.plugin_trust.items()}
