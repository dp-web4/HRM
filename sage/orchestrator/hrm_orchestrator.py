"""
HRM Orchestrator - Manages concurrent IRP plugins with trust-weighted budgets
Implements ATP (Adaptive Trust Points) allocation for resource management
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import torch
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sage.irp.base import IRPPlugin
from sage.irp.plugins.vision_impl import create_vision_irp
from sage.irp.plugins.language_impl import create_language_irp


class PluginState(Enum):
    """Plugin execution states"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    HALTED_EARLY = "halted_early"


@dataclass
class PluginResult:
    """Result from a plugin execution"""
    plugin_id: str
    state: PluginState
    output: Any
    telemetry: Dict[str, Any]
    start_time: float
    end_time: float
    atp_consumed: float
    trust_score: float
    
    @property
    def execution_time(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def efficiency(self) -> float:
        """ATP efficiency = trust_score / atp_consumed"""
        if self.atp_consumed > 0:
            return self.trust_score / self.atp_consumed
        return 0.0


@dataclass
class ATPBudget:
    """Adaptive Trust Points budget management"""
    total: float = 1000.0
    allocated: Dict[str, float] = field(default_factory=dict)
    consumed: Dict[str, float] = field(default_factory=dict)
    trust_weights: Dict[str, float] = field(default_factory=dict)
    
    def allocate(self, plugin_id: str, trust_weight: float) -> float:
        """Allocate ATP based on trust weight"""
        # Normalize trust weights
        total_weight = sum(self.trust_weights.values()) or 1.0
        normalized_weight = trust_weight / total_weight
        
        # Allocate proportional ATP
        allocation = self.total * normalized_weight
        self.allocated[plugin_id] = allocation
        self.trust_weights[plugin_id] = trust_weight
        
        return allocation
    
    def consume(self, plugin_id: str, amount: float) -> bool:
        """Consume ATP, return False if budget exceeded"""
        if plugin_id not in self.consumed:
            self.consumed[plugin_id] = 0.0
        
        available = self.allocated.get(plugin_id, 0) - self.consumed[plugin_id]
        if amount <= available:
            self.consumed[plugin_id] += amount
            return True
        return False
    
    def reallocate_unused(self):
        """Reallocate unused ATP from completed plugins"""
        unused_total = 0.0
        active_plugins = []
        
        for plugin_id, allocated in self.allocated.items():
            consumed = self.consumed.get(plugin_id, 0)
            if consumed < allocated * 0.9:  # Plugin likely finished early
                unused = allocated - consumed
                unused_total += unused * 0.5  # Reclaim 50% of unused
            else:
                active_plugins.append(plugin_id)
        
        # Redistribute to active plugins
        if active_plugins and unused_total > 0:
            per_plugin = unused_total / len(active_plugins)
            for plugin_id in active_plugins:
                self.allocated[plugin_id] += per_plugin
    
    def get_efficiency_report(self) -> Dict[str, Any]:
        """Generate efficiency report"""
        return {
            "total_budget": self.total,
            "total_allocated": sum(self.allocated.values()),
            "total_consumed": sum(self.consumed.values()),
            "utilization": sum(self.consumed.values()) / self.total if self.total > 0 else 0,
            "per_plugin": {
                pid: {
                    "allocated": self.allocated.get(pid, 0),
                    "consumed": self.consumed.get(pid, 0),
                    "utilization": self.consumed.get(pid, 0) / self.allocated.get(pid, 1) 
                                  if self.allocated.get(pid, 0) > 0 else 0
                }
                for pid in self.allocated.keys()
            }
        }


class HRMOrchestrator:
    """
    Orchestrates multiple IRP plugins with concurrent execution
    and trust-weighted resource allocation
    """
    
    def __init__(
        self,
        initial_atp: float = 1000.0,
        max_concurrent: int = 4,
        reallocation_interval: float = 0.1,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.plugins: Dict[str, IRPPlugin] = {}
        self.plugin_states: Dict[str, PluginState] = {}
        self.results: List[PluginResult] = []
        
        # Budget management
        self.budget = ATPBudget(total=initial_atp)
        self.max_concurrent = max_concurrent
        self.reallocation_interval = reallocation_interval
        
        # Execution tracking
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.start_time = None
        
    def register_plugin(
        self,
        plugin_id: str,
        plugin: IRPPlugin,
        initial_trust: float = 1.0
    ):
        """Register an IRP plugin with the orchestrator"""
        self.plugins[plugin_id] = plugin
        self.plugin_states[plugin_id] = PluginState.IDLE
        self.budget.allocate(plugin_id, initial_trust)
        
    def create_default_plugins(self):
        """Create and register default Vision and Language plugins"""
        # Vision plugin
        vision_irp = create_vision_irp(self.device)
        self.register_plugin("vision", vision_irp, initial_trust=1.0)
        
        # Language plugin
        language_irp = create_language_irp(self.device)
        self.register_plugin("language", language_irp, initial_trust=1.0)
        
    async def execute_plugin(
        self,
        plugin_id: str,
        input_data: Any,
        early_stop: bool = True
    ) -> PluginResult:
        """Execute a single plugin asynchronously"""
        if plugin_id not in self.plugins:
            raise ValueError(f"Plugin {plugin_id} not registered")
        
        plugin = self.plugins[plugin_id]
        self.plugin_states[plugin_id] = PluginState.RUNNING
        
        start_time = time.time()
        atp_consumed = 0.0
        
        try:
            # Run plugin refinement in executor to avoid blocking
            loop = asyncio.get_event_loop()
            output, telemetry = await loop.run_in_executor(
                None,
                plugin.refine,
                input_data,
                early_stop
            )
            
            # Calculate ATP consumption (based on iterations)
            iterations = telemetry.get('iterations', 1)
            atp_per_iteration = 10.0  # Base cost per iteration
            atp_consumed = iterations * atp_per_iteration
            
            # Check budget
            if not self.budget.consume(plugin_id, atp_consumed):
                self.plugin_states[plugin_id] = PluginState.FAILED
                state = PluginState.FAILED
            else:
                state = PluginState.HALTED_EARLY if telemetry.get('early_stopped', False) else PluginState.COMPLETED
                self.plugin_states[plugin_id] = state
            
            # Get trust score
            trust_score = telemetry.get('trust', 0.5)
            
        except Exception as e:
            print(f"Plugin {plugin_id} failed: {e}")
            self.plugin_states[plugin_id] = PluginState.FAILED
            output = None
            telemetry = {"error": str(e)}
            state = PluginState.FAILED
            trust_score = 0.0
        
        end_time = time.time()
        
        result = PluginResult(
            plugin_id=plugin_id,
            state=state,
            output=output,
            telemetry=telemetry,
            start_time=start_time,
            end_time=end_time,
            atp_consumed=atp_consumed,
            trust_score=trust_score
        )
        
        self.results.append(result)
        return result
    
    async def execute_parallel(
        self,
        tasks: Dict[str, Any],
        early_stop: bool = True
    ) -> List[PluginResult]:
        """Execute multiple plugins in parallel"""
        execution_tasks = []
        
        for plugin_id, input_data in tasks.items():
            if plugin_id in self.plugins:
                task = asyncio.create_task(
                    self.execute_plugin(plugin_id, input_data, early_stop)
                )
                execution_tasks.append(task)
                self.running_tasks[plugin_id] = task
        
        # Start reallocation monitor
        reallocation_task = asyncio.create_task(self._reallocation_monitor())
        
        # Wait for all plugins to complete
        results = await asyncio.gather(*execution_tasks)
        
        # Stop reallocation monitor
        reallocation_task.cancel()
        try:
            await reallocation_task
        except asyncio.CancelledError:
            pass
        
        return results
    
    async def _reallocation_monitor(self):
        """Monitor and reallocate ATP during execution"""
        while True:
            await asyncio.sleep(self.reallocation_interval)
            
            # Check for early-stopped plugins
            completed = [
                pid for pid, state in self.plugin_states.items()
                if state in [PluginState.COMPLETED, PluginState.HALTED_EARLY]
            ]
            
            if completed:
                # Reallocate unused budget
                self.budget.reallocate_unused()
                
                # Update trust weights based on results
                for result in self.results:
                    if result.plugin_id in completed and result.efficiency > 0:
                        current_trust = self.budget.trust_weights.get(result.plugin_id, 1.0)
                        new_trust = current_trust * (0.9 + 0.1 * result.efficiency)
                        self.budget.trust_weights[result.plugin_id] = new_trust
    
    def get_orchestration_summary(self) -> Dict[str, Any]:
        """Generate summary of orchestration performance"""
        if not self.results:
            return {"status": "no_results"}
        
        total_time = max(r.end_time for r in self.results) - min(r.start_time for r in self.results)
        
        plugin_summaries = {}
        for result in self.results:
            plugin_summaries[result.plugin_id] = {
                "state": result.state.value,
                "execution_time": result.execution_time,
                "atp_consumed": result.atp_consumed,
                "trust_score": result.trust_score,
                "efficiency": result.efficiency,
                "iterations": result.telemetry.get('iterations', 0),
                "compute_saved": result.telemetry.get('compute_saved', 0)
            }
        
        return {
            "total_execution_time": total_time,
            "plugins_executed": len(self.results),
            "successful": sum(1 for r in self.results if r.state != PluginState.FAILED),
            "early_stopped": sum(1 for r in self.results if r.state == PluginState.HALTED_EARLY),
            "average_efficiency": sum(r.efficiency for r in self.results) / len(self.results),
            "budget_report": self.budget.get_efficiency_report(),
            "plugin_results": plugin_summaries
        }
    
    async def run_demonstration(self):
        """Run a demonstration of orchestrated execution"""
        print("=" * 60)
        print("HRM Orchestrator Demonstration")
        print("=" * 60)
        
        # Create test data
        vision_data = torch.randn(2, 3, 224, 224).to(self.device)
        language_data = torch.randint(100, 5000, (2, 32)).to(self.device)
        
        tasks = {
            "vision": vision_data,
            "language": language_data
        }
        
        print("\n1. Starting parallel execution...")
        print(f"   Vision task: {vision_data.shape}")
        print(f"   Language task: {language_data.shape}")
        print(f"   Initial ATP budget: {self.budget.total}")
        
        # Execute in parallel
        start = time.time()
        results = await self.execute_parallel(tasks, early_stop=True)
        total_time = time.time() - start
        
        print(f"\n2. Execution completed in {total_time:.2f}s")
        
        # Show results
        for result in results:
            print(f"\n   {result.plugin_id.upper()}:")
            print(f"     State: {result.state.value}")
            print(f"     Time: {result.execution_time:.3f}s")
            print(f"     ATP consumed: {result.atp_consumed:.1f}")
            print(f"     Trust score: {result.trust_score:.3f}")
            print(f"     Efficiency: {result.efficiency:.3f}")
            if 'iterations' in result.telemetry:
                print(f"     Iterations: {result.telemetry['iterations']}")
            if 'compute_saved' in result.telemetry:
                print(f"     Compute saved: {result.telemetry['compute_saved']*100:.1f}%")
        
        # Show summary
        summary = self.get_orchestration_summary()
        
        print("\n3. Orchestration Summary:")
        print(f"   Total execution: {summary['total_execution_time']:.3f}s")
        print(f"   Plugins successful: {summary['successful']}/{summary['plugins_executed']}")
        print(f"   Early stopped: {summary['early_stopped']}")
        print(f"   Average efficiency: {summary['average_efficiency']:.3f}")
        print(f"   Budget utilization: {summary['budget_report']['utilization']*100:.1f}%")
        
        print("\n" + "=" * 60)
        
        return summary


async def main():
    """Test the orchestrator"""
    # Create orchestrator
    orchestrator = HRMOrchestrator(initial_atp=1000.0)
    
    # Register default plugins
    orchestrator.create_default_plugins()
    
    # Run demonstration
    summary = await orchestrator.run_demonstration()
    
    # Save results
    with open('orchestrator_results.json', 'w') as f:
        # Convert summary to JSON-serializable format
        json_summary = json.dumps(summary, default=str, indent=2)
        f.write(json_summary)
    
    print("\n✓ Results saved to orchestrator_results.json")


if __name__ == "__main__":
    asyncio.run(main())