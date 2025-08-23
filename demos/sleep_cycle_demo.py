#!/usr/bin/env python3
"""
Sleep Cycle Demonstration
Shows how the system consolidates experiences during 'sleep'
Extracts patterns and improves performance after consolidation
"""

import asyncio
import torch
import time
import json
import sys
import os
from typing import List, Dict, Any, Optional
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sage.orchestrator.hrm_orchestrator import HRMOrchestrator
from sage.memory.irp_memory_bridge import IRPMemoryBridge, MemoryGuidedIRP
from sage.irp.plugins.vision_impl import create_vision_irp
from sage.irp.plugins.language_impl import create_language_irp


class DayCycle:
    """Simulates a day of experiences and a night of consolidation"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create memory bridge
        self.memory_bridge = IRPMemoryBridge(
            buffer_size=100,
            consolidation_threshold=20  # Consolidate every 20 experiences
        )
        
        # Create orchestrator
        self.orchestrator = HRMOrchestrator(
            initial_atp=2000.0,  # More budget for full day
            max_concurrent=2
        )
        
        # Create memory-guided plugins
        vision_irp = create_vision_irp(self.device)
        language_irp = create_language_irp(self.device)
        
        self.vision_guided = MemoryGuidedIRP(vision_irp, self.memory_bridge)
        self.language_guided = MemoryGuidedIRP(language_irp, self.memory_bridge)
        
        # Register with orchestrator
        self.orchestrator.register_plugin("vision", self.vision_guided, initial_trust=1.0)
        self.orchestrator.register_plugin("language", self.language_guided, initial_trust=1.0)
        
        # Track day's experiences
        self.day_experiences = []
        self.performance_before_sleep = {}
        self.performance_after_sleep = {}
    
    async def simulate_day(self, num_tasks: int = 30):
        """
        Simulate a day of varied tasks
        """
        print("=" * 70)
        print("DAY PHASE - Collecting Experiences")
        print("=" * 70)
        
        task_types = ['vision_only', 'language_only', 'multi_modal']
        
        for i in range(num_tasks):
            task_type = task_types[i % len(task_types)]
            
            if task_type == 'vision_only':
                # Vision task
                image = torch.randn(2, 3, 224, 224).to(self.device)
                tasks = {"vision": image}
                
            elif task_type == 'language_only':
                # Language task
                tokens = torch.randint(100, 5000, (2, 32)).to(self.device)
                tasks = {"language": tokens}
                
            else:
                # Multi-modal task
                image = torch.randn(1, 3, 224, 224).to(self.device)
                tokens = torch.randint(100, 5000, (1, 32)).to(self.device)
                tasks = {"vision": image, "language": tokens}
            
            # Execute task
            results = await self.orchestrator.execute_parallel(tasks, early_stop=True)
            
            # Store experience
            experience = {
                'task_id': i,
                'task_type': task_type,
                'results': [
                    {
                        'plugin': r.plugin_id,
                        'iterations': r.telemetry.get('iterations', 0),
                        'efficiency': r.efficiency,
                        'trust': r.trust_score
                    }
                    for r in results
                ]
            }
            self.day_experiences.append(experience)
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_tasks} tasks")
                
                # Show current memory stats
                stats = self.memory_bridge.get_memory_stats()
                print(f"    Memory: {stats['total_memories']} stored, "
                      f"{stats['pending_consolidation']} pending")
        
        print(f"\n✓ Day complete: {len(self.day_experiences)} experiences collected")
    
    def measure_performance(self) -> Dict[str, float]:
        """
        Measure current performance metrics
        """
        # Get memory stats
        mem_stats = self.memory_bridge.get_memory_stats()
        
        # Get orchestrator efficiency
        orch_summary = self.orchestrator.get_orchestration_summary()
        
        metrics = {
            'avg_iterations': mem_stats.get('avg_iterations', 0),
            'avg_efficiency': mem_stats.get('avg_efficiency', 0),
            'avg_convergence': mem_stats.get('avg_convergence', 0),
            'patterns_available': mem_stats.get('patterns_extracted', 0),
            'trust_adaptation': sum(self.orchestrator.budget.trust_weights.values()) / 
                               len(self.orchestrator.budget.trust_weights) 
                               if self.orchestrator.budget.trust_weights else 1.0
        }
        
        return metrics
    
    def sleep_consolidation(self):
        """
        Consolidate day's experiences during 'sleep'
        Extract patterns and update memory
        """
        print("\n" + "=" * 70)
        print("SLEEP PHASE - Memory Consolidation")
        print("=" * 70)
        
        print("\n1. Pre-sleep performance:")
        self.performance_before_sleep = self.measure_performance()
        for key, value in self.performance_before_sleep.items():
            print(f"   {key}: {value:.3f}")
        
        print("\n2. Starting consolidation...")
        
        # Force consolidation of all pending memories
        initial_patterns = len(self.memory_bridge.pattern_library)
        
        # Consolidate in batches (simulating sleep cycles)
        sleep_cycles = 3
        for cycle in range(sleep_cycles):
            print(f"\n   Sleep cycle {cycle + 1}/{sleep_cycles}")
            
            # Consolidate current batch
            self.memory_bridge.consolidate()
            
            # Simulate REM processing - review patterns
            patterns = self.memory_bridge.pattern_library
            if patterns:
                print(f"     Patterns active: {len(patterns)}")
                for pattern_key in list(patterns.keys())[:3]:  # Show first 3
                    pattern = patterns[pattern_key]
                    print(f"       {pattern_key}: efficiency={pattern.get('best_efficiency', 0):.3f}")
            
            # Simulate memory replay (strengthening important patterns)
            if hasattr(self.memory_bridge.snarc, 'memories'):
                important_memories = sorted(
                    self.memory_bridge.snarc.memories,
                    key=lambda m: m.efficiency,
                    reverse=True
                )[:10]
                
                if important_memories:
                    avg_eff = sum(m.efficiency for m in important_memories) / len(important_memories)
                    print(f"     Replaying top memories: avg efficiency={avg_eff:.3f}")
            
            time.sleep(0.5)  # Simulate processing time
        
        new_patterns = len(self.memory_bridge.pattern_library) - initial_patterns
        print(f"\n3. Consolidation complete:")
        print(f"   New patterns extracted: {new_patterns}")
        print(f"   Total patterns available: {len(self.memory_bridge.pattern_library)}")
        
        # Update trust weights based on consolidated knowledge
        print("\n4. Updating trust weights based on experience...")
        
        # Analyze day's experiences to update trust
        plugin_performance = {}
        for exp in self.day_experiences:
            for result in exp['results']:
                plugin = result['plugin']
                if plugin not in plugin_performance:
                    plugin_performance[plugin] = []
                plugin_performance[plugin].append(result['efficiency'])
        
        # Update trust based on average performance
        for plugin, efficiencies in plugin_performance.items():
            avg_eff = sum(efficiencies) / len(efficiencies)
            # Adjust trust: increase for good performance, decrease for poor
            current_trust = self.orchestrator.budget.trust_weights.get(plugin, 1.0)
            new_trust = current_trust * (0.8 + 0.4 * avg_eff)  # Scale between 0.8x and 1.2x
            self.orchestrator.budget.trust_weights[plugin] = new_trust
            print(f"   {plugin}: {current_trust:.3f} → {new_trust:.3f}")
    
    async def morning_test(self, num_tests: int = 10):
        """
        Test performance after sleep consolidation
        """
        print("\n" + "=" * 70)
        print("MORNING PHASE - Testing Improvement")
        print("=" * 70)
        
        print("\n1. Running morning tests with consolidated memory...")
        
        morning_results = []
        
        for i in range(num_tests):
            # Similar tasks to yesterday
            if i % 2 == 0:
                image = torch.randn(1, 3, 224, 224).to(self.device)
                tasks = {"vision": image}
            else:
                tokens = torch.randint(100, 5000, (1, 32)).to(self.device)
                tasks = {"language": tokens}
            
            # Execute with memory guidance
            results = await self.orchestrator.execute_parallel(tasks, early_stop=True)
            
            for r in results:
                morning_results.append({
                    'plugin': r.plugin_id,
                    'iterations': r.telemetry.get('iterations', 0),
                    'efficiency': r.efficiency,
                    'memory_guided': r.telemetry.get('memory_guidance', {}) != {}
                })
        
        print(f"\n2. Morning performance:")
        self.performance_after_sleep = self.measure_performance()
        for key, value in self.performance_after_sleep.items():
            print(f"   {key}: {value:.3f}")
        
        print("\n3. Improvement analysis:")
        for key in self.performance_before_sleep:
            before = self.performance_before_sleep[key]
            after = self.performance_after_sleep.get(key, before)
            if before > 0:
                improvement = ((after - before) / before) * 100
                symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
                print(f"   {key}: {improvement:+.1f}% {symbol}")
        
        # Check if memory guidance was used
        memory_guided_count = sum(1 for r in morning_results if r.get('memory_guided', False))
        print(f"\n4. Memory utilization:")
        print(f"   Tasks using memory guidance: {memory_guided_count}/{len(morning_results)}")
        
        if morning_results:
            avg_morning_iterations = sum(r['iterations'] for r in morning_results) / len(morning_results)
            print(f"   Average iterations (morning): {avg_morning_iterations:.1f}")
        
        return morning_results


async def run_full_day_cycle():
    """Run a complete day-sleep-morning cycle"""
    
    print("=" * 70)
    print("24-HOUR CYCLE DEMONSTRATION")
    print("Sleep Consolidation for Performance Improvement")
    print("=" * 70)
    
    # Create day cycle
    cycle = DayCycle()
    
    # Day phase - collect experiences
    print("\n⏰ Starting day phase...")
    await cycle.simulate_day(num_tasks=30)
    
    # Sleep phase - consolidate
    print("\n😴 Entering sleep phase...")
    cycle.sleep_consolidation()
    
    # Morning phase - test improvement
    print("\n☀️ Morning testing...")
    morning_results = await cycle.morning_test(num_tests=10)
    
    # Final summary
    print("\n" + "=" * 70)
    print("24-HOUR CYCLE COMPLETE")
    print("=" * 70)
    
    # Calculate overall improvement
    before = cycle.performance_before_sleep
    after = cycle.performance_after_sleep
    
    overall_improvement = 0
    improvement_count = 0
    
    for key in before:
        if before[key] > 0 and key in after:
            improvement = ((after[key] - before[key]) / before[key]) * 100
            overall_improvement += improvement
            improvement_count += 1
    
    if improvement_count > 0:
        avg_improvement = overall_improvement / improvement_count
        print(f"\n📊 Average improvement after sleep: {avg_improvement:+.1f}%")
        
        if avg_improvement > 10:
            print("   ✅ Significant improvement achieved through consolidation!")
        elif avg_improvement > 0:
            print("   ✓ Modest improvement from sleep consolidation")
        else:
            print("   ⚠️ No significant improvement (may need more experiences)")
    
    # Save results
    results = {
        'day_experiences': len(cycle.day_experiences),
        'performance_before': cycle.performance_before_sleep,
        'performance_after': cycle.performance_after_sleep,
        'patterns_extracted': len(cycle.memory_bridge.pattern_library),
        'morning_test_results': morning_results
    }
    
    with open('sleep_cycle_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✓ Results saved to sleep_cycle_results.json")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_full_day_cycle())