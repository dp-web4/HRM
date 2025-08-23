#!/usr/bin/env python3
"""
HRM Orchestrator Demo - Multi-modal coordination with trust dynamics
Shows how plugins work together with resource sharing
"""

import asyncio
import torch
import time
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sage.orchestrator.hrm_orchestrator import HRMOrchestrator, PluginState
from sage.irp.plugins.vision_impl import create_vision_irp
from sage.irp.plugins.language_impl import create_language_irp


class MultiModalTask:
    """Represents a task requiring multiple modalities"""
    
    def __init__(self, image_data, text_data, task_type="describe"):
        self.image_data = image_data
        self.text_data = text_data
        self.task_type = task_type
        self.results = {}
    
    def add_result(self, modality: str, result: any):
        """Store result from a modality"""
        self.results[modality] = result
    
    def get_combined_representation(self):
        """Combine results from different modalities"""
        if 'vision' in self.results and 'language' in self.results:
            # Handle both tensor and dict outputs
            vision_result = self.results['vision']
            language_result = self.results['language']
            
            # Extract representations based on type
            if isinstance(vision_result, dict):
                vision_output = vision_result.get('meaning_latent', None)
            elif isinstance(vision_result, torch.Tensor):
                vision_output = vision_result
            else:
                vision_output = None
                
            if isinstance(language_result, dict):
                language_output = language_result.get('meaning_latent', None)
            elif isinstance(language_result, torch.Tensor):
                language_output = language_result
            else:
                language_output = None
            
            if vision_output is not None and language_output is not None:
                # Project to same dimension if needed
                if vision_output.shape[-1] != language_output.shape[-1]:
                    min_dim = min(vision_output.shape[-1], language_output.shape[-1])
                    vision_output = vision_output[..., :min_dim]
                    language_output = language_output[..., :min_dim]
                
                # Weighted combination
                combined = 0.6 * vision_output + 0.4 * language_output
                return combined
        
        return None


async def simulate_workload_variations():
    """Simulate different workload patterns to test orchestration"""
    
    print("=" * 70)
    print("Multi-Modal Orchestration Demo with Trust Dynamics")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create orchestrator with limited budget to force interesting dynamics
    orchestrator = HRMOrchestrator(
        initial_atp=500.0,  # Limited budget
        max_concurrent=2,
        reallocation_interval=0.05  # Fast reallocation
    )
    
    # Register plugins with different initial trust
    vision_irp = create_vision_irp(device)
    language_irp = create_language_irp(device)
    
    orchestrator.register_plugin("vision", vision_irp, initial_trust=1.2)  # Higher trust
    orchestrator.register_plugin("language", language_irp, initial_trust=0.8)  # Lower trust
    
    # Test Case 1: Balanced workload
    print("\n### Test 1: Balanced Workload")
    print("Both vision and language get equal-sized inputs")
    
    vision_batch1 = torch.randn(2, 3, 224, 224).to(device)
    language_batch1 = torch.randint(100, 5000, (2, 32)).to(device)
    
    tasks1 = {
        "vision": vision_batch1,
        "language": language_batch1
    }
    
    results1 = await orchestrator.execute_parallel(tasks1, early_stop=True)
    
    print(f"\nResults:")
    for r in results1:
        print(f"  {r.plugin_id}: {r.state.value}, ATP: {r.atp_consumed:.1f}, Trust: {r.trust_score:.3f}")
    
    # Test Case 2: Heavy vision workload
    print("\n### Test 2: Heavy Vision Workload")
    print("Vision gets larger batch, testing ATP reallocation")
    
    vision_batch2 = torch.randn(8, 3, 224, 224).to(device)  # 4x larger
    language_batch2 = torch.randint(100, 5000, (2, 32)).to(device)
    
    tasks2 = {
        "vision": vision_batch2,
        "language": language_batch2
    }
    
    results2 = await orchestrator.execute_parallel(tasks2, early_stop=True)
    
    print(f"\nResults:")
    for r in results2:
        print(f"  {r.plugin_id}: {r.state.value}, ATP: {r.atp_consumed:.1f}, Trust: {r.trust_score:.3f}")
    
    # Test Case 3: Multi-modal fusion
    print("\n### Test 3: Multi-Modal Fusion Task")
    print("Vision and language work together on same content")
    
    # Create correlated inputs (simulating image + caption)
    task = MultiModalTask(
        image_data=torch.randn(1, 3, 224, 224).to(device),
        text_data=torch.randint(100, 5000, (1, 32)).to(device),
        task_type="describe"
    )
    
    tasks3 = {
        "vision": task.image_data,
        "language": task.text_data
    }
    
    results3 = await orchestrator.execute_parallel(tasks3, early_stop=True)
    
    # Store results in task
    for r in results3:
        if r.output is not None:
            task.add_result(r.plugin_id, r.output)
    
    # Combine representations
    combined = task.get_combined_representation()
    
    print(f"\nResults:")
    for r in results3:
        print(f"  {r.plugin_id}: {r.state.value}, ATP: {r.atp_consumed:.1f}, Trust: {r.trust_score:.3f}")
    
    if combined is not None:
        print(f"\nCombined representation shape: {combined.shape}")
        print(f"Combined representation norm: {torch.norm(combined).item():.3f}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("ORCHESTRATION SUMMARY")
    print("=" * 70)
    
    summary = orchestrator.get_orchestration_summary()
    
    print(f"\nOverall Performance:")
    print(f"  Total plugins executed: {summary['plugins_executed']}")
    print(f"  Successful executions: {summary['successful']}")
    print(f"  Early stops: {summary['early_stopped']}")
    print(f"  Average efficiency: {summary['average_efficiency']:.4f}")
    
    print(f"\nBudget Analysis:")
    budget = summary['budget_report']
    print(f"  Total budget: {budget['total_budget']:.1f} ATP")
    print(f"  Total consumed: {budget['total_consumed']:.1f} ATP")
    print(f"  Utilization: {budget['utilization']*100:.1f}%")
    
    print(f"\nPer-Plugin Performance:")
    for plugin_id, stats in summary['plugin_results'].items():
        print(f"\n  {plugin_id.upper()}:")
        print(f"    Execution time: {stats['execution_time']:.3f}s")
        print(f"    ATP consumed: {stats['atp_consumed']:.1f}")
        print(f"    Trust score: {stats['trust_score']:.3f}")
        print(f"    Efficiency: {stats['efficiency']:.4f}")
        print(f"    Compute saved: {stats.get('compute_saved', 0)*100:.1f}%")
    
    print(f"\nTrust Evolution:")
    for plugin_id, trust in orchestrator.budget.trust_weights.items():
        initial = 1.2 if plugin_id == "vision" else 0.8
        print(f"  {plugin_id}: {initial:.1f} → {trust:.3f} ({(trust/initial-1)*100:+.1f}%)")
    
    # Save detailed results
    with open('orchestrator_demo_results.json', 'w') as f:
        detailed_results = {
            "summary": summary,
            "test_cases": [
                {"name": "balanced", "results": [r.__dict__ for r in results1]},
                {"name": "heavy_vision", "results": [r.__dict__ for r in results2]},
                {"name": "multi_modal", "results": [r.__dict__ for r in results3]}
            ],
            "trust_evolution": orchestrator.budget.trust_weights,
            "final_budget_state": orchestrator.budget.get_efficiency_report()
        }
        # Convert tensors to lists for JSON serialization
        json_str = json.dumps(detailed_results, default=lambda x: x.tolist() if isinstance(x, torch.Tensor) else str(x), indent=2)
        f.write(json_str)
    
    print("\n✓ Detailed results saved to orchestrator_demo_results.json")
    print("=" * 70)


async def test_resource_contention():
    """Test what happens when plugins compete for limited resources"""
    
    print("\n" + "=" * 70)
    print("Resource Contention Test - Limited ATP Budget")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create orchestrator with very limited budget
    orchestrator = HRMOrchestrator(
        initial_atp=100.0,  # Very limited!
        max_concurrent=2,
        reallocation_interval=0.02  # Very fast reallocation
    )
    
    # Register plugins
    orchestrator.create_default_plugins()
    
    # Create expensive tasks
    print("\nCreating expensive tasks that will compete for ATP...")
    
    # Large batches to consume more ATP
    vision_data = torch.randn(16, 3, 224, 224).to(device)  # Large batch
    language_data = torch.randint(100, 5000, (16, 64)).to(device)  # Large sequence
    
    tasks = {
        "vision": vision_data,
        "language": language_data
    }
    
    print(f"  Vision: {vision_data.shape} (expensive)")
    print(f"  Language: {language_data.shape} (expensive)")
    print(f"  ATP Budget: {orchestrator.budget.total} (limited)")
    
    # Execute and see what happens
    start = time.time()
    results = await orchestrator.execute_parallel(tasks, early_stop=True)
    duration = time.time() - start
    
    print(f"\nExecution completed in {duration:.3f}s")
    
    # Analyze results
    for r in results:
        status = "✓" if r.state != PluginState.FAILED else "✗"
        print(f"\n{status} {r.plugin_id.upper()}:")
        print(f"    State: {r.state.value}")
        print(f"    ATP requested: {r.atp_consumed:.1f}")
        print(f"    ATP available: {orchestrator.budget.allocated.get(r.plugin_id, 0):.1f}")
        
        if r.state != PluginState.FAILED:
            print(f"    Trust score: {r.trust_score:.3f}")
            print(f"    Iterations: {r.telemetry.get('iterations', 'N/A')}")
    
    # Show how reallocation worked
    print(f"\nReallocation Analysis:")
    print(f"  Reallocation events: Check logs above")
    print(f"  Final budget utilization: {orchestrator.budget.get_efficiency_report()['utilization']*100:.1f}%")
    
    if any(r.state == PluginState.FAILED for r in results):
        print("\n⚠️  Some plugins failed due to ATP exhaustion!")
        print("This demonstrates the importance of:")
        print("  1. Proper ATP budgeting")
        print("  2. Early stopping to conserve resources")
        print("  3. Trust-based prioritization")


async def main():
    """Run all orchestration demos"""
    
    # Main demonstration
    await simulate_workload_variations()
    
    # Resource contention test
    await test_resource_contention()
    
    print("\n" + "=" * 70)
    print("All orchestration demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())