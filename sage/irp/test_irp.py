#!/usr/bin/env python3
"""
Test script for IRP plugins and orchestrator
Version: 1.0 (2025-08-23)

Demonstrates the IRP framework with early stopping benchmarks.
"""

import numpy as np
import torch
import time
import json
from typing import Dict, Any

# Import IRP components
from base import IRPPlugin, IRPState
from vision import VisionIRP
from language import LanguageIRP
from control import ControlIRP
from memory import MemoryIRP
from orchestrator import HRMOrchestrator


def test_vision_irp():
    """Test Vision IRP with early stopping."""
    print("\n" + "="*60)
    print("Testing Vision IRP Plugin")
    print("="*60)
    
    # Configure plugin
    config = {
        'latent_dim': 128,
        'max_iterations': 50,
        'halt_eps': 1e-4,
        'halt_K': 3,
        'confidence_threshold': 0.9,
        'device': 'cpu'  # Use CPU for testing
    }
    
    vision = VisionIRP(config)
    
    # Create dummy input image (3x224x224)
    image = torch.randn(3, 224, 224)
    task_ctx = {'target_level': 'objects'}
    
    # Run refinement
    start_time = time.time()
    final_state, history = vision.refine(image, task_ctx)
    elapsed = time.time() - start_time
    
    # Get results
    results = vision.get_semantic_representation(final_state)
    
    print(f"Refinement completed in {len(history)} steps")
    print(f"Time elapsed: {elapsed:.3f}s")
    print(f"Final confidence: {results['confidence']:.3f}")
    print(f"Refinement level reached: {results['level']}")
    print(f"Early stopped: {len(history) < config['max_iterations']}")
    
    # Check convergence
    energies = [s.energy_val for s in history if s.energy_val is not None]
    if len(energies) > 1:
        energy_reduction = (energies[0] - energies[-1]) / abs(energies[0]) * 100
        print(f"Energy reduction: {energy_reduction:.1f}%")
    
    return results


def test_language_irp():
    """Test Language IRP with masked denoising."""
    print("\n" + "="*60)
    print("Testing Language IRP Plugin")
    print("="*60)
    
    # Configure plugin
    config = {
        'vocab_size': 1000,
        'hidden_dim': 256,
        'max_seq_len': 128,
        'meaning_dim': 64,
        'max_iterations': 30,
        'halt_eps': 1e-4,
        'device': 'cpu'
    }
    
    language = LanguageIRP(config)
    
    # Create dummy text input
    text = "The quick brown fox jumps over the lazy dog"
    task_ctx = {'mode': 'understand'}
    
    # Run refinement
    start_time = time.time()
    final_state, history = language.refine(text, task_ctx)
    elapsed = time.time() - start_time
    
    # Get results
    results = language.get_understanding(final_state)
    
    print(f"Refinement completed in {len(history)} steps")
    print(f"Time elapsed: {elapsed:.3f}s")
    print(f"Final perplexity: {results['final_perplexity']:.3f}")
    print(f"Refinement level: {results['refinement_level']}")
    print(f"Masks remaining: {results['masks_remaining']}")
    print(f"Early stopped: {len(history) < config['max_iterations']}")
    
    return results


def test_control_irp():
    """Test Control IRP with trajectory planning."""
    print("\n" + "="*60)
    print("Testing Control IRP Plugin")
    print("="*60)
    
    # Configure plugin
    config = {
        'state_dim': 4,
        'action_dim': 2,
        'horizon': 20,
        'dt': 0.1,
        'max_iterations': 50,
        'halt_eps': 1e-3,
        'feasibility_margin': 0.1,
        'device': 'cpu'
    }
    
    control = ControlIRP(config)
    
    # Define start and goal
    start = np.array([0.0, 0.0, 0.0, 0.0])  # [x, y, vx, vy]
    goal = np.array([5.0, 5.0, 0.0, 0.0])
    
    x0 = {'start': start, 'goal': goal}
    task_ctx = {'avoid_obstacles': True}
    
    # Run refinement
    start_time = time.time()
    final_state, history = control.refine(x0, task_ctx)
    elapsed = time.time() - start_time
    
    # Get results
    results = control.get_trajectory(final_state)
    
    print(f"Refinement completed in {len(history)} steps")
    print(f"Time elapsed: {elapsed:.3f}s")
    print(f"Trajectory feasible: {results['is_feasible']}")
    print(f"Final cost: {results['final_cost']:.3f}")
    print(f"Terminal error: {results['terminal_error']:.3f}")
    print(f"Early stopped: {len(history) < config['max_iterations']}")
    
    return results


def test_memory_irp():
    """Test Memory IRP with consolidation."""
    print("\n" + "="*60)
    print("Testing Memory IRP Plugin")
    print("="*60)
    
    # Configure plugin
    config = {
        'memory_dim': 128,
        'max_iterations': 40,
        'consolidation_rate': 0.15,
        'db_path': 'test_memory.db',
        'device': 'cpu'
    }
    
    memory = MemoryIRP(config)
    
    # Create dummy experiences
    experiences = [
        {'embedding': np.random.randn(128), 'timestamp': i}
        for i in range(10)
    ]
    
    task_ctx = {'consolidation_goal': 'extract_patterns'}
    
    # Run consolidation
    start_time = time.time()
    final_state, history = memory.refine(experiences, task_ctx)
    elapsed = time.time() - start_time
    
    # Get results
    results = memory.get_consolidated_memory(final_state)
    
    print(f"Consolidation completed in {len(history)} steps")
    print(f"Time elapsed: {elapsed:.3f}s")
    print(f"Final abstraction level: {results['final_level']}")
    print(f"Compression achieved: {results['compression_achieved']:.2f}x")
    print(f"Retrieval accuracy: {results['retrieval_accuracy']:.3f}")
    print(f"Value created: {results['value_created']:.3f}")
    print(f"Early stopped: {len(history) < config['max_iterations']}")
    
    return results


def test_orchestrator():
    """Test HRM Orchestrator with multiple plugins."""
    print("\n" + "="*60)
    print("Testing HRM Orchestrator")
    print("="*60)
    
    # Configure orchestrator
    config = {
        'total_ATP': 50.0,
        'max_workers': 4,
        'trust_update_rate': 0.1,
        'enable_vision': True,
        'enable_language': True,
        'enable_control': True,
        'enable_memory': False,  # Skip memory for quick test
        'vision_config': {
            'latent_dim': 64,
            'max_iterations': 20
        },
        'language_config': {
            'hidden_dim': 128,
            'max_iterations': 20
        },
        'control_config': {
            'horizon': 10,
            'max_iterations': 20
        }
    }
    
    orchestrator = HRMOrchestrator(config)
    
    # Prepare inputs for each plugin
    inputs = {
        'vision': torch.randn(3, 224, 224),
        'language': "Test input text for processing",
        'control': {
            'start': np.array([0.0, 0.0, 0.0, 0.0]),
            'goal': np.array([3.0, 3.0, 0.0, 0.0])
        }
    }
    
    # Run orchestrated processing
    print("\nRunning asynchronous orchestration...")
    start_time = time.time()
    results = orchestrator.process(inputs)
    elapsed = time.time() - start_time
    
    print(f"\nOrchestration completed in {elapsed:.3f}s")
    print(f"System coherence: {results['system_coherence']:.3f}")
    print(f"Total ATP used: {results['total_ATP_used']:.1f}")
    
    print("\nPlugin execution times:")
    for name, exec_time in results['execution_times'].items():
        print(f"  {name}: {exec_time:.3f}s")
    
    print("\nTrust weights:")
    for name, weight in results['trust_weights'].items():
        print(f"  {name}: {weight:.3f}")
    
    # Get telemetry summary
    summary = orchestrator.get_telemetry_summary()
    if summary:
        print(f"\nAverage ATP per run: {summary['average_ATP_per_run']:.2f}")
        print(f"Average convergence steps: {summary['average_convergence_steps']:.1f}")
    
    return results


def benchmark_early_stopping():
    """Benchmark early stopping across different configurations."""
    print("\n" + "="*60)
    print("Benchmarking Early Stopping")
    print("="*60)
    
    # Test different halt_eps values
    eps_values = [1e-2, 1e-3, 1e-4, 1e-5]
    
    results = []
    
    for eps in eps_values:
        config = {
            'latent_dim': 64,
            'max_iterations': 100,
            'halt_eps': eps,
            'halt_K': 3,
            'confidence_threshold': 0.95,
            'device': 'cpu'
        }
        
        vision = VisionIRP(config)
        image = torch.randn(3, 224, 224)
        
        start_time = time.time()
        final_state, history = vision.refine(image, {})
        elapsed = time.time() - start_time
        
        result = {
            'eps': eps,
            'steps': len(history),
            'time': elapsed,
            'final_energy': history[-1].energy_val if history else float('inf')
        }
        results.append(result)
        
        print(f"eps={eps}: {result['steps']} steps, {result['time']:.3f}s, energy={result['final_energy']:.4f}")
    
    # Analyze results
    print("\nEarly Stopping Analysis:")
    baseline = results[0]
    for result in results[1:]:
        speedup = baseline['time'] / result['time']
        step_reduction = (baseline['steps'] - result['steps']) / baseline['steps'] * 100
        print(f"eps={result['eps']}: {speedup:.2f}x speedup, {step_reduction:.1f}% fewer steps")
    
    return results


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("IRP Framework Test Suite")
    print("="*60)
    
    # Test individual plugins
    vision_results = test_vision_irp()
    language_results = test_language_irp()
    control_results = test_control_irp()
    memory_results = test_memory_irp()
    
    # Test orchestrator
    orchestrator_results = test_orchestrator()
    
    # Run benchmarks
    benchmark_results = benchmark_early_stopping()
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)
    
    # Save results
    all_results = {
        'vision': vision_results,
        'language': language_results,
        'control': control_results,
        'memory': memory_results,
        'orchestrator': orchestrator_results,
        'benchmark': benchmark_results
    }
    
    with open('test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\nResults saved to test_results.json")


if __name__ == "__main__":
    main()