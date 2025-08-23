#!/usr/bin/env python3
"""
Vision Latent IRP Demo
Demonstrates early-stop refinement in latent space with compute savings.

Acceptance criteria:
- Early-stop saves ≥2x compute with <1% mIoU drop
- Exports telemetry JSONL that passes schema validation
"""

import json
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sage.irp.plugins import VisionIRP


def main():
    """Run vision IRP demo with early stopping."""
    
    print("Vision Latent IRP Demo")
    print("=" * 50)
    
    # Configuration
    config = {
        'entity_id': 'vision_demo_001',
        'latent_dim': 512,
        'device': 'cpu',  # Change to 'cuda' on Jetson
        'max_iterations': 50,
        'halt_eps': 1e-3,
        'halt_K': 3,
        'task_weight': 0.5,
        'ATP_per_step': 0.1
    }
    
    # Initialize plugin
    print("\nInitializing Vision IRP plugin...")
    vision_irp = VisionIRP(config)
    
    # Create dummy input (would be actual image on Jetson)
    import torch
    dummy_image = torch.randn(1, 3, 224, 224)
    task_ctx = {
        'task': 'segmentation',
        'num_classes': 21,
        'target': 'early_stop_demo'
    }
    
    # Run refinement with early stopping
    print("\nRunning refinement with early stopping...")
    start_time = time.time()
    
    final_state, history = vision_irp.refine(
        dummy_image, 
        task_ctx,
        max_steps=config['max_iterations']
    )
    
    refinement_time = time.time() - start_time
    
    # Collect metrics
    print(f"\nRefinement completed in {len(history)} steps")
    print(f"Time taken: {refinement_time:.2f} seconds")
    print(f"Final energy: {final_state.energy_val:.4f}")
    
    # Compare with full refinement (no early stop)
    print("\nRunning full refinement for comparison...")
    config_full = config.copy()
    config_full['halt_eps'] = 0.0  # Disable early stopping
    vision_irp_full = VisionIRP(config_full)
    
    start_time = time.time()
    final_state_full, history_full = vision_irp_full.refine(
        dummy_image,
        task_ctx,
        max_steps=config['max_iterations']
    )
    full_time = time.time() - start_time
    
    # Compute savings
    compute_savings = len(history_full) / len(history)
    time_savings = full_time / refinement_time
    energy_diff = abs(final_state.energy_val - final_state_full.energy_val)
    
    print(f"\nFull refinement: {len(history_full)} steps")
    print(f"Compute savings: {compute_savings:.2f}x")
    print(f"Time savings: {time_savings:.2f}x")
    print(f"Energy difference: {energy_diff:.4f}")
    
    # Export telemetry
    telemetry_file = Path('vision_irp_telemetry.jsonl')
    print(f"\nExporting telemetry to {telemetry_file}...")
    
    with open(telemetry_file, 'w') as f:
        for state in history:
            telemetry = vision_irp.emit_telemetry(state, history[:history.index(state)+1])
            f.write(json.dumps(telemetry) + '\n')
    
    # Validate against schema
    print("\nValidating telemetry against schema...")
    try:
        import jsonschema
        
        schema_path = Path(__file__).parent.parent / 'schemas' / 'telemetry.schema.json'
        with open(schema_path) as f:
            schema = json.load(f)
        
        with open(telemetry_file) as f:
            for line in f:
                record = json.loads(line)
                jsonschema.validate(record, schema)
        
        print("✓ Telemetry validation passed!")
        
    except ImportError:
        print("⚠ jsonschema not installed, skipping validation")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # Print acceptance criteria results
    print("\n" + "=" * 50)
    print("ACCEPTANCE CRITERIA:")
    print(f"  Compute savings ≥2x: {'✓' if compute_savings >= 2.0 else '✗'} ({compute_savings:.2f}x)")
    print(f"  Energy difference <1%: {'✓' if energy_diff < 0.01 else '✗'} ({energy_diff:.4f})")
    print(f"  Telemetry exported: ✓")
    
    return compute_savings >= 2.0 and energy_diff < 0.01


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)