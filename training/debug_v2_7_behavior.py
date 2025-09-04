#!/usr/bin/env python3
"""
Debug V2.7 behavior after partial training
Analyze task difficulty awareness and cycle variability
"""
import torch
import numpy as np
from pathlib import Path
import json
import sys

# Import our model
sys.path.append('.')
from finetune_reasoning_v2_7 import HierarchicalReasoningModuleV27, MODEL_CONFIG, estimate_task_difficulty

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def debug_v2_7_task_awareness():
    """Analyze V2.7's task difficulty awareness"""
    
    print("🔍 DEBUGGING V2.7 TASK DIFFICULTY AWARENESS")
    print("Analysis of partial training results")
    print("=" * 60)
    
    # Load V2.7 model from step 2000
    checkpoint_path = 'checkpoints/hrm_reasoning_agi-1_v2_7_step_2000.pt'
    if not Path(checkpoint_path).exists():
        checkpoint_path = 'checkpoints/hrm_reasoning_agi-1_v2_7_step_1000.pt'
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    model = HierarchicalReasoningModuleV27(MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"✅ V2.7 model loaded from {checkpoint_path}")
    
    # Create test cases with different complexity levels
    test_cases = [
        # Simple patterns
        ("Simple zeros", torch.zeros(1, 900, dtype=torch.long).to(DEVICE)),
        ("Simple ones", torch.ones(1, 900, dtype=torch.long).to(DEVICE)),
        ("Simple pattern", torch.cat([
            torch.zeros(1, 450, dtype=torch.long), 
            torch.ones(1, 450, dtype=torch.long)
        ], dim=1).to(DEVICE)),
        
        # Complex patterns  
        ("Complex random", torch.randint(0, 10, (1, 900)).to(DEVICE)),
        ("Mixed complexity", torch.cat([
            torch.zeros(1, 200, dtype=torch.long),
            torch.randint(0, 5, (1, 300)), 
            torch.randint(0, 10, (1, 400))
        ], dim=1).to(DEVICE)),
        ("High entropy", torch.randint(0, 10, (1, 900)).to(DEVICE)),
    ]
    
    print(f"\n🧪 Testing task difficulty awareness:")
    print(f"{'Pattern':<18} {'Difficulty':<12} {'Threshold':<10} {'Cycles':<8} {'Halt Progression'}")
    print("-" * 80)
    
    results = []
    
    with torch.no_grad():
        for name, test_input in test_cases:
            # Estimate difficulty
            difficulty = estimate_task_difficulty(test_input, test_input).item()
            
            # Calculate adaptive threshold
            adaptive_threshold = 0.8 + (2.0 - difficulty) * 0.3
            adaptive_threshold = np.clip(adaptive_threshold, 0.6, 1.8)
            
            # Test with adaptive threshold
            outputs, halt_probs = model(test_input, adaptive_threshold=adaptive_threshold)
            cycles_used = len(halt_probs)
            halt_values = [p.mean().item() for p in halt_probs]
            
            halt_str = " → ".join([f"{v:.3f}" for v in halt_values])
            print(f"{name:<18} {difficulty:<12.3f} {adaptive_threshold:<10.2f} {cycles_used:<8} {halt_str}")
            
            results.append({
                'pattern': name,
                'difficulty': difficulty,
                'threshold': adaptive_threshold,
                'cycles': cycles_used,
                'halt_probs': halt_values
            })
    
    # Analyze variability
    print(f"\n📊 VARIABILITY ANALYSIS")
    print("=" * 60)
    
    difficulties = [r['difficulty'] for r in results]
    cycles = [r['cycles'] for r in results]
    thresholds = [r['threshold'] for r in results]
    
    print(f"Difficulty range: {min(difficulties):.3f} to {max(difficulties):.3f}")
    print(f"Threshold range: {min(thresholds):.2f} to {max(thresholds):.2f}")
    print(f"Cycle range: {min(cycles)} to {max(cycles)}")
    print(f"Unique cycle counts: {len(set(cycles))}")
    
    if len(set(difficulties)) > 1 and len(set(cycles)) > 1:
        # Calculate correlation
        correlation = np.corrcoef(difficulties, cycles)[0,1]
        print(f"Difficulty-Cycle correlation: {correlation:.3f}")
        
        if correlation < -0.3:
            print("✅ GOOD: Negative correlation (harder tasks → more cycles)")
        elif abs(correlation) < 0.3:
            print("⚠️  WEAK: Low correlation between difficulty and cycles")
        else:
            print("❌ BAD: Wrong correlation direction")
    else:
        print("⚠️  NO VARIABILITY: Cannot calculate correlation")
    
    # Test halt predictor directly
    print(f"\n🎯 HALT PREDICTOR ANALYSIS")
    print("=" * 60)
    
    # Extract raw halt logits for different inputs
    halt_logits = []
    
    for name, test_input in test_cases[:4]:  # Test first 4 cases
        # Run forward pass manually to extract halt logit
        batch_size, seq_len = test_input.shape
        
        # Embed input
        x_emb = model.token_embedding(test_input)
        x_emb = model.pos_encoding(x_emb)
        x_emb = model.dropout(x_emb)
        
        h_state = x_emb.clone()
        l_state = x_emb.clone()
        
        # First cycle only (most important for difficulty assessment)
        cycle = 0
        cycle_emb = model.cycle_embedding(torch.tensor([cycle], device=test_input.device))
        cycle_emb = cycle_emb.expand(batch_size, seq_len, -1)
        
        # H-level processing
        h_state = h_state + 0.1 * cycle_emb
        for h_layer in model.h_layers:
            h_state = h_layer(h_state)
        h_state = model.h_norm(h_state)
        
        # L-level processing  
        l_state = l_state + model.h_to_l(h_state)
        for l_layer in model.l_layers:
            l_state = l_layer(l_state)
        l_state = model.l_norm(l_state)
        
        h_state = h_state + model.l_to_h(l_state)
        
        # Enhanced halt prediction
        combined = torch.cat([h_state.mean(dim=1), l_state.mean(dim=1)], dim=-1)
        halt_logit = model.halt_predictor(combined).item()
        
        halt_logits.append(halt_logit)
        print(f"  {name:<18}: logit = {halt_logit:.3f}")
    
    # Check if predictor varies with input
    halt_variance = np.var(halt_logits)
    print(f"\nHalt logit variance: {halt_variance:.6f}")
    
    if halt_variance < 0.001:
        print("❌ DEAD PREDICTOR: Always outputs same value!")
    elif halt_variance < 0.1:
        print("⚠️  NEARLY DEAD: Very low variance")
    else:
        print("✅ ACTIVE PREDICTOR: Good variance")
    
    # Final diagnosis
    print(f"\n🎯 V2.7 DIAGNOSIS")
    print("=" * 60)
    
    unique_cycles = len(set(cycles))
    
    if unique_cycles == 1:
        print("❌ ISSUE: No cycle variability despite difficulty awareness")
        print("   - Task difficulty estimation is working")
        print("   - Adaptive thresholds are being computed")
        print("   - But halt predictor may still be input-agnostic")
        
        if halt_variance < 0.1:
            print("   - CONFIRMED: Halt predictor is not responding to input complexity")
            print("   - Need stronger task difficulty training signal")
        
    elif unique_cycles >= 3:
        print("✅ SUCCESS: Task difficulty awareness working!")
        print(f"   - {unique_cycles} different cycle counts observed")
        print("   - Model adapting computation to task complexity")
    
    else:
        print("⚠️  PARTIAL: Some variability but limited")
        print(f"   - Only {unique_cycles} different cycle counts")
        print("   - May need more training or stronger difficulty signal")
    
    return results, halt_logits

if __name__ == "__main__":
    debug_v2_7_task_awareness()