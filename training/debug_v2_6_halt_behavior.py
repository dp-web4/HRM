#!/usr/bin/env python3
"""
Debug V2.6 halt behavior - why exactly 4 cycles always?
"""
import torch
import numpy as np
from pathlib import Path
import json
import sys

# Import our model
sys.path.append('.')
from finetune_reasoning_fixed_v2 import HierarchicalReasoningModule, MODEL_CONFIG

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def debug_halt_behavior():
    """Deep dive into why V2.6 always uses 4 cycles"""
    
    print("🔍 DEBUGGING V2.6 HALT BEHAVIOR")
    print("Why exactly 4 cycles for every task?")
    print("=" * 60)
    
    # Load V2.6 model
    checkpoint_path = 'checkpoints/hrm_reasoning_agi-1_v2_6_final.pt'
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"✅ V2.6 model loaded")
    
    # Test on different input patterns (ensure integer type)
    test_cases = [
        ("Simple pattern", torch.zeros(1, 900, dtype=torch.long).to(DEVICE)),
        ("Random pattern", torch.randint(0, 10, (1, 900)).to(DEVICE)),
        ("Complex pattern", torch.randint(0, 10, (1, 900)).to(DEVICE)),
        ("All ones", torch.ones(1, 900, dtype=torch.long).to(DEVICE)),
        ("Mixed pattern", torch.cat([
            torch.zeros(1, 300, dtype=torch.long), 
            torch.ones(1, 300, dtype=torch.long), 
            torch.randint(0, 10, (1, 300))
        ], dim=1).to(DEVICE))
    ]
    
    print(f"\n🧪 Testing different input patterns:")
    print(f"{'Pattern':<15} {'Cycles':<8} {'Halt Progression'}")
    print("-" * 70)
    
    all_halt_progressions = []
    
    with torch.no_grad():
        for name, test_input in test_cases:
            # Modified forward pass to extract halt logits
            batch_size, seq_len = test_input.shape
            max_cycles = MODEL_CONFIG['max_cycles']
            
            # Embed input
            x_emb = model.token_embedding(test_input)
            x_emb = model.pos_encoding(x_emb)
            x_emb = model.dropout(x_emb)
            
            h_state = x_emb.clone()
            l_state = x_emb.clone()
            
            halt_probs = []
            halt_logits = []
            cycle_biases = []
            cumulative_halt = torch.zeros(batch_size, 1).to(test_input.device)
            
            for cycle in range(max_cycles):
                # Forward through layers (same as model)
                cycle_emb = model.cycle_embedding(torch.tensor([cycle], device=test_input.device))
                cycle_emb = cycle_emb.expand(batch_size, seq_len, -1)
                
                h_state = h_state + 0.1 * cycle_emb
                for h_layer in model.h_layers:
                    h_state = h_layer(h_state)
                h_state = model.h_norm(h_state)
                
                l_state = l_state + model.h_to_l(h_state)
                for l_layer in model.l_layers:
                    l_state = l_layer(l_state)
                l_state = model.l_norm(l_state)
                
                h_state = h_state + model.l_to_h(l_state)
                
                # Extract halt computation details
                combined = torch.cat([h_state.mean(dim=1), l_state.mean(dim=1)], dim=-1)
                halt_logit = model.halt_predictor(combined)
                
                # Apply bias (same as V2.3/V2.6)
                cycle_bias = -0.2 + (cycle / (max_cycles - 1)) * 1.1
                biased_logit = halt_logit + cycle_bias
                halt_prob = torch.sigmoid(biased_logit)
                
                halt_logits.append(halt_logit.item())
                cycle_biases.append(cycle_bias)
                halt_probs.append(halt_prob.item())
                
                # Check stopping conditions
                cumulative_halt = cumulative_halt + halt_prob
                
                if cycle >= 3:
                    if cumulative_halt.mean() > 1.0 or halt_prob.mean() > 0.95:
                        actual_cycles = cycle + 1
                        break
            else:
                actual_cycles = max_cycles
            
            halt_str = " → ".join([f"{p:.3f}" for p in halt_probs[:actual_cycles]])
            print(f"{name:<15} {actual_cycles:<8} {halt_str}")
            
            all_halt_progressions.append({
                'pattern': name,
                'cycles': actual_cycles,
                'halt_logits': halt_logits[:actual_cycles],
                'cycle_biases': cycle_biases[:actual_cycles],
                'halt_probs': halt_probs[:actual_cycles],
                'cumulative': cumulative_halt.item()
            })
    
    # Detailed analysis of the stopping mechanism
    print(f"\n🔬 DETAILED HALT ANALYSIS")
    print("=" * 60)
    
    for result in all_halt_progressions:
        print(f"\n{result['pattern']}:")
        print(f"  Cycles: {result['cycles']}")
        print(f"  Cumulative halt: {result['cumulative']:.3f}")
        
        print(f"  {'Cycle':<6} {'Logit':<8} {'Bias':<8} {'Final':<8} {'Prob':<8}")
        print(f"  {'-'*40}")
        
        for i in range(result['cycles']):
            logit = result['halt_logits'][i]
            bias = result['cycle_biases'][i]
            final = logit + bias
            prob = result['halt_probs'][i]
            
            print(f"  {i:<6} {logit:<8.3f} {bias:<8.3f} {final:<8.3f} {prob:<8.3f}")
        
        # Check stopping condition
        if result['cycles'] < MODEL_CONFIG['max_cycles']:
            final_prob = result['halt_probs'][-1]
            cumulative = result['cumulative']
            
            if final_prob > 0.95:
                print(f"  → STOPPED: Final halt prob {final_prob:.3f} > 0.95")
            elif cumulative > 1.0:
                print(f"  → STOPPED: Cumulative halt {cumulative:.3f} > 1.0")
            else:
                print(f"  → STOPPED: Unknown reason (final={final_prob:.3f}, cum={cumulative:.3f})")
    
    # Check if halt predictor learned a constant output
    print(f"\n🎯 HALT PREDICTOR ANALYSIS")
    print("=" * 60)
    
    # Test halt predictor on various inputs
    test_inputs = [
        torch.zeros(1, MODEL_CONFIG['hidden_size'] * 2).to(DEVICE),
        torch.ones(1, MODEL_CONFIG['hidden_size'] * 2).to(DEVICE),
        torch.randn(1, MODEL_CONFIG['hidden_size'] * 2).to(DEVICE),
        torch.randn(1, MODEL_CONFIG['hidden_size'] * 2).to(DEVICE) * 5,
    ]
    
    halt_outputs = []
    with torch.no_grad():
        for i, test_input in enumerate(test_inputs):
            halt_logit = model.halt_predictor(test_input)
            halt_outputs.append(halt_logit.item())
            print(f"  Input {i}: logit = {halt_logit.item():.3f}")
    
    # Check if predictor is "dead" (always same output)
    halt_variance = np.var(halt_outputs)
    print(f"\n  Halt logit variance: {halt_variance:.6f}")
    
    if halt_variance < 0.001:
        print("  ⚠️  DEAD PREDICTOR: Always outputs same value!")
    elif halt_variance < 0.1:
        print("  ⚠️  NEARLY DEAD: Very low variance")
    else:
        print("  ✅ ACTIVE PREDICTOR: Good variance")
    
    # Final diagnosis
    print(f"\n🎯 DIAGNOSIS")
    print("=" * 60)
    
    avg_logit = np.mean(halt_outputs)
    
    # Test stopping at cycle 3 (minimum)
    cycle_3_bias = -0.2 + (3 / 19) * 1.1  # bias at cycle 3
    cycle_3_final = avg_logit + cycle_3_bias
    cycle_3_prob = 1 / (1 + np.exp(-cycle_3_final))
    
    print(f"At cycle 3:")
    print(f"  Avg logit: {avg_logit:.3f}")
    print(f"  Cycle bias: {cycle_3_bias:.3f}")
    print(f"  Final logit: {cycle_3_final:.3f}")
    print(f"  Halt prob: {cycle_3_prob:.3f}")
    
    if cycle_3_prob > 0.95:
        print(f"  → Model ALWAYS stops at cycle 3 (prob > 0.95)")
    elif avg_logit > 2.0:
        print(f"  → Halt predictor learned to output HIGH positive values")
    elif avg_logit < -2.0:
        print(f"  → Halt predictor learned to output HIGH negative values")
    else:
        print(f"  → Halt predictor looks reasonable")
    
    return all_halt_progressions, halt_outputs

if __name__ == "__main__":
    debug_halt_behavior()