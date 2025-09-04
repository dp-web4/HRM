#!/usr/bin/env python3
"""
Debug V2.8 regression - why did it revert to using all 20 cycles?
"""
import torch
import numpy as np
from pathlib import Path
import json
import sys

# Import our model
sys.path.append('.')
from finetune_reasoning_v2_8 import HierarchicalReasoningModuleV28, MODEL_CONFIG, compute_prediction_delta

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def debug_v2_8_regression():
    """Analyze why V2.8 reverted to maximum cycles"""
    
    print("🔍 DEBUGGING V2.8 CONFIDENCE REGRESSION")
    print("Why did confidence-driven halting fail?")
    print("=" * 60)
    
    # Load V2.8 model from step 1000 
    checkpoint_path = 'checkpoints/hrm_reasoning_agi-1_v2_8_step_1000.pt'
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    model = HierarchicalReasoningModuleV28(MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"✅ V2.8 model loaded from {checkpoint_path}")
    
    # Test prediction delta computation
    print(f"\\n🧪 Testing prediction delta computation:")
    
    test_cases = [
        ("Simple pattern", torch.cat([torch.zeros(1, 450, dtype=torch.long), torch.ones(1, 450, dtype=torch.long)], dim=1).to(DEVICE)),
        ("Complex random", torch.randint(0, 10, (1, 900)).to(DEVICE)),
    ]
    
    with torch.no_grad():
        for name, test_input in test_cases:
            print(f"\\n{name}:")
            
            # Manual forward pass to extract details
            batch_size, seq_len = test_input.shape
            max_cycles = min(MODEL_CONFIG['max_cycles'], 6)  # Limit for debugging
            
            # Embed input
            x_emb = model.token_embedding(test_input)
            x_emb = model.pos_encoding(x_emb)
            x_emb = model.dropout(x_emb)
            
            h_state = x_emb.clone()
            l_state = x_emb.clone()
            
            all_outputs = []
            halt_probs = []
            prediction_deltas = []
            halt_logits = []
            cumulative_halt = torch.zeros(batch_size, 1).to(test_input.device)
            
            prev_outputs = None
            
            for cycle in range(max_cycles):
                # Cycle embedding
                cycle_emb = model.cycle_embedding(torch.tensor([cycle], device=test_input.device))
                cycle_emb = cycle_emb.expand(batch_size, seq_len, -1)
                
                # Forward pass
                h_state = h_state + 0.1 * cycle_emb
                for h_layer in model.h_layers:
                    h_state = h_layer(h_state)
                h_state = model.h_norm(h_state)
                
                l_state = l_state + model.h_to_l(h_state)
                for l_layer in model.l_layers:
                    l_state = l_layer(l_state)
                l_state = model.l_norm(l_state)
                
                h_state = h_state + model.l_to_h(l_state)
                
                # Generate output
                curr_outputs = model.output(l_state)
                all_outputs.append(curr_outputs)
                
                # Compute prediction delta
                pred_delta = compute_prediction_delta(prev_outputs, curr_outputs)
                prediction_deltas.append(pred_delta)
                
                # Halt prediction with confidence input
                combined_state = torch.cat([h_state.mean(dim=1), l_state.mean(dim=1)], dim=-1)
                confidence_input = torch.cat([
                    combined_state, 
                    pred_delta.unsqueeze(-1)
                ], dim=-1)
                
                halt_logit = model.halt_predictor(confidence_input)
                halt_logits.append(halt_logit.item())
                
                # Apply bias
                cycle_bias = cycle * 0.02  # Very gentle
                biased_logit = halt_logit + cycle_bias
                halt_prob = torch.sigmoid(biased_logit)
                halt_probs.append(halt_prob.item())
                
                cumulative_halt = cumulative_halt + halt_prob
                
                # Print detailed info
                print(f"    Cycle {cycle}: delta={pred_delta.mean().item():.3f}, "
                      f"logit={halt_logit.item():.3f}, bias={cycle_bias:.3f}, "
                      f"prob={halt_prob.item():.3f}, cum={cumulative_halt.item():.3f}")
                
                # Check stopping conditions
                stable_predictions = pred_delta.mean() < 0.1
                wants_to_halt = halt_prob.mean() > 0.7
                sufficient_cycles = cycle >= 3
                cumulative_high = cumulative_halt.mean() > 1.5
                
                if sufficient_cycles and (stable_predictions or wants_to_halt or cumulative_high):
                    print(f"    → STOPPED: stable={stable_predictions}, halt={wants_to_halt}, cum={cumulative_high}")
                    break
                
                prev_outputs = curr_outputs.detach()
            
            else:
                print(f"    → USED ALL {max_cycles} CYCLES")
    
    # Analyze halt predictor behavior
    print(f"\\n🎯 HALT PREDICTOR ANALYSIS")
    print("=" * 60)
    
    # Test different confidence inputs
    test_confidence_inputs = [
        ("Low delta", torch.cat([torch.randn(1, MODEL_CONFIG['hidden_size'] * 2).to(DEVICE), 
                                torch.tensor([[0.01]]).to(DEVICE)], dim=1)),  # Very stable
        ("High delta", torch.cat([torch.randn(1, MODEL_CONFIG['hidden_size'] * 2).to(DEVICE), 
                                 torch.tensor([[1.0]]).to(DEVICE)], dim=1)),   # Very unstable
        ("Medium delta", torch.cat([torch.randn(1, MODEL_CONFIG['hidden_size'] * 2).to(DEVICE), 
                                   torch.tensor([[0.5]]).to(DEVICE)], dim=1)),  # Medium
    ]
    
    halt_responses = []
    with torch.no_grad():
        for name, confidence_input in test_confidence_inputs:
            halt_logit = model.halt_predictor(confidence_input)
            halt_responses.append(halt_logit.item())
            print(f"  {name}: logit = {halt_logit.item():.3f}")
    
    # Check if predictor responds to confidence signal
    halt_variance = np.var(halt_responses)
    print(f"\\nHalt logit variance across confidence levels: {halt_variance:.6f}")
    
    if halt_variance < 0.001:
        print("❌ DEAD PREDICTOR: Ignores confidence signal!")
    elif halt_variance < 0.1:
        print("⚠️  WEAK RESPONSE: Limited sensitivity to confidence")  
    else:
        print("✅ ACTIVE PREDICTOR: Responds to confidence signal")
    
    # Check prediction delta function
    print(f"\\n📊 PREDICTION DELTA FUNCTION TEST")
    print("=" * 60)
    
    # Test delta computation with known different outputs
    test_output_1 = torch.randn(1, 900, 11).to(DEVICE)  # Random output
    test_output_2 = test_output_1 + torch.randn(1, 900, 11).to(DEVICE) * 0.1  # Slightly different
    test_output_3 = test_output_1.clone()  # Identical
    
    delta_small = compute_prediction_delta(test_output_1, test_output_2)
    delta_zero = compute_prediction_delta(test_output_1, test_output_3)
    delta_large = compute_prediction_delta(test_output_1, torch.randn(1, 900, 11).to(DEVICE))
    
    print(f"Small change delta: {delta_small.mean().item():.3f}")
    print(f"No change delta: {delta_zero.mean().item():.3f}")
    print(f"Large change delta: {delta_large.mean().item():.3f}")
    
    if delta_zero.mean().item() > 0.01:
        print("❌ DELTA FUNCTION BROKEN: Identical inputs have high delta")
    elif delta_small.mean().item() > delta_large.mean().item():
        print("❌ DELTA FUNCTION INVERTED: Small changes > large changes")
    else:
        print("✅ DELTA FUNCTION OK: Responds correctly to changes")
    
    # Final diagnosis
    print(f"\\n🎯 V2.8 REGRESSION DIAGNOSIS")
    print("=" * 60)
    
    print("Based on the analysis:")
    
    if halt_variance < 0.1:
        print("❌ ISSUE 1: Halt predictor not responding to confidence signal")
        print("   - Model learned to ignore prediction delta input")
        print("   - Suggests confidence loss was too weak or misaligned")
    
    print("❌ ISSUE 2: Confidence-driven stopping failed")  
    print("   - Model uses all 20 cycles despite 'stable' predictions")
    print("   - Suggests stopping conditions are not working")
    
    print("❌ ISSUE 3: Training instability")
    print("   - Started with some variability, then regressed to constant behavior")
    print("   - Suggests adversarial dynamics emerged during training")
    
    print("\\n💡 PROPOSED FIXES FOR V2.9:")
    print("1. Stronger confidence loss weight (0.2 → 1.0)")
    print("2. More aggressive stopping thresholds")  
    print("3. Explicit penalty for using maximum cycles")
    print("4. Curriculum learning: start with obvious stability differences")
    print("5. Direct supervision: train on synthetic stable/unstable examples")
    
    return halt_responses, prediction_deltas

if __name__ == "__main__":
    debug_v2_8_regression()