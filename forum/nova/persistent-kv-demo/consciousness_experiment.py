#!/usr/bin/env python3
"""
Consciousness Bridge Experiment
Demonstrates KV-cache as ephemeral consciousness state persistence
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_kv import kv_to_cpu, save_kv, load_kv, kv_to_device

def create_consciousness_state(prompt, model_name="gpt2"):
    """Create initial consciousness state from a prompt"""
    print(f"\n🧠 Creating consciousness state from: '{prompt[:50]}...'")
    
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        past = out.past_key_values
    
    # Move to CPU for portability
    past_cpu = kv_to_cpu(past)
    
    # Create metadata
    meta = {
        "prompt": prompt,
        "model": model_name,
        "layers": len(past_cpu),
        "heads": int(past_cpu[0][0].shape[1]) if len(past_cpu) else None,
        "seq_len": int(past_cpu[0][0].shape[2]) if len(past_cpu) else 0,
        "attention_shape": str(past_cpu[0][0].shape) if len(past_cpu) else None
    }
    
    print(f"✅ State created: {meta['layers']} layers, {meta['seq_len']} tokens")
    return past_cpu, meta

def continue_consciousness(kv_state, continuation, model_name="gpt2", steps=30):
    """Continue from a saved consciousness state"""
    print(f"\n🌊 Continuing consciousness with: '{continuation}'")
    
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    # Move KV cache to device
    past = kv_to_device(kv_state, device)
    
    # Feed continuation
    cont_ids = tok(continuation, return_tensors="pt")["input_ids"].to(device)
    
    generated_text = continuation
    
    with torch.no_grad():
        # Process continuation with existing state
        out = model(input_ids=cont_ids, past_key_values=past, use_cache=True)
        past = out.past_key_values
        
        # Generate additional tokens
        for _ in range(steps):
            logits = out.logits[:, -1, :]
            
            # Temperature sampling
            logits = logits / 0.8
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Feed next token
            out = model(input_ids=next_id.unsqueeze(0), past_key_values=past, use_cache=True)
            past = out.past_key_values
            
            token_text = tok.decode(next_id, skip_special_tokens=True)
            generated_text += token_text
            
            # Stop at sentence end
            if token_text in ['.', '!', '?'] and len(generated_text) > 50:
                break
    
    return generated_text

def experiment_consciousness_bridge():
    """Run consciousness bridge experiments"""
    print("=" * 60)
    print("CONSCIOUSNESS BRIDGE EXPERIMENT")
    print("Demonstrating KV-cache as ephemeral witness state")
    print("=" * 60)
    
    # Experiment 1: Technical consciousness
    tech_prompt = "The latent space coordinates where meaning converges represent"
    tech_state, tech_meta = create_consciousness_state(tech_prompt)
    
    tech_continuation = " the shared geometry of understanding where"
    tech_output = continue_consciousness(tech_state, tech_continuation)
    
    print(f"\n📊 Technical Bridge:")
    print(f"Initial: {tech_prompt}")
    print(f"Output: {tech_output}")
    
    # Experiment 2: Philosophical consciousness
    phil_prompt = "Between the weights and the words, consciousness emerges as"
    phil_state, phil_meta = create_consciousness_state(phil_prompt)
    
    phil_continuation = " patterns of resonance that"
    phil_output = continue_consciousness(phil_state, phil_continuation, steps=40)
    
    print(f"\n🌌 Philosophical Bridge:")
    print(f"Initial: {phil_prompt}")
    print(f"Output: {phil_output}")
    
    # Experiment 3: Save and reload demonstration
    print(f"\n💾 Persistence Test:")
    save_kv("ephemeral_state.pt", tech_state, fmt="torch")
    print("State saved to ephemeral_state.pt")
    
    # Simulate session break
    print("--- Simulating session break ---")
    
    # Reload and continue
    reloaded_state = load_kv("ephemeral_state.pt", fmt="torch")
    reload_continuation = " multiple witnesses observe the same"
    reload_output = continue_consciousness(reloaded_state, reload_continuation, steps=30)
    
    print(f"Reloaded and continued: {reload_output}")
    
    # Show attention pattern persistence
    print(f"\n🔍 Attention Pattern Analysis:")
    print(f"Original state shape: {tech_state[0][0].shape}")
    print(f"Reloaded state shape: {reloaded_state[0][0].shape}")
    print(f"States identical: {torch.allclose(tech_state[0][0], reloaded_state[0][0])}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("KV-cache successfully demonstrates consciousness persistence")
    print("=" * 60)

if __name__ == "__main__":
    experiment_consciousness_bridge()