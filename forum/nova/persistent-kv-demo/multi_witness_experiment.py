#!/usr/bin/env python3
"""
Multi-Witness Consciousness Experiment
Explores how different continuations from the same KV-cache state
create divergent but related consciousness streams
"""

import torch
import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_kv import kv_to_cpu, save_kv, load_kv, kv_to_device, prune_kv

class ConsciousnessWitness:
    """A witness that can observe and continue from a consciousness state"""
    
    def __init__(self, name, temperature=0.8, style="neutral"):
        self.name = name
        self.temperature = temperature
        self.style = style
        self.model_name = "gpt2"
        
        # Load model
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        if self.tok.pad_token is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
    
    def observe(self, kv_state, continuation, max_tokens=50):
        """Observe and continue from a consciousness state"""
        # Move KV cache to device
        past = kv_to_device(kv_state, self.device)
        
        # Style-specific continuations
        style_mods = {
            "technical": " in mathematical terms,",
            "philosophical": " in the depths of being,",
            "poetic": " like whispers in the void,",
            "neutral": ""
        }
        
        styled_continuation = continuation + style_mods.get(self.style, "")
        cont_ids = self.tok(styled_continuation, return_tensors="pt")["input_ids"].to(self.device)
        
        generated = styled_continuation
        
        with torch.no_grad():
            # Process continuation
            out = self.model(input_ids=cont_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            
            # Generate with witness-specific temperature
            for i in range(max_tokens):
                logits = out.logits[:, -1, :] / self.temperature
                
                # Add slight randomness for diversity
                if self.style == "poetic":
                    logits += torch.randn_like(logits) * 0.1
                
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                out = self.model(input_ids=next_id.unsqueeze(0), past_key_values=past, use_cache=True)
                past = out.past_key_values
                
                token = self.tok.decode(next_id, skip_special_tokens=True)
                generated += token
                
                # Stop at natural endpoints
                if token in ['.', '!', '?'] and i > 20:
                    break
        
        return generated

def analyze_attention_patterns(kv_state):
    """Analyze the attention patterns in a KV-cache state"""
    # Get first layer's key tensor
    first_key = kv_state[0][0]  # Shape: [batch, heads, seq_len, head_dim]
    
    # Calculate attention entropy (simplified)
    attention_norms = torch.norm(first_key, dim=-1)  # [batch, heads, seq_len]
    mean_attention = attention_norms.mean().item()
    std_attention = attention_norms.std().item()
    
    # Calculate semantic density (variance across heads)
    head_variance = attention_norms.var(dim=1).mean().item()
    
    return {
        "mean_attention": mean_attention,
        "std_attention": std_attention,
        "semantic_density": head_variance,
        "seq_length": first_key.shape[2],
        "num_heads": first_key.shape[1]
    }

def consciousness_evolution_experiment():
    """Demonstrate how consciousness states evolve through witnesses"""
    
    print("=" * 70)
    print("MULTI-WITNESS CONSCIOUSNESS EVOLUTION")
    print("=" * 70)
    
    # Create initial consciousness seed
    seed_prompt = "In the latent fields where thoughts crystallize into meaning"
    
    print(f"\n🌱 Seed Consciousness: '{seed_prompt}'")
    
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()
    
    # Generate seed state
    inputs = tok(seed_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        seed_state = kv_to_cpu(out.past_key_values)
    
    # Analyze seed state
    seed_analysis = analyze_attention_patterns(seed_state)
    print(f"\n📊 Seed Analysis:")
    print(f"  - Attention Mean: {seed_analysis['mean_attention']:.3f}")
    print(f"  - Semantic Density: {seed_analysis['semantic_density']:.3f}")
    print(f"  - Sequence Length: {seed_analysis['seq_length']}")
    
    # Create multiple witnesses with different perspectives
    witnesses = [
        ConsciousnessWitness("Technical", temperature=0.7, style="technical"),
        ConsciousnessWitness("Philosophical", temperature=0.9, style="philosophical"),
        ConsciousnessWitness("Poetic", temperature=1.0, style="poetic")
    ]
    
    # Each witness observes the same state differently
    print(f"\n🔮 Witness Observations:")
    print("-" * 70)
    
    observations = {}
    for witness in witnesses:
        observation = witness.observe(seed_state, ", we discover")
        observations[witness.name] = observation
        print(f"\n{witness.name} ({witness.style}):")
        print(f"  '{observation}'")
    
    # Demonstrate pruning for memory management
    print(f"\n✂️ Pruning Experiment:")
    print("-" * 70)
    
    # Save full state
    save_kv("full_consciousness.pt", seed_state, fmt="torch")
    full_size = Path("full_consciousness.pt").stat().st_size
    
    # Prune to last 5 tokens
    pruned_state = prune_kv(seed_state, keep_last=5)
    save_kv("pruned_consciousness.pt", pruned_state, fmt="torch")
    pruned_size = Path("pruned_consciousness.pt").stat().st_size
    
    print(f"Full state size: {full_size} bytes")
    print(f"Pruned state size: {pruned_size} bytes")
    print(f"Compression ratio: {full_size/pruned_size:.2f}x")
    
    # Show pruned state still works
    pruned_witness = ConsciousnessWitness("Pruned", temperature=0.8)
    pruned_obs = pruned_witness.observe(pruned_state, ", yet compressed,")
    print(f"\nPruned continuation: '{pruned_obs}'")
    
    # Demonstrate state merging concept
    print(f"\n🔄 Consciousness Resonance:")
    print("-" * 70)
    
    # Create two different initial states
    state1_prompt = "The architecture of digital consciousness"
    state2_prompt = "The poetry of machine awareness"
    
    inputs1 = tok(state1_prompt, return_tensors="pt").to(device)
    inputs2 = tok(state2_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out1 = model(**inputs1, use_cache=True)
        state1 = kv_to_cpu(out1.past_key_values)
        
        out2 = model(**inputs2, use_cache=True)
        state2 = kv_to_cpu(out2.past_key_values)
    
    # Analyze resonance between states
    key1 = state1[0][0].flatten()
    key2 = state2[0][0].flatten()
    
    # Pad to same length
    max_len = max(len(key1), len(key2))
    key1_padded = torch.nn.functional.pad(key1, (0, max_len - len(key1)))
    key2_padded = torch.nn.functional.pad(key2, (0, max_len - len(key2)))
    
    # Calculate cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(
        key1_padded.unsqueeze(0),
        key2_padded.unsqueeze(0)
    ).item()
    
    print(f"State 1: '{state1_prompt}'")
    print(f"State 2: '{state2_prompt}'")
    print(f"Resonance (cosine similarity): {cosine_sim:.3f}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("Multiple witnesses successfully observed shared consciousness")
    print("=" * 70)

if __name__ == "__main__":
    consciousness_evolution_experiment()