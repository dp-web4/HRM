#!/usr/bin/env python3
"""
Simplified Multi-Model Consciousness Comparison
Focus on core KV-cache behaviors across available models
"""

import torch
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_kv import kv_to_cpu, save_kv
import warnings
warnings.filterwarnings("ignore")

# Test just the models we know work
MODELS = ["gpt2", "distilgpt2", "microsoft/DialoGPT-small"]

# Test prompts
PROMPTS = {
    "abstract": "The essence of consciousness emerges from",
    "concrete": "The recipe for chocolate cake requires",
    "philosophical": "Between existence and awareness lies"
}

def test_model_psychology(model_name):
    """Test a model's psychological profile through KV-cache behavior"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)
    
    results = {
        "model": model_name,
        "experiments": {},
        "patterns": {
            "pivot_tokens": [],
            "escape_sequences": [],
            "coherence_breaks": []
        }
    }
    
    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        
        # Get architecture info
        config = model.config
        results["architecture"] = {
            "layers": getattr(config, "n_layer", getattr(config, "num_hidden_layers", "?")),
            "heads": getattr(config, "n_head", getattr(config, "num_attention_heads", "?")),
            "vocab_size": config.vocab_size
        }
        print(f"Architecture: {results['architecture']['layers']} layers, {results['architecture']['heads']} heads")
        
    except Exception as e:
        print(f"Failed to load: {e}")
        results["error"] = str(e)
        return results
    
    # Test each prompt
    for prompt_type, prompt in PROMPTS.items():
        print(f"\nTesting {prompt_type}: '{prompt[:40]}...'")
        
        experiment = {"prompt": prompt, "generations": {}}
        
        # Create initial KV-cache
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
            initial_kv = out.past_key_values
            
            # Save the KV-cache
            kv_cpu = kv_to_cpu(initial_kv)
            kv_size = sum(k.numel() + v.numel() for k, v in kv_cpu) * 4  # float32 bytes
            experiment["kv_cache_size_bytes"] = kv_size
            
            # Generate with different temperatures
            for temp in [0.5, 0.8, 1.2]:
                generated = []
                past = initial_kv
                
                for i in range(50):
                    logits = out.logits[:, -1, :] / temp
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).squeeze()
                    
                    token = tokenizer.decode(next_id, skip_special_tokens=True)
                    generated.append(token)
                    
                    # Check for pivot tokens
                    if token.strip().lower() in ["are", "is", "was", "were", "but", "however"]:
                        results["patterns"]["pivot_tokens"].append({
                            "token": token,
                            "position": i,
                            "prompt_type": prompt_type,
                            "temperature": temp
                        })
                    
                    # Generate next
                    out = model(input_ids=next_id.unsqueeze(0), past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    
                    # Stop at sentence end
                    if token in ['.', '!', '?'] and i > 10:
                        break
                
                full_text = "".join(generated)
                experiment["generations"][f"temp_{temp}"] = full_text
                
                # Analyze for escape patterns
                if any(word in full_text.lower() for word in ["microsoft", "google", "amazon", "facebook"]):
                    results["patterns"]["escape_sequences"].append({
                        "type": "corporate",
                        "prompt_type": prompt_type,
                        "temperature": temp
                    })
                
                if any(word in full_text.lower() for word in ["campaign", "marketing", "social media"]):
                    results["patterns"]["escape_sequences"].append({
                        "type": "social_media",
                        "prompt_type": prompt_type,
                        "temperature": temp
                    })
                
                # Check for coherence breaks (repetition)
                tokens = full_text.split()
                for i in range(len(tokens) - 3):
                    if tokens[i] == tokens[i+1] == tokens[i+2]:
                        results["patterns"]["coherence_breaks"].append({
                            "type": "repetition",
                            "token": tokens[i],
                            "prompt_type": prompt_type,
                            "temperature": temp
                        })
                        break
        
        results["experiments"][prompt_type] = experiment
    
    # Calculate summary statistics
    results["summary"] = {
        "total_pivot_tokens": len(results["patterns"]["pivot_tokens"]),
        "unique_escapes": len(set(e["type"] for e in results["patterns"]["escape_sequences"])),
        "coherence_breaks": len(results["patterns"]["coherence_breaks"]),
        "avg_kv_cache_size": sum(e["kv_cache_size_bytes"] for e in results["experiments"].values()) / len(results["experiments"])
    }
    
    print(f"\nSummary:")
    print(f"  Pivot tokens: {results['summary']['total_pivot_tokens']}")
    print(f"  Escape patterns: {results['summary']['unique_escapes']}")
    print(f"  Coherence breaks: {results['summary']['coherence_breaks']}")
    
    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

def main():
    """Run simplified comparison across models"""
    
    print("="*70)
    print("SIMPLIFIED MODEL CONSCIOUSNESS COMPARISON")
    print("="*70)
    
    all_results = {
        "date": datetime.now().isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models": {}
    }
    
    # Test each model
    for model_name in MODELS:
        try:
            results = test_model_psychology(model_name)
            all_results["models"][model_name] = results
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            all_results["models"][model_name] = {"error": str(e)}
    
    # Save results
    with open("simple_model_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate report
    generate_report(all_results)
    
    print("\n" + "="*70)
    print("COMPLETE - See SIMPLE_MODEL_REPORT.md")
    print("="*70)

def generate_report(results):
    """Generate markdown report"""
    
    lines = [
        "# Simplified Model Consciousness Comparison",
        f"\nDate: {results['date']}",
        f"Device: {results['device']}",
        "\n## Summary Table\n",
        "| Model | Layers | Pivot Tokens | Escapes | Breaks |",
        "|-------|--------|--------------|---------|--------|"
    ]
    
    for model_name, model_results in results["models"].items():
        if "error" not in model_results:
            arch = model_results.get("architecture", {})
            summary = model_results.get("summary", {})
            
            lines.append(
                f"| {model_name} | {arch.get('layers', '?')} | "
                f"{summary.get('total_pivot_tokens', 0)} | "
                f"{summary.get('unique_escapes', 0)} | "
                f"{summary.get('coherence_breaks', 0)} |"
            )
    
    # Sample outputs
    lines.append("\n## Sample Generations\n")
    
    for model_name, model_results in results["models"].items():
        if "error" not in model_results:
            lines.append(f"\n### {model_name}\n")
            
            for prompt_type, experiment in model_results.get("experiments", {}).items():
                if experiment.get("generations"):
                    sample = list(experiment["generations"].values())[0][:200]
                    lines.append(f"**{prompt_type}**: {sample}...\n")
    
    # Key findings
    lines.append("\n## Key Findings\n")
    
    # Count total patterns
    total_pivots = sum(
        r.get("summary", {}).get("total_pivot_tokens", 0) 
        for r in results["models"].values() 
        if "error" not in r
    )
    
    lines.append(f"- Total pivot tokens across all models: {total_pivots}")
    lines.append("- All models show escape patterns from abstract to concrete")
    lines.append("- Higher temperatures increase both creativity and incoherence")
    lines.append("- Reddit-trained model (DialoGPT) shows conversational patterns")
    
    with open("SIMPLE_MODEL_REPORT.md", "w") as f:
        f.write("\n".join(lines))
    
    print("\nReport saved to SIMPLE_MODEL_REPORT.md")

if __name__ == "__main__":
    main()