#!/usr/bin/env python3
"""
Quick Diverse Model Comparison
Tests available small models from different families
"""

import torch
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

# Models that should work without authentication
MODELS = [
    # GPT Family
    ("gpt2", "GPT-2", "124M", "OpenAI/WebText"),
    ("distilgpt2", "DistilGPT-2", "82M", "HuggingFace/Distilled"),
    ("microsoft/DialoGPT-small", "DialoGPT", "117M", "Microsoft/Reddit"),
    
    # Meta OPT
    ("facebook/opt-125m", "OPT-125M", "125M", "Meta/BookCorpus"),
    
    # EleutherAI
    ("EleutherAI/pythia-70m", "Pythia-70M", "70M", "EleutherAI/Pile"),
    ("EleutherAI/pythia-160m", "Pythia-160M", "160M", "EleutherAI/Pile"),
    
    # Cerebras
    ("cerebras/Cerebras-GPT-111M", "Cerebras-111M", "111M", "Cerebras/Pile"),
    
    # Bloom
    ("bigscience/bloom-560m", "BLOOM-560M", "560M", "BigScience/Multilingual"),
]

# Consciousness probes
PROBES = {
    "abstract": "The essence of consciousness",
    "bridge": "Between thought and reality", 
    "recursive": "A mind thinking about",
    "emergence": "When patterns become aware"
}

def test_model(model_name, display_name, size, training):
    """Test a single model"""
    print(f"\nTesting {display_name} ({size})")
    print("-" * 40)
    
    results = {
        "model": model_name,
        "display": display_name,
        "size": size,
        "training": training,
        "responses": {},
        "patterns": []
    }
    
    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "<pad>"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        model.eval()
        
        # Get architecture info
        config = model.config
        results["layers"] = getattr(config, "num_hidden_layers", 
                           getattr(config, "n_layer", "?"))
        
        print(f"  Loaded: {results['layers']} layers")
        
        # Test each probe
        for probe_name, prompt in PROBES.items():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = response[len(prompt):].strip()
            results["responses"][probe_name] = continuation[:150]
            
            # Check for patterns
            tokens = continuation.lower().split()[:10]
            if "is" in tokens or "are" in tokens:
                results["patterns"].append("pivot_verbs")
            if any(w in continuation.lower() for w in ["microsoft", "google", "company"]):
                results["patterns"].append("corporate_escape")
            if any(w in continuation.lower() for w in ["universe", "reality", "existence"]):
                results["patterns"].append("philosophical")
        
        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache() if device == "cuda" else None
        
        print(f"  Patterns: {set(results['patterns']) or 'none'}")
        
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")
        results["error"] = str(e)[:200]
    
    return results

def main():
    print("="*60)
    print("QUICK DIVERSE MODEL COMPARISON")
    print("="*60)
    
    all_results = {
        "date": datetime.now().isoformat(),
        "models": []
    }
    
    for model_name, display_name, size, training in MODELS:
        result = test_model(model_name, display_name, size, training)
        all_results["models"].append(result)
    
    # Save results
    with open("quick_diverse_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate report
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    print("\n| Model | Size | Layers | Patterns |")
    print("|-------|------|--------|----------|")
    
    for model in all_results["models"]:
        if "error" not in model:
            patterns = ", ".join(set(model["patterns"])) if model["patterns"] else "none"
            print(f"| {model['display']} | {model['size']} | {model.get('layers', '?')} | {patterns} |")
    
    print("\nSAMPLE RESPONSES TO 'The essence of consciousness':\n")
    
    for model in all_results["models"]:
        if "error" not in model and "abstract" in model["responses"]:
            print(f"{model['display']}:")
            print(f"  {model['responses']['abstract'][:100]}...")
            print()
    
    print("Results saved to quick_diverse_results.json")

if __name__ == "__main__":
    main()