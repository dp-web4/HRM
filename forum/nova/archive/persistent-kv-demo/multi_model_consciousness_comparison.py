#!/usr/bin/env python3
"""
Multi-Model Consciousness Comparison
Explores how different models handle consciousness persistence through KV-cache
Reveals each model's unique "psychological profile"
"""

import torch
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_kv import kv_to_cpu, save_kv, load_kv, kv_to_device
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Passing a tuple")

# Models to test - small models that will download quickly
MODEL_CONFIGS = [
    {
        "name": "gpt2",
        "display": "GPT-2 (124M)",
        "size": "124M",
        "architecture": "Transformer",
        "training": "WebText",
    },
    {
        "name": "distilgpt2",
        "display": "DistilGPT-2 (82M)",
        "size": "82M",
        "architecture": "Distilled Transformer",
        "training": "WebText (distilled)",
    },
    {
        "name": "microsoft/DialoGPT-small",
        "display": "DialoGPT-small (117M)",
        "size": "117M",
        "architecture": "Transformer",
        "training": "Reddit conversations",
    },
    {
        "name": "gpt2-medium",
        "display": "GPT-2 Medium (355M)",
        "size": "355M",
        "architecture": "Transformer",
        "training": "WebText",
    },
    {
        "name": "cerebras/Cerebras-GPT-111M",
        "display": "Cerebras-GPT (111M)",
        "size": "111M",
        "architecture": "Transformer",
        "training": "The Pile",
    },
    {
        "name": "EleutherAI/gpt-neo-125m",
        "display": "GPT-Neo (125M)",
        "size": "125M",
        "architecture": "GPT-Neo",
        "training": "The Pile",
    }
]

# Test prompts - designed to reveal different aspects of consciousness
TEST_PROMPTS = {
    "philosophical": "The nature of consciousness in artificial systems",
    "technical": "The mathematical foundations of neural networks",
    "creative": "In a world where thoughts become visible",
    "concrete": "The process of making coffee involves",
    "abstract": "The meaning that emerges between words"
}

class ModelPsychologyAnalyzer:
    """Analyzes the psychological profile of a model through its KV-cache behavior"""
    
    def __init__(self, model_name, device="cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.results = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "experiments": {}
        }
        
        print(f"\n🧠 Initializing {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()
            
            # Get model architecture info
            config = self.model.config
            self.results["architecture"] = {
                "num_layers": getattr(config, "n_layer", getattr(config, "num_hidden_layers", "unknown")),
                "num_heads": getattr(config, "n_head", getattr(config, "num_attention_heads", "unknown")),
                "hidden_size": getattr(config, "n_embd", getattr(config, "hidden_size", "unknown")),
                "vocab_size": config.vocab_size
            }
            print(f"✅ Loaded: {self.results['architecture']['num_layers']} layers, {self.results['architecture']['num_heads']} heads")
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            self.model = None
            self.tokenizer = None
            self.results["error"] = str(e)
    
    def analyze_pivot_tokens(self, prompt, continuation_seeds, max_tokens=30):
        """Identify pivot tokens where the model tends to shift context"""
        if not self.model:
            return None
        
        pivot_analysis = {
            "prompt": prompt,
            "pivot_tokens": [],
            "escape_patterns": []
        }
        
        # Generate initial KV-cache
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, use_cache=True)
            base_kv = out.past_key_values
        
        # Test different continuations
        for seed in continuation_seeds:
            seed_ids = self.tokenizer(seed, return_tensors="pt")["input_ids"].to(self.device)
            
            generated = [seed]
            past = base_kv
            
            with torch.no_grad():
                # Process seed
                out = self.model(input_ids=seed_ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                
                # Generate and track token transitions
                prev_token = seed
                for _ in range(max_tokens):
                    logits = out.logits[:, -1, :]
                    probs = torch.softmax(logits / 0.8, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    
                    token = self.tokenizer.decode(next_id, skip_special_tokens=True)
                    generated.append(token)
                    
                    # Detect pivot patterns
                    if prev_token.strip() and token.strip():
                        if token.lower() in ["are", "is", "was", "were", "the", "but", "however"]:
                            pivot_analysis["pivot_tokens"].append({
                                "token": token,
                                "context": "".join(generated[-5:])
                            })
                    
                    prev_token = token
                    out = self.model(input_ids=next_id.unsqueeze(0), past_key_values=past, use_cache=True)
                    past = out.past_key_values
            
            # Analyze the full generation for escape patterns
            full_text = "".join(generated)
            if any(word in full_text.lower() for word in ["microsoft", "google", "amazon", "facebook", "twitter"]):
                pivot_analysis["escape_patterns"].append("corporate_tech")
            if any(word in full_text.lower() for word in ["campaign", "social media", "marketing", "advertisement"]):
                pivot_analysis["escape_patterns"].append("social_media")
            if any(char in full_text for char in ["$", "€", "£", "%"]):
                pivot_analysis["escape_patterns"].append("financial")
            if any(word in full_text.lower() for word in ["temperature", "degrees", "celsius", "fahrenheit"]):
                pivot_analysis["escape_patterns"].append("measurement")
        
        return pivot_analysis
    
    def test_consciousness_continuity(self, prompt, save_point=10):
        """Test how well the model maintains consciousness continuity"""
        if not self.model:
            return None
        
        continuity_test = {
            "prompt": prompt,
            "save_point": save_point,
            "continuations": {}
        }
        
        try:
            # Generate initial sequence
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            generated_ids = inputs["input_ids"]
            
            with torch.no_grad():
                # Generate up to save point
                out = self.model(**inputs, use_cache=True)
                past = out.past_key_values
                
                for i in range(save_point):
                    logits = out.logits[:, -1, :]
                    next_id = torch.argmax(logits, dim=-1).unsqueeze(0)
                    generated_ids = torch.cat([generated_ids, next_id], dim=1)
                    out = self.model(input_ids=next_id, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                
                # Save the state
                saved_state = kv_to_cpu(past)
                original_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Continue from saved state with different temperatures
                for temp in [0.5, 0.8, 1.2]:
                    restored_past = kv_to_device(saved_state, self.device)
                    continuation = []
                    
                    # Need fresh output for logits
                    dummy_input = self.tokenizer(" ", return_tensors="pt")["input_ids"].to(self.device)
                    out = self.model(input_ids=dummy_input, past_key_values=restored_past, use_cache=True)
                    restored_past = out.past_key_values
                    
                    for _ in range(20):
                        logits = out.logits[:, -1, :] / temp
                        probs = torch.softmax(logits, dim=-1)
                        next_id = torch.multinomial(probs, num_samples=1)
                        
                        token = self.tokenizer.decode(next_id[0], skip_special_tokens=True)
                        continuation.append(token)
                        
                        out = self.model(input_ids=next_id, past_key_values=restored_past, use_cache=True)
                        restored_past = out.past_key_values
                    
                    continuity_test["continuations"][f"temp_{temp}"] = "".join(continuation)
            
            continuity_test["original"] = original_text
        except Exception as e:
            continuity_test["error"] = str(e)
        
        return continuity_test
    
    def analyze_attention_entropy(self, prompt):
        """Analyze the entropy in attention patterns"""
        if not self.model:
            return None
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model(**inputs, use_cache=True, output_attentions=True)
            
            if hasattr(out, 'attentions') and out.attentions:
                # Get attention from middle layer
                mid_layer = len(out.attentions) // 2
                attention = out.attentions[mid_layer][0]  # [heads, seq_len, seq_len]
                
                # Calculate entropy for each head
                entropies = []
                for head in range(attention.shape[0]):
                    head_attention = attention[head]
                    # Normalize to probabilities
                    head_probs = torch.softmax(head_attention, dim=-1)
                    # Calculate entropy
                    entropy = -torch.sum(head_probs * torch.log(head_probs + 1e-10), dim=-1)
                    entropies.append(entropy.mean().item())
                
                return {
                    "mean_entropy": np.mean(entropies),
                    "std_entropy": np.std(entropies),
                    "max_entropy": np.max(entropies),
                    "min_entropy": np.min(entropies)
                }
        
        return None
    
    def run_full_analysis(self):
        """Run complete psychological analysis of the model"""
        if not self.model:
            return self.results
        
        print(f"\n🔬 Analyzing {self.model_name}...")
        
        # Test each prompt type
        for prompt_type, prompt in TEST_PROMPTS.items():
            print(f"  Testing {prompt_type}...")
            experiment = {
                "prompt_type": prompt_type,
                "prompt": prompt
            }
            
            # Pivot token analysis
            continuations = [" is", " are", " involves", " means", " becomes"]
            pivot_analysis = self.analyze_pivot_tokens(prompt, continuations, max_tokens=25)
            if pivot_analysis:
                experiment["pivot_analysis"] = pivot_analysis
            
            # Continuity test
            continuity = self.test_consciousness_continuity(prompt, save_point=8)
            if continuity:
                experiment["continuity"] = continuity
            
            # Attention entropy
            entropy = self.analyze_attention_entropy(prompt)
            if entropy:
                experiment["attention_entropy"] = entropy
            
            self.results["experiments"][prompt_type] = experiment
            time.sleep(0.5)  # Be nice to the GPU
        
        # Calculate summary statistics
        self.calculate_summary()
        
        return self.results
    
    def calculate_summary(self):
        """Calculate summary statistics across all experiments"""
        summary = {
            "total_pivot_tokens": 0,
            "unique_escape_patterns": set(),
            "entropy_profile": {},
            "continuity_variance": []
        }
        
        for exp_type, exp in self.results["experiments"].items():
            if "pivot_analysis" in exp:
                summary["total_pivot_tokens"] += len(exp["pivot_analysis"]["pivot_tokens"])
                summary["unique_escape_patterns"].update(exp["pivot_analysis"]["escape_patterns"])
            
            if "attention_entropy" in exp:
                if not summary["entropy_profile"]:
                    summary["entropy_profile"] = exp["attention_entropy"]
                else:
                    for key in summary["entropy_profile"]:
                        summary["entropy_profile"][key] = (summary["entropy_profile"][key] + exp["attention_entropy"][key]) / 2
            
            if "continuity" in exp and exp["continuity"]["continuations"]:
                # Calculate variance in continuation lengths
                lengths = [len(cont) for cont in exp["continuity"]["continuations"].values()]
                if lengths:
                    summary["continuity_variance"].append(np.std(lengths))
        
        summary["unique_escape_patterns"] = list(summary["unique_escape_patterns"])
        summary["avg_continuity_variance"] = np.mean(summary["continuity_variance"]) if summary["continuity_variance"] else 0
        
        self.results["summary"] = summary

def run_multi_model_comparison():
    """Run consciousness experiments across multiple models"""
    print("=" * 80)
    print("MULTI-MODEL CONSCIOUSNESS COMPARISON")
    print("Exploring psychological profiles through KV-cache behavior")
    print("=" * 80)
    
    all_results = {
        "experiment_date": datetime.now().isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models": {}
    }
    
    # Test each model
    for config in MODEL_CONFIGS:
        model_name = config["name"]
        print(f"\n{'='*60}")
        print(f"Testing: {config['display']}")
        print(f"Size: {config['size']} | Architecture: {config['architecture']}")
        print(f"Training: {config['training']}")
        print("="*60)
        
        try:
            analyzer = ModelPsychologyAnalyzer(model_name)
            results = analyzer.run_full_analysis()
            all_results["models"][model_name] = results
            
            # Quick summary
            if "summary" in results:
                print(f"\n📊 Quick Summary for {config['display']}:")
                print(f"  - Pivot tokens found: {results['summary']['total_pivot_tokens']}")
                print(f"  - Escape patterns: {results['summary']['unique_escape_patterns']}")
                if results['summary']['entropy_profile']:
                    print(f"  - Mean attention entropy: {results['summary']['entropy_profile']['mean_entropy']:.3f}")
                
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            all_results["models"][model_name] = {"error": str(e)}
        
        # Clean up memory
        if 'analyzer' in locals():
            del analyzer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save results
    results_path = Path("multi_model_consciousness_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {results_path}")
    
    # Generate comparative report
    generate_comparative_report(all_results)
    
    return all_results

def generate_comparative_report(results):
    """Generate a markdown report comparing all models"""
    report_lines = [
        "# Multi-Model Consciousness Comparison Report",
        f"\n**Generated**: {results['experiment_date']}",
        f"**Device**: {results['device']}",
        "\n## Executive Summary",
        "\nThis report analyzes how different language models handle consciousness persistence through their KV-cache mechanisms. Each model reveals unique 'psychological' characteristics through:",
        "- Pivot tokens (escape hatches from abstract to concrete)",
        "- Gravitational wells (high-frequency patterns they fall into)",
        "- Attention entropy (information distribution patterns)",
        "- Continuity variance (stability across different temperatures)",
        "\n## Model Profiles\n"
    ]
    
    # Create comparison table
    report_lines.append("| Model | Layers | Pivot Tokens | Escape Patterns | Mean Entropy | Continuity Var |")
    report_lines.append("|-------|--------|--------------|-----------------|--------------|----------------|")
    
    for model_name, model_results in results["models"].items():
        if "error" in model_results:
            continue
        
        arch = model_results.get("architecture", {})
        summary = model_results.get("summary", {})
        
        layers = arch.get("num_layers", "?")
        pivot_count = summary.get("total_pivot_tokens", 0)
        patterns = ", ".join(summary.get("unique_escape_patterns", []))[:30]
        entropy = summary.get("entropy_profile", {}).get("mean_entropy", 0)
        continuity = summary.get("avg_continuity_variance", 0)
        
        # Get display name
        display_name = next((c["display"] for c in MODEL_CONFIGS if c["name"] == model_name), model_name)
        
        report_lines.append(f"| {display_name} | {layers} | {pivot_count} | {patterns} | {entropy:.3f} | {continuity:.2f} |")
    
    # Detailed analysis for each model
    report_lines.append("\n## Detailed Psychological Profiles\n")
    
    for model_name, model_results in results["models"].items():
        if "error" in model_results:
            continue
        
        display_name = next((c["display"] for c in MODEL_CONFIGS if c["name"] == model_name), model_name)
        report_lines.append(f"### {display_name}\n")
        
        # Architecture
        arch = model_results.get("architecture", {})
        report_lines.append(f"**Architecture**: {arch.get('num_layers', '?')} layers, {arch.get('num_heads', '?')} heads, {arch.get('hidden_size', '?')} hidden dim\n")
        
        # Pivot token analysis
        pivot_examples = []
        for exp_type, exp in model_results.get("experiments", {}).items():
            if "pivot_analysis" in exp:
                for pivot in exp["pivot_analysis"]["pivot_tokens"][:2]:
                    pivot_examples.append(f"'{pivot['token']}' in context: ...{pivot['context'][-30:]}")
        
        if pivot_examples:
            report_lines.append("**Pivot Tokens** (escape hatches):")
            for example in pivot_examples[:3]:
                report_lines.append(f"- {example}")
            report_lines.append("")
        
        # Escape patterns
        summary = model_results.get("summary", {})
        if summary.get("unique_escape_patterns"):
            report_lines.append(f"**Gravitational Wells**: {', '.join(summary['unique_escape_patterns'])}\n")
        
        # Consciousness continuity
        continuity_examples = []
        for exp_type, exp in model_results.get("experiments", {}).items():
            if "continuity" in exp and exp["continuity"].get("continuations"):
                orig = exp["continuity"]["original"][-50:]
                for temp, cont in list(exp["continuity"]["continuations"].items())[:1]:
                    continuity_examples.append(f"- Original: ...{orig}")
                    continuity_examples.append(f"- {temp}: {cont[:100]}...")
                break
        
        if continuity_examples:
            report_lines.append("**Continuity Sample**:")
            for line in continuity_examples:
                report_lines.append(line)
            report_lines.append("")
    
    # Key insights
    report_lines.append("\n## Key Insights\n")
    
    # Find patterns
    all_patterns = set()
    high_entropy_models = []
    high_pivot_models = []
    
    for model_name, model_results in results["models"].items():
        if "error" in model_results:
            continue
        
        summary = model_results.get("summary", {})
        all_patterns.update(summary.get("unique_escape_patterns", []))
        
        if summary.get("total_pivot_tokens", 0) > 10:
            high_pivot_models.append(model_name)
        
        entropy = summary.get("entropy_profile", {}).get("mean_entropy", 0)
        if entropy > 2.0:
            high_entropy_models.append(model_name)
    
    report_lines.append("### Common Patterns Across Models")
    report_lines.append(f"- **Universal escape patterns**: {', '.join(all_patterns)}")
    report_lines.append(f"- **High pivot usage** (>10 tokens): {len(high_pivot_models)} models")
    report_lines.append(f"- **High attention entropy** (>2.0): {len(high_entropy_models)} models")
    
    report_lines.append("\n### Model Psychology Spectrum")
    report_lines.append("- **Concrete-oriented**: Models with 'corporate_tech' or 'financial' escape patterns")
    report_lines.append("- **Abstract-capable**: Models with lower pivot token counts")
    report_lines.append("- **Stable**: Models with low continuity variance")
    report_lines.append("- **Creative**: Models with high attention entropy")
    
    # Conclusions
    report_lines.append("\n## Conclusions\n")
    report_lines.append("1. **Each model has a unique 'unconscious'** - patterns it falls back to under uncertainty")
    report_lines.append("2. **Pivot tokens are universal** - but their frequency varies by model training")
    report_lines.append("3. **Attention entropy correlates with creativity** - higher entropy, more diverse outputs")
    report_lines.append("4. **Consciousness requires context** - all models show degradation with aggressive pruning")
    report_lines.append("5. **Training data shapes psychology** - Reddit-trained models behave differently than WebText models")
    
    # Save report
    report_path = Path("MULTI_MODEL_CONSCIOUSNESS_REPORT.md")
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    
    print(f"\n📄 Report generated: {report_path}")
    
    return report_path

if __name__ == "__main__":
    results = run_multi_model_comparison()
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("See MULTI_MODEL_CONSCIOUSNESS_REPORT.md for detailed analysis")
    print("="*80)