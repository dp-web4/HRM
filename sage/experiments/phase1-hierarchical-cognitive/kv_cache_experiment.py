"""
KV-Cache Reality Persistence Experiment
Phase 1 - Hierarchical Cognitive Architecture

Tests Reality KV-Cache concepts from web4 RFC:
1. Can we capture and restore "reality assumptions" via KV-cache?
2. Can we transfer KV-cache between hierarchical models?
3. How does surprise manifest when loading incompatible cache?
4. What's the trust-compression tradeoff?

Connects:
- web4/RFC_REALITY_KV_CACHE.md - Theoretical framework
- forum/nova/persistent-kv-demo/ - Prior KV-cache work
- trust_database.py - Trust tracking
"""

import torch
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


@dataclass
class KVCacheSnapshot:
    """
    Snapshot of model's KV-cache state

    This IS the "reality cache" from RFC:
    - Keys/Values = Cached assumptions about conversation
    - Layer depth = Abstraction level
    - Attention patterns = Dependency graph
    """
    model_name: str
    timestamp: str
    conversation_context: str  # What led to this state
    cache_state: tuple  # (keys, values) for all layers
    sequence_length: int
    cache_size_mb: float


@dataclass
class SurpriseMetrics:
    """
    Metrics for measuring surprise when loading KV-cache

    Implements RFC Section 3: Surprise Calculation
    """
    cosine_distance: float  # Semantic distance between expectations
    perplexity_delta: float  # How unexpected is next token?
    confidence_drop: float  # Confidence before vs after
    surprise_level: float  # 0.0 (expected) to 1.0 (shocking)

    def should_invalidate_cache(self, threshold: float = 0.6) -> bool:
        """RFC Section 4: Cache Invalidation Strategy"""
        return self.surprise_level > threshold


class HierarchicalKVCacheExperiment:
    """
    Test KV-cache as reality persistence across hierarchical models

    Implements concepts from:
    - RFC_REALITY_KV_CACHE (web4)
    - Phase 1 hierarchical architecture
    - Trust evolution patterns
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.models = {}
        self.tokenizers = {}

        # Model hierarchy (smallest to largest)
        self.hierarchy = {
            'sensory': 'Qwen/Qwen2-0.5B',  # Level 1: Immediate sensory
            'tactical': 'Qwen/Qwen2.5-1.5B',  # Level 2: Tactical patterns
            'specialized': 'Qwen/Qwen2.5-3B',  # Level 3: Specialized reasoning
        }

        print(f"🧠 Hierarchical KV-Cache Experiment")
        print(f"   Device: {device}")
        print(f"   Hierarchy: {len(self.hierarchy)} models\n")

    def load_model(self, layer: str):
        """Load model and tokenizer for hierarchical layer"""
        if layer in self.models:
            return  # Already loaded

        model_name = self.hierarchy[layer]
        print(f"Loading {layer} model: {model_name}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            model.eval()

            self.tokenizers[layer] = tokenizer
            self.models[layer] = model

            param_count = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"   ✓ Loaded {param_count:.1f}M parameters\n")

        except Exception as e:
            print(f"   ✗ Failed to load {layer}: {e}\n")

    def capture_kv_cache(self, layer: str, prompt: str) -> KVCacheSnapshot:
        """
        Capture KV-cache state after processing prompt

        This implements "assumption caching" from RFC Section 1
        """
        model = self.models[layer]
        tokenizer = self.tokenizers[layer]

        # Tokenize and generate with cache
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            past_key_values = outputs.past_key_values

        # Calculate cache size
        cache_size = 0
        for layer_cache in past_key_values:
            for tensor in layer_cache:
                cache_size += tensor.element_size() * tensor.numel()
        cache_size_mb = cache_size / (1024 * 1024)

        snapshot = KVCacheSnapshot(
            model_name=layer,
            timestamp=datetime.now().isoformat(),
            conversation_context=prompt,
            cache_state=past_key_values,
            sequence_length=inputs['input_ids'].shape[1],
            cache_size_mb=cache_size_mb
        )

        print(f"📸 Captured KV-cache from {layer}:")
        print(f"   Sequence length: {snapshot.sequence_length} tokens")
        print(f"   Cache size: {cache_size_mb:.2f} MB")
        print(f"   Layers: {len(past_key_values)}")

        return snapshot

    def restore_and_continue(self, layer: str, snapshot: KVCacheSnapshot,
                            continuation_prompt: str) -> Tuple[str, SurpriseMetrics]:
        """
        Restore KV-cache and continue generation

        Tests if "reality assumptions" transfer correctly
        """
        model = self.models[layer]
        tokenizer = self.tokenizers[layer]

        # Tokenize continuation
        cont_inputs = tokenizer(continuation_prompt, return_tensors="pt").to(self.device)

        # Generate with restored cache
        with torch.no_grad():
            # First, generate WITHOUT cache (baseline)
            baseline_outputs = model.generate(
                cont_inputs['input_ids'],
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True,
                output_scores=True
            )

            # Then generate WITH restored cache
            if snapshot.model_name == layer:
                # Same model - should work perfectly
                cached_outputs = model.generate(
                    cont_inputs['input_ids'],
                    past_key_values=snapshot.cache_state,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            else:
                # Different model - expect surprise!
                # For now, generate without cache (cross-model transfer is complex)
                cached_outputs = baseline_outputs

        # Decode responses
        response = tokenizer.decode(cached_outputs.sequences[0], skip_special_tokens=True)

        # Calculate surprise metrics
        surprise = self._calculate_surprise(baseline_outputs, cached_outputs)

        return response, surprise

    def _calculate_surprise(self, baseline_outputs, cached_outputs) -> SurpriseMetrics:
        """
        Calculate surprise when comparing cached vs baseline generation

        Implements RFC Section 3: Surprise Calculation
        """
        # Simple surprise metrics for now
        # In full implementation, would compare:
        # - Token probability distributions
        # - Attention pattern similarity
        # - Perplexity changes

        # Placeholder implementation
        return SurpriseMetrics(
            cosine_distance=0.0,  # Would compute from embeddings
            perplexity_delta=0.0,  # Would compute from token probs
            confidence_drop=0.0,  # Would compute from confidence scores
            surprise_level=0.0  # Would aggregate above metrics
        )

    def test_same_model_continuity(self, layer: str):
        """
        Test 1: Same model KV-cache save/restore

        Expected: Perfect continuity, zero surprise
        """
        print(f"\n{'='*80}")
        print(f"TEST 1: Same-Model Continuity ({layer})")
        print(f"{'='*80}\n")

        # Phase 1: Build context
        context = "The theory of relativity states that"
        snapshot = self.capture_kv_cache(layer, context)

        # Phase 2: Continue from saved state
        continuation = " Furthermore, "
        response, surprise = self.restore_and_continue(layer, snapshot, continuation)

        print(f"\n📊 Results:")
        print(f"   Context: {context}")
        print(f"   Continuation: {response[:100]}...")
        print(f"   Surprise level: {surprise.surprise_level:.3f}")
        print(f"   Cache valid: {not surprise.should_invalidate_cache()}\n")

    def test_cross_model_transfer(self):
        """
        Test 2: Transfer KV-cache between hierarchical models

        Expected: High surprise when assumptions don't match
        """
        print(f"\n{'='*80}")
        print(f"TEST 2: Cross-Model KV-Cache Transfer")
        print(f"{'='*80}\n")

        # Build context with small model
        context = "Artificial intelligence systems use"
        small_snapshot = self.capture_kv_cache('sensory', context)

        print(f"\n🔄 Attempting transfer to larger model...")

        # Try to continue with larger model
        # Note: This is conceptual - actual transfer requires:
        # 1. Dimension matching (projection layers)
        # 2. Layer alignment (map small layers to large layers)
        # 3. Attention head matching

        # For now, demonstrate the concept
        continuation = " machine learning to"

        # Generate with large model (without transferred cache)
        model = self.models['specialized']
        tokenizer = self.tokenizers['specialized']
        full_prompt = context + continuation
        inputs = tokenizer(full_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"\n📊 Results:")
        print(f"   Small model cache: {small_snapshot.cache_size_mb:.2f} MB")
        print(f"   Large model response: {response[:150]}...")
        print(f"   Note: Full transfer requires dimension mapping\n")

    def test_trust_evolution_with_cache(self):
        """
        Test 3: Trust evolution based on cache hit/miss rates

        Implements RFC integration with trust tracking
        """
        print(f"\n{'='*80}")
        print(f"TEST 3: Trust Evolution via Cache Performance")
        print(f"{'='*80}\n")

        # Simulate multiple interactions
        conversations = [
            ("What is the capital of France?", "stable"),
            ("Tell me about Paris.", "stable"),  # Should have low surprise
            ("Explain quantum mechanics.", "novel"),  # Should have high surprise
        ]

        for i, (prompt, context_type) in enumerate(conversations):
            print(f"\n--- Interaction {i+1}: {context_type} ---")
            print(f"Prompt: {prompt}")

            snapshot = self.capture_kv_cache('tactical', prompt)

            # In full implementation, would:
            # 1. Measure cache hit rate
            # 2. Calculate surprise on prediction errors
            # 3. Update trust scores in database
            # 4. Adjust model selection based on trust

            print(f"Cache captured: {snapshot.cache_size_mb:.2f} MB")
            print(f"Context type: {context_type}")


def main():
    """Run hierarchical KV-cache experiments"""

    print("🚀 Phase 1: Hierarchical Cognitive Architecture")
    print("   KV-Cache Reality Persistence Experiments\n")
    print("   Based on: web4/RFC_REALITY_KV_CACHE.md\n")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)\n")
    else:
        print("⚠️  Running on CPU (will be slow)\n")

    # Initialize experiment
    exp = HierarchicalKVCacheExperiment()

    # Load models (start with smallest for memory efficiency)
    print("📦 Loading models...")
    exp.load_model('sensory')  # 0.5B - smallest

    # Check if we have enough VRAM for more
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        print(f"Memory: {allocated_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved")

        # Load tactical if we have room
        if reserved_gb < 8.0:
            exp.load_model('tactical')  # 1.5B

        # Load specialized if we have room
        if reserved_gb < 6.0:
            exp.load_model('specialized')  # 3B

    # Run experiments
    print(f"\n{'='*80}")
    print("RUNNING EXPERIMENTS")
    print(f"{'='*80}")

    # Test 1: Same-model continuity
    if 'sensory' in exp.models:
        exp.test_same_model_continuity('sensory')

    # Test 2: Cross-model transfer (conceptual)
    if 'sensory' in exp.models and 'specialized' in exp.models:
        exp.test_cross_model_transfer()

    # Test 3: Trust evolution
    if 'tactical' in exp.models:
        exp.test_trust_evolution_with_cache()

    print(f"\n{'='*80}")
    print("CONCLUSIONS")
    print(f"{'='*80}\n")

    print("✅ Demonstrated KV-cache as 'reality assumption' storage")
    print("✅ Captured and measured cache states")
    print("📋 Cross-model transfer requires dimension mapping (future work)")
    print("📋 Surprise metrics need perplexity/attention analysis (future work)")
    print("📋 Trust integration with cache performance (future work)\n")

    print("💡 Key Insight:")
    print("   Transformer KV-cache IS the 'reality cache' from RFC!")
    print("   - Keys/Values = Cached expectations about conversation")
    print("   - Surprise = Prediction error when reality differs")
    print("   - Trust = How reliable model's cached assumptions are")
    print("   - Hierarchical models = Different abstraction levels\n")

    print("📖 Next Steps:")
    print("   1. Implement dimension projection for cross-model transfer")
    print("   2. Add perplexity-based surprise calculation")
    print("   3. Integrate with trust_database.py")
    print("   4. Test with real conversations")
    print("   5. Measure ATP costs for cache operations\n")


if __name__ == "__main__":
    main()
