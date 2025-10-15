"""
Model Selection Test Harness - Phase 1
Hierarchical Cognitive Architecture

Tests trust-based model selection across the Qwen family
Integrates with trust_database.py for dynamic trust evolution
"""

import time
import json
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import subprocess
import re

from trust_database import TrustTrackingDatabase, TrainingExample


@dataclass
class ModelConfig:
    """Configuration for a cognitive layer model"""
    name: str
    ollama_name: str  # Name in Ollama
    params: str  # Parameter count (for display)
    layer: str  # 'strategic', 'specialized', 'tactical', 'sensory'
    expected_speed: str  # Expected inference speed


# Model hierarchy configuration
MODEL_CONFIGS = {
    'claude': ModelConfig(
        name='claude',
        ollama_name='claude-sonnet-4.5',  # Conceptual (API)
        params='N/A',
        layer='strategic',
        expected_speed='slow'
    ),
    'qwen-3b': ModelConfig(
        name='qwen-3b',
        ollama_name='qwen2.5:3b',
        params='3B',
        layer='specialized',
        expected_speed='medium'
    ),
    'qwen-1.5b': ModelConfig(
        name='qwen-1.5b',
        ollama_name='qwen2.5:1.5b',
        params='1.5B',
        layer='tactical',
        expected_speed='fast'
    ),
    'qwen-0.5b': ModelConfig(
        name='qwen-0.5b',
        ollama_name='qwen2:0.5b',
        params='0.5B',
        layer='sensory',
        expected_speed='very-fast'
    ),
    'phi3': ModelConfig(
        name='phi3',
        ollama_name='phi3:mini',
        params='3.8B',
        layer='specialized',
        expected_speed='medium'
    ),
    'gemma': ModelConfig(
        name='gemma',
        ollama_name='gemma:2b',
        params='2B',
        layer='tactical',
        expected_speed='fast'
    ),
    'tinyllama': ModelConfig(
        name='tinyllama',
        ollama_name='tinyllama:latest',
        params='1.1B',
        layer='sensory',
        expected_speed='very-fast'
    )
}


@dataclass
class InferenceResult:
    """Result of model inference"""
    model_name: str
    prompt: str
    response: str
    latency_ms: float
    tokens_per_sec: float
    confidence: float  # Estimated from response quality
    success: bool
    error: Optional[str] = None


class ContextClassifier:
    """
    Classifies cognitive contexts for trust-based model selection

    Context types:
    - stable: Well-known, repeated patterns
    - moving: Familiar but changing
    - unstable: High uncertainty, conflicting information
    - novel: New patterns, no prior experience
    """

    def __init__(self):
        # Simple heuristic classifier (could be learned later)
        self.history = []

    def classify(self, prompt: str, conversation_history: List[str] = None) -> str:
        """Classify the cognitive context of a prompt"""

        prompt_lower = prompt.lower()

        # Novel: Questions, explorations, new topics
        if any(word in prompt_lower for word in ['what', 'why', 'how', 'explain', 'tell me about']):
            return 'novel'

        # Unstable: Contradictions, uncertainties, complex reasoning
        if any(word in prompt_lower for word in ['however', 'but', 'uncertain', 'complex', 'analyze']):
            return 'unstable'

        # Moving: Variations on known topics
        if conversation_history and len(conversation_history) > 0:
            # Check if similar to recent prompts
            recent_similarity = self._compute_similarity(prompt, conversation_history[-3:])
            if 0.3 < recent_similarity < 0.7:
                return 'moving'

        # Stable: Simple, direct, repeated patterns
        if any(word in prompt_lower for word in ['hello', 'thanks', 'yes', 'no', 'list', 'show']):
            return 'stable'

        # Default to moving for general queries
        return 'moving'

    def _compute_similarity(self, text: str, reference_texts: List[str]) -> float:
        """Compute simple word overlap similarity"""
        words = set(text.lower().split())

        if not reference_texts:
            return 0.0

        similarities = []
        for ref in reference_texts:
            ref_words = set(ref.lower().split())
            if not ref_words:
                continue
            overlap = len(words & ref_words)
            union = len(words | ref_words)
            similarities.append(overlap / union if union > 0 else 0.0)

        return max(similarities) if similarities else 0.0


class ModelSelector:
    """
    Trust-based hierarchical model selector

    Implements cognitive layer selection based on:
    1. Context classification (stable/moving/unstable/novel)
    2. Trust scores from previous performance
    3. ATP budget constraints
    """

    def __init__(self, db: TrustTrackingDatabase, atp_budget: float = 100.0):
        self.db = db
        self.atp_budget = atp_budget
        self.atp_spent = 0.0
        self.context_classifier = ContextClassifier()
        self.conversation_history = []

    def select_model(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Select best model for prompt based on trust and context

        Returns: model name to use
        """
        # Classify context if not provided
        if context is None:
            context = self.context_classifier.classify(prompt, self.conversation_history)

        # Get trust scores for all models in this context
        trust_scores = self.db.get_all_trust(context)

        # Select model with highest trust
        if trust_scores:
            best_model = max(trust_scores.items(), key=lambda x: x[1])[0]
        else:
            # Default fallback
            best_model = 'qwen-1.5b'

        return best_model

    def estimate_atp_cost(self, model_name: str, prompt: str) -> float:
        """
        Estimate ATP cost for running this model on this prompt

        Simple heuristic for now (could be learned)
        """
        # Base costs by model size
        base_costs = {
            'claude': 10.0,  # API call
            'qwen-3b': 3.0,
            'qwen-1.5b': 1.5,
            'qwen-0.5b': 0.5,
            'phi3': 3.5,
            'gemma': 2.0,
            'tinyllama': 1.0
        }

        base = base_costs.get(model_name, 2.0)

        # Scale by prompt complexity
        word_count = len(prompt.split())
        complexity_factor = 1.0 + (word_count / 100.0)

        return base * complexity_factor

    def invoke_model(self, model_name: str, prompt: str,
                    max_tokens: int = 256) -> InferenceResult:
        """
        Invoke a model via Ollama and measure performance
        """
        # Skip Claude (not available via Ollama)
        if model_name == 'claude':
            return InferenceResult(
                model_name=model_name,
                prompt=prompt,
                response="[Claude API not available in test harness]",
                latency_ms=0.0,
                tokens_per_sec=0.0,
                confidence=0.0,
                success=False,
                error="API model not available"
            )

        config = MODEL_CONFIGS.get(model_name)
        if not config:
            return InferenceResult(
                model_name=model_name,
                prompt=prompt,
                response="",
                latency_ms=0.0,
                tokens_per_sec=0.0,
                confidence=0.0,
                success=False,
                error=f"Unknown model: {model_name}"
            )

        # Invoke via Ollama
        start_time = time.time()

        try:
            result = subprocess.run(
                ['ollama', 'run', config.ollama_name, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )

            latency = (time.time() - start_time) * 1000  # ms

            if result.returncode == 0:
                response = result.stdout.strip()

                # Estimate tokens (rough approximation)
                tokens = len(response.split())
                tokens_per_sec = tokens / (latency / 1000.0) if latency > 0 else 0.0

                # Estimate confidence from response quality
                confidence = self._estimate_confidence(response, prompt)

                return InferenceResult(
                    model_name=model_name,
                    prompt=prompt,
                    response=response,
                    latency_ms=latency,
                    tokens_per_sec=tokens_per_sec,
                    confidence=confidence,
                    success=True
                )
            else:
                return InferenceResult(
                    model_name=model_name,
                    prompt=prompt,
                    response="",
                    latency_ms=latency,
                    tokens_per_sec=0.0,
                    confidence=0.0,
                    success=False,
                    error=result.stderr
                )

        except subprocess.TimeoutExpired:
            latency = (time.time() - start_time) * 1000
            return InferenceResult(
                model_name=model_name,
                prompt=prompt,
                response="",
                latency_ms=latency,
                tokens_per_sec=0.0,
                confidence=0.0,
                success=False,
                error="Timeout"
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return InferenceResult(
                model_name=model_name,
                prompt=prompt,
                response="",
                latency_ms=latency,
                tokens_per_sec=0.0,
                confidence=0.0,
                success=False,
                error=str(e)
            )

    def _estimate_confidence(self, response: str, prompt: str) -> float:
        """
        Estimate confidence from response quality

        Heuristics:
        - Longer responses tend to be more confident
        - Complete sentences indicate confidence
        - Hedging words reduce confidence
        """
        if not response:
            return 0.0

        confidence = 0.5  # Baseline

        # Length bonus
        word_count = len(response.split())
        if word_count > 20:
            confidence += 0.2
        elif word_count > 10:
            confidence += 0.1

        # Penalize hedging
        hedging_words = ['maybe', 'perhaps', 'uncertain', 'not sure', 'might', 'could']
        hedge_count = sum(1 for word in hedging_words if word in response.lower())
        confidence -= hedge_count * 0.1

        # Bonus for complete sentences
        if response.strip().endswith(('.', '!', '?')):
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def update_trust_from_result(self, result: InferenceResult,
                                 context: str, user_rating: Optional[float] = None):
        """
        Update trust database based on inference result

        Outcome determined by:
        - Success/failure of inference
        - Confidence score
        - Optional user rating
        """
        # Determine outcome
        if not result.success:
            outcome = 'failure'
        elif user_rating is not None:
            outcome = 'success' if user_rating > 0.5 else 'failure'
        elif result.confidence > 0.6:
            outcome = 'success'
        else:
            outcome = 'uncertain'

        # Update trust
        new_trust = self.db.update_trust(
            result.model_name,
            context,
            outcome,
            learning_rate=0.1
        )

        return new_trust


class BenchmarkSuite:
    """Benchmark suite for testing all models"""

    def __init__(self, db_path: str = "phase1_benchmark.db"):
        self.db = TrustTrackingDatabase(db_path)
        self.selector = ModelSelector(self.db)

    def benchmark_all_models(self, prompt: str,
                            models: List[str] = None) -> Dict[str, InferenceResult]:
        """Benchmark all models on the same prompt"""

        if models is None:
            # Test all available models (skip Claude)
            models = [m for m in MODEL_CONFIGS.keys() if m != 'claude']

        results = {}

        print(f"\n{'='*80}")
        print(f"Benchmarking prompt: {prompt[:60]}...")
        print(f"{'='*80}\n")

        for model in models:
            print(f"Testing {model}...", end=' ', flush=True)
            result = self.selector.invoke_model(model, prompt)
            results[model] = result

            if result.success:
                print(f"✓ {result.latency_ms:.0f}ms ({result.tokens_per_sec:.1f} tok/s)")
            else:
                print(f"✗ {result.error}")

        return results

    def compare_models(self, results: Dict[str, InferenceResult]):
        """Print comparison table"""

        print(f"\n{'='*80}")
        print("BENCHMARK RESULTS")
        print(f"{'='*80}")
        print(f"{'Model':<15} {'Latency':<12} {'Speed':<15} {'Confidence':<12} {'Status'}")
        print(f"{'-'*80}")

        # Sort by latency
        sorted_results = sorted(results.items(), key=lambda x: x[1].latency_ms)

        for model, result in sorted_results:
            if result.success:
                print(f"{model:<15} {result.latency_ms:>8.0f} ms  "
                      f"{result.tokens_per_sec:>8.1f} tok/s  "
                      f"{result.confidence:>8.2f}      ✓")
            else:
                print(f"{model:<15} {result.latency_ms:>8.0f} ms  "
                      f"{'N/A':>8}       {'N/A':>8}      ✗ {result.error}")

        print(f"{'='*80}\n")

    def test_trust_evolution(self, prompts: List[str], contexts: List[str]):
        """Test trust evolution across multiple prompts"""

        print(f"\n{'='*80}")
        print("TRUST EVOLUTION TEST")
        print(f"{'='*80}\n")

        for i, (prompt, context) in enumerate(zip(prompts, contexts)):
            print(f"\n--- Test {i+1}: {context.upper()} context ---")
            print(f"Prompt: {prompt}\n")

            # Get trust-based model selection
            selected_model = self.selector.select_model(prompt, context)
            print(f"Selected model: {selected_model}")

            # Get current trust scores
            trust_scores = self.db.get_all_trust(context)
            print(f"Trust scores before: {json.dumps(trust_scores, indent=2)}")

            # Run inference
            result = self.selector.invoke_model(selected_model, prompt)

            # Update trust (simulate user rating based on confidence)
            user_rating = result.confidence if result.success else 0.0
            new_trust = self.selector.update_trust_from_result(result, context, user_rating)

            print(f"\nResult: {'Success' if result.success else 'Failure'}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"New trust for {selected_model}: {new_trust:.3f}")

            # Show updated trust scores
            updated_trust = self.db.get_all_trust(context)
            print(f"Trust scores after: {json.dumps(updated_trust, indent=2)}")


if __name__ == "__main__":
    print("🧠 Phase 1: Hierarchical Cognitive Architecture")
    print("   Model Selection Test Harness\n")

    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU: {gpu_name}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    else:
        print("⚠️  No GPU detected - tests will run on CPU\n")

    # Initialize benchmark suite
    suite = BenchmarkSuite("phase1_hierarchical_test.db")

    # Test 1: Benchmark all models on same prompt
    print("\n" + "="*80)
    print("TEST 1: BENCHMARK ALL MODELS")
    print("="*80)

    test_prompt = "Explain the concept of trust in machine learning systems in one sentence."

    results = suite.benchmark_all_models(test_prompt)
    suite.compare_models(results)

    # Test 2: Trust evolution across different contexts
    print("\n" + "="*80)
    print("TEST 2: TRUST EVOLUTION ACROSS CONTEXTS")
    print("="*80)

    test_cases = [
        ("Hello! How are you today?", "stable"),
        ("What is the capital of France?", "stable"),
        ("Explain quantum entanglement briefly.", "novel"),
        ("How does climate change affect biodiversity?", "novel"),
        ("The data shows contradictory trends. Analyze this.", "unstable"),
    ]

    prompts, contexts = zip(*test_cases)
    suite.test_trust_evolution(list(prompts), list(contexts))

    # Print final statistics
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)

    stats = suite.db.get_statistics()
    print(json.dumps(stats, indent=2))

    print("\n✅ Phase 1 test harness complete!")
    print(f"   Database: phase1_hierarchical_test.db")
    print(f"   Trust tracking: {stats['total_examples']} examples collected\n")
