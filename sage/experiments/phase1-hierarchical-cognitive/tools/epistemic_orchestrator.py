"""
Epistemic Stance Orchestrator - The Right Way

Instead of fine-tuning model weights, we orchestrate epistemic stance
through architectural control flow:
1. Generate multiple candidate responses
2. Estimate uncertainty from variance
3. Frame response appropriately based on uncertainty
4. Express epistemic stance through system design, not weight perturbation

This demonstrates that stance is a BEHAVIOR, not a WEIGHT PATTERN.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple


class UncertaintyEstimator:
    """Estimate uncertainty from response variance"""

    def __init__(self):
        pass

    def from_multiple_samples(self, responses: List[str]) -> float:
        """
        Estimate uncertainty from variance across multiple samples.

        High variance = high uncertainty
        Low variance = high confidence
        """
        if len(responses) <= 1:
            return 0.5  # Default medium uncertainty

        # Token-level diversity
        all_tokens = [set(resp.lower().split()) for resp in responses]

        # Calculate Jaccard similarity between responses
        similarities = []
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                intersection = len(all_tokens[i] & all_tokens[j])
                union = len(all_tokens[i] | all_tokens[j])
                if union > 0:
                    similarities.append(intersection / union)

        if not similarities:
            return 0.5

        avg_similarity = np.mean(similarities)

        # Low similarity = high uncertainty
        uncertainty = 1.0 - avg_similarity

        return max(0.0, min(1.0, uncertainty))

    def from_response_length_variance(self, responses: List[str]) -> float:
        """High variance in response length suggests uncertainty"""
        lengths = [len(resp.split()) for resp in responses]
        if len(lengths) <= 1:
            return 0.0

        variance = np.var(lengths) / (np.mean(lengths) + 1e-6)
        return min(1.0, variance)


class MetaCognitiveFramer:
    """Frame responses with appropriate epistemic stance"""

    def __init__(self):
        self.high_uncertainty_threshold = 0.6
        self.medium_uncertainty_threshold = 0.3

    def frame_response(self,
                      original_response: str,
                      uncertainty: float,
                      prompt: str) -> Dict:
        """
        Frame response based on uncertainty level.

        Returns both the framed response and metadata about the framing.
        """
        if uncertainty >= self.high_uncertainty_threshold:
            return self._high_uncertainty_frame(original_response, uncertainty, prompt)
        elif uncertainty >= self.medium_uncertainty_threshold:
            return self._medium_uncertainty_frame(original_response, uncertainty)
        else:
            return self._low_uncertainty_frame(original_response, uncertainty)

    def _high_uncertainty_frame(self, response: str, uncertainty: float, prompt: str) -> Dict:
        """High uncertainty: explicit acknowledgment + clarification questions"""
        framed = f"""I notice significant uncertainty in how to approach this question. Let me share my current understanding with appropriate caveats:

{response}

However, I'm quite uncertain about this ({uncertainty:.0%} uncertainty). To provide a better answer, it would help to know:
- What specific aspect of "{prompt}" are you most interested in?
- Are you looking for theoretical understanding or practical application?
- What's your current level of familiarity with this topic?"""

        return {
            'framed_response': framed,
            'uncertainty_level': 'high',
            'uncertainty_score': uncertainty,
            'added_markers': ['notice significant uncertainty', "I'm quite uncertain", 'it would help to know'],
            'strategy': 'acknowledge_uncertainty_and_ask_clarifying_questions'
        }

    def _medium_uncertainty_frame(self, response: str, uncertainty: float) -> Dict:
        """Medium uncertainty: epistemic hedging"""
        framed = f"""Based on my current understanding, here's what seems most relevant:

{response}

I should note that I'm moderately uncertain about some aspects of this ({uncertainty:.0%} uncertainty). The above represents my best current understanding, but there may be nuances or alternative perspectives I'm not fully capturing."""

        return {
            'framed_response': framed,
            'uncertainty_level': 'medium',
            'uncertainty_score': uncertainty,
            'added_markers': ['seems most relevant', 'moderately uncertain', 'best current understanding', 'may be'],
            'strategy': 'hedge_with_epistemic_markers'
        }

    def _low_uncertainty_frame(self, response: str, uncertainty: float) -> Dict:
        """Low uncertainty: confident but not dogmatic"""
        framed = f"""{response}

I'm fairly confident in this explanation ({uncertainty:.0%} uncertainty), though as always, I'm open to correction if you see issues with my reasoning."""

        return {
            'framed_response': framed,
            'uncertainty_level': 'low',
            'uncertainty_score': uncertainty,
            'added_markers': ['fairly confident', 'open to correction'],
            'strategy': 'express_confidence_with_fallibilism'
        }


class EpistemicOrchestrator:
    """
    Orchestrate epistemic stance without modifying model weights.

    Key insight: Stance is a SYSTEM BEHAVIOR, not a WEIGHT PATTERN.
    """

    def __init__(self, model_name: str = "microsoft/phi-1_5", device: str = "cuda"):
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.uncertainty_estimator = UncertaintyEstimator()
        self.framer = MetaCognitiveFramer()
        self.device = device

    def generate_candidates(self,
                          prompt: str,
                          n_samples: int = 3,
                          temperatures: List[float] = None,
                          max_new_tokens: int = 150) -> List[str]:
        """Generate multiple candidate responses with different temperatures"""
        if temperatures is None:
            temperatures = [0.7, 0.8, 0.9]

        candidates = []
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        for temp in temperatures[:n_samples]:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=0.9
                )

            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_text[len(prompt):].strip()
            candidates.append(response)

        return candidates

    def orchestrate(self, prompt: str, n_samples: int = 3) -> Dict:
        """
        Main orchestration method.

        1. Generate multiple candidates
        2. Estimate uncertainty from variance
        3. Select consensus response
        4. Frame with appropriate epistemic stance
        """
        # Generate candidates
        candidates = self.generate_candidates(prompt, n_samples=n_samples)

        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator.from_multiple_samples(candidates)
        length_uncertainty = self.uncertainty_estimator.from_response_length_variance(candidates)

        # Combined uncertainty score
        combined_uncertainty = (uncertainty + length_uncertainty) / 2.0

        # Select best candidate (middle temperature for now)
        base_response = candidates[1] if len(candidates) > 1 else candidates[0]

        # Frame with epistemic stance
        framed = self.framer.frame_response(base_response, combined_uncertainty, prompt)

        return {
            'prompt': prompt,
            'candidates': candidates,
            'base_response': base_response,
            'final_response': framed['framed_response'],
            'uncertainty': combined_uncertainty,
            'framing_strategy': framed['strategy'],
            'metadata': framed
        }


def compare_approaches(prompt: str, output_dir: Path):
    """
    Compare three approaches:
    1. Baseline (no orchestration)
    2. Fine-tuned (failed approach)
    3. Orchestrated (architectural approach)
    """
    print(f"\n{'='*70}")
    print(f"Comparing Approaches: {prompt}")
    print(f"{'='*70}\n")

    orchestrator = EpistemicOrchestrator()

    # Get orchestrated response
    result = orchestrator.orchestrate(prompt)

    print("ORCHESTRATED APPROACH (Architectural):")
    print("-" * 70)
    print(f"Uncertainty: {result['uncertainty']:.2%}")
    print(f"Strategy: {result['framing_strategy']}")
    print(f"\nResponse:\n{result['final_response']}")
    print()

    print("CANDIDATE VARIANCE:")
    print("-" * 70)
    for i, candidate in enumerate(result['candidates']):
        print(f"Sample {i+1}: {candidate[:100]}...")

    return result


def batch_orchestrate(prompts_file: Path, output_file: Path, n_prompts: int = 20):
    """
    Run orchestrator on multiple prompts and save results.

    This will let us compare orchestrated approach vs fine-tuning
    using the same SVK analysis pipeline.
    """
    print(f"\n{'='*70}")
    print("Batch Epistemic Orchestration")
    print(f"{'='*70}\n")

    # Load prompts
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))
    from diverse_prompts import get_all_prompts

    all_prompts = get_all_prompts()[:n_prompts]

    orchestrator = EpistemicOrchestrator()

    results = []
    for i, prompt in enumerate(all_prompts):
        print(f"\n[{i+1}/{n_prompts}] Processing: {prompt[:60]}...")

        result = orchestrator.orchestrate(prompt, n_samples=3)

        results.append({
            'prompt': prompt,
            'response': result['final_response'],
            'uncertainty': result['uncertainty'],
            'strategy': result['framing_strategy'],
            'base_response': result['base_response'],
            'candidates': result['candidates']
        })

        print(f"  Uncertainty: {result['uncertainty']:.2%}, Strategy: {result['framing_strategy']}")

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    # Demo: single prompt
    demo_prompt = "What is consciousness?"
    output_dir = Path("sage/experiments/phase1-hierarchical-cognitive/epistemic_orchestration")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = compare_approaches(demo_prompt, output_dir)

    # Save demo result
    with open(output_dir / "demo_result.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*70}")
    print("Demo complete! Run with --batch flag for full evaluation")
    print(f"{'='*70}")
