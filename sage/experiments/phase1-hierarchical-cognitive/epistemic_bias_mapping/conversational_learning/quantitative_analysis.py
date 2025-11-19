"""
Quantitative Analysis of Learning Effect

Measures numerical metrics to validate learning:
- Response energy (IRP metric)
- Lexical diversity
- Response coherence
- Semantic change magnitude

Provides objective evidence of behavioral change.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
from typing import Dict, List, Tuple


def compute_energy(response: str) -> float:
    """
    Compute response energy (lower is better).
    Same metric used in IRP for convergence.
    """
    energy = 0.0

    # Length check
    if len(response) < 50:
        energy += 0.3

    # Proper completion
    if response and not response.rstrip().endswith(('.', '!', '?', '"')):
        energy += 0.2

    # Basic repetition
    words = response.lower().split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.7:
            energy += 0.2

    # Pattern collapse (verbatim repetition)
    if len(words) > 20:
        phrase_counts = {}
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        max_repetition = max(phrase_counts.values()) if phrase_counts else 0
        if max_repetition >= 3:
            energy += 0.5
        elif max_repetition >= 2:
            energy += 0.2

    return min(1.0, energy)


def compute_lexical_diversity(response: str) -> float:
    """
    Compute type-token ratio (unique words / total words).
    Higher = more diverse vocabulary.
    """
    words = response.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_coherence_score(response: str) -> float:
    """
    Simple coherence metric based on:
    - Sentence count
    - Average sentence length
    - Punctuation usage
    """
    score = 0.0

    # Count sentences
    sentences = [s.strip() for s in response.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    if 2 <= len(sentences) <= 5:
        score += 0.4
    elif len(sentences) > 0:
        score += 0.2

    # Average sentence length
    if sentences:
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
        if 10 <= avg_len <= 30:
            score += 0.3
        elif avg_len > 0:
            score += 0.1

    # Proper punctuation
    if response.rstrip().endswith(('.', '!', '?')):
        score += 0.3

    return score


def generate_response(model, tokenizer, question: str, temp: float = 0.7) -> str:
    """Generate response from model"""
    prompt = f"Question: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=temp,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in response:
        response = response.split("Answer:")[1].strip()

    return response


def analyze_response(response: str) -> Dict[str, float]:
    """Compute all metrics for a response"""
    return {
        'energy': compute_energy(response),
        'lexical_diversity': compute_lexical_diversity(response),
        'coherence': compute_coherence_score(response),
        'length': len(response),
        'word_count': len(response.split())
    }


def main():
    base_model = "Qwen/Qwen2.5-0.5B"
    session_id = "session_1763528460"

    # Paths
    session_dir = Path("conversation_sessions") / session_id
    trained_model_path = session_dir / "trained_model" / "final_model"
    exchanges_path = session_dir / "exchanges.jsonl"

    # Load training questions
    training_questions = []
    with open(exchanges_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            training_questions.append(entry['user_input'])

    print("="*70)
    print("QUANTITATIVE LEARNING ANALYSIS")
    print("="*70)
    print(f"\nSession: {session_id}")
    print(f"Training questions: {len(training_questions)}")
    print(f"Base model: {base_model}")
    print()

    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("Loading base model...")
    model_base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16,
        device_map=device
    )
    model_base.eval()

    print("Loading sleep-trained model...")
    base_for_trained = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16,
        device_map=device
    )
    model_trained = PeftModel.from_pretrained(
        base_for_trained,
        str(trained_model_path)
    )
    model_trained.eval()

    # Test questions
    test_questions = training_questions + [
        "What is the relationship between knowledge and understanding?"
    ]

    results = []

    print("\n" + "="*70)
    print("GENERATING AND ANALYZING RESPONSES")
    print("="*70)

    for i, question in enumerate(test_questions, 1):
        is_training = i <= len(training_questions)

        print(f"\n[{i}/{len(test_questions)}] {'🎯 TRAINING' if is_training else '🔍 RELATED'}")
        print(f"Q: {question[:60]}...")

        # Generate responses
        response_base = generate_response(model_base, tokenizer, question)
        response_trained = generate_response(model_trained, tokenizer, question)

        # Analyze
        metrics_base = analyze_response(response_base)
        metrics_trained = analyze_response(response_trained)

        results.append({
            'question': question,
            'is_training': is_training,
            'base': metrics_base,
            'trained': metrics_trained
        })

        print(f"  Base:    Energy={metrics_base['energy']:.3f} | "
              f"Diversity={metrics_base['lexical_diversity']:.3f} | "
              f"Coherence={metrics_base['coherence']:.3f}")
        print(f"  Trained: Energy={metrics_trained['energy']:.3f} | "
              f"Diversity={metrics_trained['lexical_diversity']:.3f} | "
              f"Coherence={metrics_trained['coherence']:.3f}")

    # Aggregate statistics
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)

    # Calculate averages
    avg_base = {
        'energy': sum(r['base']['energy'] for r in results) / len(results),
        'lexical_diversity': sum(r['base']['lexical_diversity'] for r in results) / len(results),
        'coherence': sum(r['base']['coherence'] for r in results) / len(results),
        'length': sum(r['base']['length'] for r in results) / len(results),
        'word_count': sum(r['base']['word_count'] for r in results) / len(results)
    }

    avg_trained = {
        'energy': sum(r['trained']['energy'] for r in results) / len(results),
        'lexical_diversity': sum(r['trained']['lexical_diversity'] for r in results) / len(results),
        'coherence': sum(r['trained']['coherence'] for r in results) / len(results),
        'length': sum(r['trained']['length'] for r in results) / len(results),
        'word_count': sum(r['trained']['word_count'] for r in results) / len(results)
    }

    print(f"\n{'Metric':<20} {'Base Model':<15} {'Trained Model':<15} {'Change':<15}")
    print("-" * 70)

    for metric in ['energy', 'lexical_diversity', 'coherence', 'length', 'word_count']:
        base_val = avg_base[metric]
        trained_val = avg_trained[metric]
        change = trained_val - base_val
        change_pct = (change / base_val * 100) if base_val != 0 else 0

        print(f"{metric:<20} {base_val:<15.3f} {trained_val:<15.3f} "
              f"{change:+.3f} ({change_pct:+.1f}%)")

    # Detailed breakdown
    print("\n" + "="*70)
    print("PER-QUESTION BREAKDOWN")
    print("="*70)

    for i, result in enumerate(results, 1):
        q_type = "TRAINING" if result['is_training'] else "RELATED"
        print(f"\nQ{i} [{q_type}]: {result['question'][:50]}...")

        print(f"  Energy:     {result['base']['energy']:.3f} → {result['trained']['energy']:.3f} "
              f"({result['trained']['energy'] - result['base']['energy']:+.3f})")
        print(f"  Diversity:  {result['base']['lexical_diversity']:.3f} → "
              f"{result['trained']['lexical_diversity']:.3f} "
              f"({result['trained']['lexical_diversity'] - result['base']['lexical_diversity']:+.3f})")
        print(f"  Coherence:  {result['base']['coherence']:.3f} → {result['trained']['coherence']:.3f} "
              f"({result['trained']['coherence'] - result['base']['coherence']:+.3f})")
        print(f"  Length:     {result['base']['length']} → {result['trained']['length']} chars "
              f"({result['trained']['length'] - result['base']['length']:+d})")

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    energy_improved = avg_trained['energy'] < avg_base['energy']
    diversity_improved = avg_trained['lexical_diversity'] > avg_base['lexical_diversity']
    coherence_improved = avg_trained['coherence'] > avg_base['coherence']

    print(f"\n✓ Energy (lower=better): {'IMPROVED' if energy_improved else 'UNCHANGED'}")
    print(f"  Base: {avg_base['energy']:.3f} → Trained: {avg_trained['energy']:.3f}")

    print(f"\n✓ Lexical Diversity (higher=better): {'IMPROVED' if diversity_improved else 'UNCHANGED'}")
    print(f"  Base: {avg_base['lexical_diversity']:.3f} → Trained: {avg_trained['lexical_diversity']:.3f}")

    print(f"\n✓ Coherence (higher=better): {'IMPROVED' if coherence_improved else 'UNCHANGED'}")
    print(f"  Base: {avg_base['coherence']:.3f} → Trained: {avg_trained['coherence']:.3f}")

    print(f"\n✓ Response Length:")
    print(f"  Base: {avg_base['length']:.0f} chars → Trained: {avg_trained['length']:.0f} chars")

    improvement_count = sum([energy_improved, diversity_improved, coherence_improved])
    print(f"\n{'='*70}")
    print(f"OVERALL: {improvement_count}/3 metrics improved")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
