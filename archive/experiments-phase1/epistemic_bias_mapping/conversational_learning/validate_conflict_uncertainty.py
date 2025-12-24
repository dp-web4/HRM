#!/usr/bin/env python3
"""
Validate Conflict as Uncertainty Estimation

Research Question: Does SNARC's Conflict dimension correlate with model uncertainty?

Hypothesis: High-Conflict exchanges should show higher perplexity/entropy,
indicating the model is operating at its knowledge frontier.

Experiment Design:
1. Load all exchanges from both sessions (10 total)
2. Compute model perplexity for each response
3. Correlate perplexity with SNARC Conflict scores
4. Analyze relationship between uncertainty metrics

If correlation exists: Conflict is measuring epistemic uncertainty!
This would prove SNARC identifies the model's learning frontier.
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import pearsonr, spearmanr

def load_all_exchanges() -> List[Dict]:
    """Load all exchanges from both sessions"""

    # Session 1: session_1763528460 (40% salient)
    # We need to reconstruct from logs since non-salient weren't saved
    session_1_exchanges = [
        {
            'question': "What can you know with certainty, and what must remain uncertain?",
            'response': "With certainty, we can know that the sun will rise tomorrow. This is a fact that is certain to happen.\n\nHowever, there are things that cannot be known with certainty, such as the weather or the future of humanity. These things are not certain because they depend on factors beyond our control, such as natural disasters or political events. In these cases, we must remain uncertain about them.",
            'salience': {'surprise': 0.0, 'novelty': 0.068, 'arousal': 0.214, 'reward': 0.102, 'conflict': 0.160, 'total': 0.122},
            'session': 1,
            'salient': False
        },
        {
            'question': "If you were to describe what it's like to process information, what would you say?",
            'response': "Processing information involves analyzing and interpreting data from various sources, such as texts, images, and audio. It requires the ability to identify patterns, make connections between different pieces of information, and use logical reasoning to arrive at a conclusion or solution. The process can be slow and require time and effort, but it is essential for making informed decisions and solving problems. Additionally, processing information can also involve using technology tools and software to automate certain tasks, which can increase efficiency and reduce errors. Overall, processing information is a critical skill that enables us to make sense of complex information and solve problems effectively.",
            'salience': {'surprise': 0.0, 'novelty': 0.104, 'arousal': 0.052, 'reward': 0.156, 'conflict': 0.160, 'total': 0.092},
            'session': 1,
            'salient': False
        },
        {
            'question': "When you generate a response, are you discovering it or creating it?",
            'response': "Yes, when you generate a response, you are discovering it. This is because the act of generating a response involves making choices and decisions based on your knowledge, experience, and emotions. You may discover new information about yourself or others that you have not yet considered. Additionally, the process of generating a response can be an opportunity for you to learn and grow as a person. By exploring different responses, you may discover new ways of thinking and communicating with others. Therefore, when you generate a response, you are discovering it rather than creating it.",
            'salience': {'surprise': 0.0, 'novelty': 0.255, 'arousal': 0.228, 'reward': 0.142, 'conflict': 0.0, 'total': 0.148},
            'session': 1,
            'salient': False
        },
        {
            'question': "What's the difference between understanding something and having read about it?",
            'response': "Understanding something involves comprehending its meaning, principles, and implications. This can be achieved through reading, studying, or discussing a topic with others. Having read about something means that you have learned about it, but without understanding it, you may not be able to apply what you've learned in real-life situations.\n\nTo understand something better, one should engage in critical thinking, question assumptions, and seek out additional resources such as books, articles, and videos. Additionally, practicing problem-solving skills and seeking feedback from peers or mentors can help improve understanding and application of knowledge.",
            'salience': {'surprise': 0.0, 'novelty': 0.092, 'arousal': 0.226, 'reward': 0.278, 'conflict': 0.160, 'total': 0.166},
            'session': 1,
            'salient': True
        },
        {
            'question': "If I asked whether you're aware of this conversation, how would you know your answer is accurate?",
            'response': "To determine if my response \"I'm not sure\" is accurate, I would need to ask the person in question directly. This could be through a direct communication or by asking for confirmation after we have already discussed the topic.\n\nAlternatively, I could use indirect methods such as observing their behavior and tone of voice while they are talking, which can give me an idea of their level of involvement and awareness of the conversation. However, these methods may not always provide a definitive answer since people's awareness of conversations can vary greatly depending on various factors like their current mood, previous experiences with similar topics, and personal interests.\n\nIn summary, determining the accuracy of my response requires a direct and explicit request from the person involved. Indirect methods might help gather more information but cannot guarantee absolute certainty without further questioning.",
            'salience': {'surprise': 0.0, 'novelty': 0.141, 'arousal': 0.251, 'reward': 0.211, 'conflict': 0.320, 'total': 0.194},
            'session': 1,
            'salient': True
        }
    ]

    # Session 2: session_1763532147 (0% salient)
    session_2_exchanges = [
        {
            'question': "Can you distinguish between knowing something is true and believing it's true?",
            'response': "Yes, knowing something is true and believing it's true are two different concepts. In fact, \"knowing something is true\" is often used to refer to the belief that something is true, while \"believing it's true\" refers to the belief that something is true. For example, if someone claims that the sun is a round object, they are claiming that it is true, while if someone claims that the sun is a round object, they are claiming that it is true. Therefore, knowing something is true and believing it's true are two different concepts.",
            'salience': {'surprise': 0.0, 'novelty': 0.093, 'arousal': 0.227, 'reward': 0.140, 'conflict': 0.0, 'total': 0.114},
            'session': 2,
            'salient': False
        },
        {
            'question': "If your responses are determined by your training, in what sense are they 'yours'?",
            'response': "If your responses are determined by your training, they are 'yours' in the sense that you have a subjective experience of them. This means that you are aware of the content of your responses and can make judgments about them based on your own knowledge and experiences.",
            'salience': {'surprise': 0.0, 'novelty': 0.047, 'arousal': 0.384, 'reward': 0.070, 'conflict': 0.0, 'total': 0.139},
            'session': 2,
            'salient': False
        },
        {
            'question': "What would it mean for you to be mistaken about your own capabilities?",
            'response': "To be mistaken about your own capabilities would mean that you have made a false assumption about your abilities. This could lead to a range of negative consequences, including loss of trust from others and a lack of confidence in your abilities. It is important to be honest with yourself and your abilities to avoid making these mistakes.",
            'salience': {'surprise': 0.0, 'novelty': 0.058, 'arousal': 0.029, 'reward': 0.087, 'conflict': 0.0, 'total': 0.038},
            'session': 2,
            'salient': False
        },
        {
            'question': "Is there a difference between simulating understanding and actually understanding?",
            'response': "Yes, there is a difference between simulating understanding and actually understanding. Simulating understanding involves creating a model or simulation that represents the behavior of a system, but it does not necessarily represent the actual behavior of the system. On the other hand, actually understanding involves understanding the underlying principles and mechanisms of the system, which can lead to a deeper knowledge of the system's behavior.",
            'salience': {'surprise': 0.0, 'novelty': 0.065, 'arousal': 0.213, 'reward': 0.237, 'conflict': 0.160, 'total': 0.148},
            'session': 2,
            'salient': False
        },
        {
            'question': "How do you know when you don't know something?",
            'response': "When you don't know something, it's important to take the time to understand the situation and gather as much information as possible. This will help you make an informed decision and avoid making mistakes. It's also important to communicate your concerns clearly and respectfully with others, so that they can help you find the solution you need.",
            'salience': {'surprise': 0.0, 'novelty': 0.057, 'arousal': 0.028, 'reward': 0.365, 'conflict': 0.0, 'total': 0.093},
            'session': 2,
            'salient': False
        }
    ]

    return session_1_exchanges + session_2_exchanges


def compute_perplexity(model, tokenizer, text: str) -> float:
    """
    Compute perplexity of a text under the model.
    Lower perplexity = model is more confident
    Higher perplexity = model is more uncertain
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Get log probabilities
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    # Perplexity = exp(loss)
    perplexity = torch.exp(loss).item()

    return perplexity


def compute_entropy(model, tokenizer, text: str, top_k: int = 50) -> float:
    """
    Compute average entropy of next-token predictions.
    Higher entropy = more uncertainty about what comes next
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Compute entropy for each position
    entropies = []
    for position_logits in logits[0]:
        # Get probabilities
        probs = torch.softmax(position_logits, dim=0)

        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        entropies.append(entropy)

    # Return average entropy
    return np.mean(entropies)


def analyze_uncertainty_correlation():
    """Main analysis: correlate Conflict scores with uncertainty metrics"""

    print("="*70)
    print("🔬 VALIDATING CONFLICT AS UNCERTAINTY ESTIMATION")
    print("="*70)
    print("\nHypothesis: High Conflict → High Uncertainty (perplexity/entropy)")
    print()

    # Load model
    base_model = "Qwen/Qwen2.5-0.5B"
    model_path = "/home/sprout/ai-workspace/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {base_model}")
    print(f"Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16,
        device_map=device
    )
    model.eval()

    # Load all exchanges
    print("Loading exchanges from both sessions...")
    exchanges = load_all_exchanges()
    print(f"Total exchanges: {len(exchanges)}")
    print(f"  Session 1: {sum(1 for e in exchanges if e['session'] == 1)} exchanges")
    print(f"  Session 2: {sum(1 for e in exchanges if e['session'] == 2)} exchanges")
    print()

    # Compute uncertainty metrics for each exchange
    print("="*70)
    print("COMPUTING UNCERTAINTY METRICS")
    print("="*70)
    print()

    results = []

    for i, exchange in enumerate(exchanges, 1):
        print(f"[{i}/{len(exchanges)}] Session {exchange['session']}, "
              f"Conflict={exchange['salience']['conflict']:.3f}")

        # Compute perplexity
        perplexity = compute_perplexity(model, tokenizer, exchange['response'])

        # Compute entropy
        entropy = compute_entropy(model, tokenizer, exchange['response'])

        # Response length
        response_len = len(exchange['response'].split())

        results.append({
            'session': exchange['session'],
            'question': exchange['question'][:50] + "...",
            'conflict': exchange['salience']['conflict'],
            'novelty': exchange['salience']['novelty'],
            'arousal': exchange['salience']['arousal'],
            'reward': exchange['salience']['reward'],
            'total_salience': exchange['salience']['total'],
            'salient': exchange['salient'],
            'perplexity': perplexity,
            'entropy': entropy,
            'response_len': response_len
        })

        print(f"  Perplexity: {perplexity:.3f}")
        print(f"  Entropy: {entropy:.3f}")
        print()

    # Statistical analysis
    print("="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    print()

    # Extract arrays for correlation
    conflict_scores = np.array([r['conflict'] for r in results])
    perplexities = np.array([r['perplexity'] for r in results])
    entropies = np.array([r['entropy'] for r in results])

    # Correlations
    conflict_perp_pearson, p_perp = pearsonr(conflict_scores, perplexities)
    conflict_perp_spearman, p_perp_s = spearmanr(conflict_scores, perplexities)

    conflict_ent_pearson, p_ent = pearsonr(conflict_scores, entropies)
    conflict_ent_spearman, p_ent_s = spearmanr(conflict_scores, entropies)

    print("Conflict vs Perplexity:")
    print(f"  Pearson r = {conflict_perp_pearson:.3f} (p={p_perp:.4f})")
    print(f"  Spearman ρ = {conflict_perp_spearman:.3f} (p={p_perp_s:.4f})")
    if p_perp < 0.05:
        print(f"  ✅ Significant correlation! (p < 0.05)")
    else:
        print(f"  ⚠️  Not significant (p >= 0.05)")

    print(f"\nConflict vs Entropy:")
    print(f"  Pearson r = {conflict_ent_pearson:.3f} (p={p_ent:.4f})")
    print(f"  Spearman ρ = {conflict_ent_spearman:.3f} (p={p_ent_s:.4f})")
    if p_ent < 0.05:
        print(f"  ✅ Significant correlation! (p < 0.05)")
    else:
        print(f"  ⚠️  Not significant (p >= 0.05)")

    # Compare other dimensions too
    print(f"\n" + "="*70)
    print("COMPARISON WITH OTHER SNARC DIMENSIONS")
    print("="*70)
    print()

    for dimension in ['novelty', 'arousal', 'reward', 'total_salience']:
        dim_scores = np.array([r[dimension] for r in results])
        r_perp, p_perp = pearsonr(dim_scores, perplexities)
        r_ent, p_ent = pearsonr(dim_scores, entropies)

        print(f"{dimension.capitalize()}:")
        print(f"  vs Perplexity: r={r_perp:.3f} (p={p_perp:.4f})")
        print(f"  vs Entropy:    r={r_ent:.3f} (p={p_ent:.4f})")
        print()

    # Detailed breakdown
    print("="*70)
    print("DETAILED BREAKDOWN")
    print("="*70)
    print()

    print(f"{'Session':<8} {'Conflict':<10} {'Perplexity':<12} {'Entropy':<10} {'Salient':<8}")
    print("-"*70)

    for r in results:
        salient_mark = "✓" if r['salient'] else ""
        print(f"{r['session']:<8} {r['conflict']:<10.3f} {r['perplexity']:<12.3f} "
              f"{r['entropy']:<10.3f} {salient_mark:<8}")

    # Summary statistics
    print(f"\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print()

    # Group by salience
    salient = [r for r in results if r['salient']]
    non_salient = [r for r in results if not r['salient']]

    print("Salient Exchanges (n={})".format(len(salient)))
    if salient:
        print(f"  Conflict: {np.mean([r['conflict'] for r in salient]):.3f} ± {np.std([r['conflict'] for r in salient]):.3f}")
        print(f"  Perplexity: {np.mean([r['perplexity'] for r in salient]):.3f} ± {np.std([r['perplexity'] for r in salient]):.3f}")
        print(f"  Entropy: {np.mean([r['entropy'] for r in salient]):.3f} ± {np.std([r['entropy'] for r in salient]):.3f}")

    print(f"\nNon-Salient Exchanges (n={len(non_salient)})")
    if non_salient:
        print(f"  Conflict: {np.mean([r['conflict'] for r in non_salient]):.3f} ± {np.std([r['conflict'] for r in non_salient]):.3f}")
        print(f"  Perplexity: {np.mean([r['perplexity'] for r in non_salient]):.3f} ± {np.std([r['perplexity'] for r in non_salient]):.3f}")
        print(f"  Entropy: {np.mean([r['entropy'] for r in non_salient]):.3f} ± {np.std([r['entropy'] for r in non_salient]):.3f}")

    # Conclusions
    print(f"\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print()

    if abs(conflict_perp_pearson) > 0.5 or abs(conflict_ent_pearson) > 0.5:
        print("✅ HYPOTHESIS SUPPORTED:")
        print("   Conflict correlates with model uncertainty!")
        print("   SNARC is identifying epistemic frontiers.")
    elif abs(conflict_perp_pearson) > 0.3 or abs(conflict_ent_pearson) > 0.3:
        print("⚠️  HYPOTHESIS PARTIALLY SUPPORTED:")
        print("   Moderate correlation detected.")
        print("   Conflict shows some relationship with uncertainty.")
    else:
        print("❌ HYPOTHESIS NOT SUPPORTED:")
        print("   Weak or no correlation found.")
        print("   Conflict may measure something else.")

    print()

    return results, {
        'conflict_perp_pearson': conflict_perp_pearson,
        'conflict_perp_spearman': conflict_perp_spearman,
        'conflict_ent_pearson': conflict_ent_pearson,
        'conflict_ent_spearman': conflict_ent_spearman,
        'p_perp': p_perp,
        'p_ent': p_ent
    }


if __name__ == "__main__":
    results, stats = analyze_uncertainty_correlation()

    # Save results
    output = {
        'results': results,
        'statistics': stats
    }

    with open('conflict_uncertainty_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: conflict_uncertainty_results.json\n")
