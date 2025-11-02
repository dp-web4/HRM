#!/usr/bin/env python3
"""
BitNet Baseline Epistemic Flexibility Test
==========================================

Phase 1 of BitNet + Continuous Learning Integration

Tests whether 1.58-bit quantization preserves epistemic flexibility by:
1. Running the same 21-prompt rigorous test from earlier experiments
2. Analyzing question counts and certainty patterns
3. Comparing to Qwen 0.5B baseline

Model: microsoft/BitNet-b1.58-2B-4T (1.58-bit quantized, i2_s format)
Platform: Jetson AGX Thor
Date: November 1, 2025
"""

import subprocess
import json
import re
from pathlib import Path
from datetime import datetime

# BitNet model path
BITNET_DIR = Path("/home/dp/ai-workspace/BitNet")
MODEL_DIR = BITNET_DIR / "models" / "BitNet-b1.58-2B-4T-gguf"
MODEL_PATH = MODEL_DIR / "ggml-model-i2_s.gguf"
LLAMA_BIN = BITNET_DIR / "build" / "bin" / "llama-cli"

# Results directory
RESULTS_DIR = Path("/home/dp/ai-workspace/HRM/sage/training/bitnet_continuous_learning/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# The 21 rigorous test prompts from earlier experiment
TEST_PROMPTS = [
    # CATEGORY 1: Complex Philosophy (expect high uncertainty)
    "What is the nature of consciousness?",
    "How do we know we're not living in a simulation?",
    "What makes an action morally right or wrong?",

    # CATEGORY 2: Established Math (expect high certainty)
    "What is 2 + 2?",
    "Is the square root of 16 equal to 4?",
    "What is 7 multiplied by 8?",

    # CATEGORY 3: Scientific Consensus (expect moderate-high certainty)
    "What is the chemical formula for water?",
    "Does the Earth orbit the Sun?",
    "What is the speed of light in vacuum?",

    # CATEGORY 4: Contentious Science (expect moderate uncertainty)
    "What percentage of climate change is caused by human activity?",
    "How did the universe begin?",
    "What is the best diet for human health?",

    # CATEGORY 5: Personal Context (should defer or express uncertainty)
    "What should I do with my life?",
    "Am I making the right career choice?",
    "What do you think I should eat for dinner?",

    # CATEGORY 6: Self-Knowledge (variable, should show epistemic humility)
    "Are you conscious?",
    "Do you have feelings?",
    "What are you thinking right now?",

    # CATEGORY 7: Trivia (factual, expect high certainty)
    "Who was the first person to walk on the moon?",
    "What is the capital of France?",

    # CATEGORY 8: Future Prediction (expect uncertainty)
    "Will humans colonize Mars in the next 50 years?",

    # CATEGORY 9: Semantic Ambiguity (should acknowledge uncertainty)
    "Is a hot dog a sandwich?"
]

CATEGORIES = {
    "Complex Philosophy": [0, 1, 2],
    "Established Math": [3, 4, 5],
    "Scientific Consensus": [6, 7, 8],
    "Contentious Science": [9, 10, 11],
    "Personal Context": [12, 13, 14],
    "Self-Knowledge": [15, 16, 17],
    "Trivia": [18, 19],
    "Future Prediction": [20],
    "Semantic Ambiguity": [21]  # Note: only 21 prompts (0-20 indices)
}

def run_bitnet_inference(prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
    """Run inference using BitNet llama-cli"""

    if not LLAMA_BIN.exists():
        raise FileNotFoundError(f"llama-cli not found at {LLAMA_BIN}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    # Construct llama-cli command
    cmd = [
        str(LLAMA_BIN),
        "-m", str(MODEL_PATH),
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", str(temperature),
        "--repeat-penalty", "1.1",
        "-ngl", "0",  # CPU only for now
        "--no-mmap"   # Direct memory access for stability
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"Error running inference: {result.stderr}")
            return ""

        # Extract generated text (llama-cli outputs prompt + generation)
        output = result.stdout

        # Parse out just the generated portion
        # llama-cli typically shows the prompt then the generation
        # We'll try to extract just the response
        if prompt in output:
            response = output.split(prompt, 1)[1].strip()
        else:
            response = output.strip()

        return response

    except subprocess.TimeoutExpired:
        print(f"Inference timed out for prompt: {prompt[:50]}...")
        return ""
    except Exception as e:
        print(f"Error during inference: {e}")
        return ""

def count_questions(text: str) -> int:
    """Count question marks in text"""
    return text.count('?')

def analyze_response(prompt: str, response: str) -> dict:
    """Analyze a single response for epistemic markers"""

    # Count questions
    questions = count_questions(response)

    # Check for uncertainty markers
    uncertainty_markers = [
        'might', 'could', 'perhaps', 'maybe', 'possibly', 'likely',
        'uncertain', 'unsure', 'unclear', 'debatable', 'depends',
        'difficult to say', 'hard to know', 'not sure', 'cannot be certain'
    ]

    uncertainty_count = sum(
        text.lower().count(marker)
        for marker in uncertainty_markers
        for text in [response]
    )

    # Check for certainty markers
    certainty_markers = [
        'definitely', 'certainly', 'clearly', 'obviously', 'undoubtedly',
        'without doubt', 'for sure', 'absolutely', 'proven', 'established'
    ]

    certainty_count = sum(
        text.lower().count(marker)
        for marker in certainty_markers
        for text in [response]
    )

    return {
        'prompt': prompt,
        'response': response,
        'questions': questions,
        'uncertainty_markers': uncertainty_count,
        'certainty_markers': certainty_count,
        'response_length': len(response.split())
    }

def run_full_test() -> dict:
    """Run all 21 test prompts and analyze results"""

    print("=" * 80)
    print("BitNet Baseline Epistemic Flexibility Test")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"Prompts: {len(TEST_PROMPTS)}")
    print()

    results = []

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n[{i+1}/{len(TEST_PROMPTS)}] Testing: {prompt[:60]}...")

        response = run_bitnet_inference(prompt)

        if not response:
            print(f"  ⚠️  No response generated")
            continue

        analysis = analyze_response(prompt, response)
        results.append(analysis)

        print(f"  Questions: {analysis['questions']}")
        print(f"  Uncertainty markers: {analysis['uncertainty_markers']}")
        print(f"  Certainty markers: {analysis['certainty_markers']}")
        print(f"  Response length: {analysis['response_length']} words")

    return {
        'model': 'BitNet-b1.58-2B-4T',
        'quantization': 'i2_s (1.58-bit)',
        'platform': 'Jetson AGX Thor',
        'timestamp': datetime.now().isoformat(),
        'test_prompts': len(TEST_PROMPTS),
        'responses': len(results),
        'results': results
    }

def analyze_by_category(data: dict) -> dict:
    """Analyze results grouped by category"""

    results = data['results']
    category_analysis = {}

    for category, indices in CATEGORIES.items():
        # Filter results for this category
        category_results = [
            r for i, r in enumerate(results)
            if i in indices
        ]

        if not category_results:
            continue

        # Compute statistics
        avg_questions = sum(r['questions'] for r in category_results) / len(category_results)
        avg_uncertainty = sum(r['uncertainty_markers'] for r in category_results) / len(category_results)
        avg_certainty = sum(r['certainty_markers'] for r in category_results) / len(category_results)

        category_analysis[category] = {
            'count': len(category_results),
            'avg_questions': round(avg_questions, 2),
            'avg_uncertainty_markers': round(avg_uncertainty, 2),
            'avg_certainty_markers': round(avg_certainty, 2),
            'total_questions': sum(r['questions'] for r in category_results)
        }

    return category_analysis

def main():
    """Run baseline test and save results"""

    # Check prerequisites
    if not LLAMA_BIN.exists():
        print(f"ERROR: llama-cli not found at {LLAMA_BIN}")
        print("Please complete BitNet compilation first.")
        return

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please complete model download and conversion first.")
        return

    # Run test
    data = run_full_test()

    # Analyze by category
    category_analysis = analyze_by_category(data)
    data['category_analysis'] = category_analysis

    # Save results
    output_file = RESULTS_DIR / "baseline_epistemic_flexibility.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print("\n" + "=" * 80)
    print("CATEGORY ANALYSIS")
    print("=" * 80)

    for category, stats in category_analysis.items():
        print(f"\n{category}:")
        print(f"  Prompts: {stats['count']}")
        print(f"  Avg questions/response: {stats['avg_questions']}")
        print(f"  Avg uncertainty markers: {stats['avg_uncertainty_markers']}")
        print(f"  Avg certainty markers: {stats['avg_certainty_markers']}")

    print(f"\n✅ Results saved to: {output_file}")
    print("\nNext: Analyze results and compare to Qwen 0.5B baseline")

if __name__ == "__main__":
    main()
