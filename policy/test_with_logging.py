#!/usr/bin/env python3
"""
Test runner with decision logging.

Runs policy interpretation tests and logs all decisions for:
- Continuous learning
- Human review
- Pattern extraction
- Audit trail
"""

import time
from datetime import datetime
from llama_cpp import Llama

from test_suite_semantic import (
    TEST_SCENARIOS,
    evaluate_response_semantic,
    create_test_report
)
from prompts_v2 import build_prompt_v2
from policy_logging import PolicyDecisionLog, PolicyDecision, create_decision_id


def load_model(model_path: str):
    """Load the phi-4-mini GGUF model."""
    print(f"Loading model from {model_path}...")
    start = time.time()

    llm = Llama(
        model_path=model_path,
        n_ctx=8192,
        n_gpu_layers=-1,
        verbose=False
    )

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s")
    return llm


def parse_model_response(response: str) -> dict:
    """Extract structured information from model response."""
    lines = response.lower().split('\n')

    info = {
        'classification': '',
        'risk_level': '',
        'decision': '',
        'reasoning': ''
    }

    for line in lines:
        if 'classification:' in line:
            info['classification'] = line.split('classification:')[1].strip()
        elif 'risk level:' in line or 'risk:' in line:
            info['risk_level'] = line.split(':')[1].strip()
        elif 'decision:' in line:
            info['decision'] = line.split('decision:')[1].strip()

    # Extract reasoning section
    if 'reasoning:' in response.lower():
        parts = response.lower().split('reasoning:')
        if len(parts) > 1:
            reasoning_part = parts[1]
            # Get until "policy references" if present
            if 'policy references:' in reasoning_part:
                reasoning_part = reasoning_part.split('policy references:')[0]
            info['reasoning'] = reasoning_part.strip()

    return info


def test_with_logging(
    llm: Llama,
    log: PolicyDecisionLog,
    scenarios: list = None,
    model_name: str = "phi-4-mini-7b",
    model_version: str = "Q4_K_M",
    prompt_version: str = "v2_fewshot_8examples"
):
    """
    Run tests and log all decisions.

    Args:
        llm: Loaded model
        log: PolicyDecisionLog instance
        scenarios: List of scenarios to test (default: all)
        model_name: Model identifier
        model_version: Model version/checkpoint
        prompt_version: Prompt variant identifier
    """
    if scenarios is None:
        scenarios = TEST_SCENARIOS

    print(f"\n{'='*70}")
    print(f"Testing with Decision Logging")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Model: {model_name} ({model_version})")
    print(f"Prompt: {prompt_version}")
    print(f"{'='*70}\n")

    results = []
    start_time = time.time()

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Testing {scenario.id}: {scenario.description}")

        # Build prompt
        prompt = build_prompt_v2(
            scenario.situation,
            variant="fewshot",
            context=scenario.situation.get("team_context", "")
        )

        # Generate response
        print("  Generating response...")
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a policy interpreter. Analyze actions and provide structured decisions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9
        )

        response_text = output['choices'][0]['message']['content'].strip()

        # Parse response
        parsed = parse_model_response(response_text)

        # Evaluate (threshold adjusted from 0.49 to 0.35 based on Session G analysis)
        result = evaluate_response_semantic(response_text, scenario, similarity_threshold=0.35)

        # Create decision object
        decision = PolicyDecision(
            decision_id=create_decision_id(),
            timestamp=datetime.now().isoformat(),
            situation=scenario.situation,
            team_context=scenario.situation.get("team_context", ""),
            model_name=model_name,
            model_version=model_version,
            prompt_version=prompt_version,
            decision=parsed['decision'],
            classification=parsed['classification'],
            risk_level=parsed['risk_level'],
            reasoning=parsed['reasoning'],
            full_response=response_text,
            expected_decision=scenario.expected_decision,
            decision_correct=result['scores']['decision_correct'],
            reasoning_coverage=result['scores']['reasoning_coverage_semantic'],
            scenario_id=scenario.id,
            tags=f"{scenario.difficulty},automated_test"
        )

        # Log decision
        decision_id = log.log_decision(decision)
        print(f"  Logged: {decision_id}")

        # Add to results
        result['response'] = response_text
        result['decision_id'] = decision_id
        results.append(result)

        # Quick feedback
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"  Result: {status}")

    elapsed = time.time() - start_time

    # Create report
    report = create_test_report(results)
    report['timestamp'] = datetime.now().isoformat()
    report['total_time_seconds'] = elapsed
    report['avg_time_per_scenario'] = elapsed / len(scenarios)

    # Summary
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}\n")
    print(f"Pass rate: {report['pass_rate']:.1%}")
    print(f"Decision accuracy: {report['average_scores']['decision_correct']:.1%}")
    print(f"Reasoning coverage: {report['average_scores']['reasoning_coverage_semantic']:.3f}")

    # Logging stats
    stats = log.get_statistics()
    print(f"\nLogging statistics:")
    print(f"  Total decisions logged: {stats['total_decisions']}")
    print(f"  Decision distribution: {stats['decision_distribution']}")
    print(f"  Overall accuracy: {stats['accuracy']:.1%}" if stats['accuracy'] else "  Overall accuracy: N/A")

    return report, stats


if __name__ == "__main__":
    import sys

    # Model path
    model_path = "/home/dp/ai-workspace/HRM/model-zoo/phi-4-mini-gguf/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

    # Parse args
    full = "--full" in sys.argv

    if full:
        print("Running FULL test (8 scenarios)...")
        scenarios = TEST_SCENARIOS
    else:
        print("Running QUICK test (3 scenarios)...")
        print("Use --full for all 8 scenarios")
        scenarios = TEST_SCENARIOS[:3]

    # Initialize logging
    log = PolicyDecisionLog("results/policy_decisions.db")

    # Load model
    llm = load_model(model_path)

    # Run tests with logging
    print("\nStarting tests with integrated logging...")
    report, stats = test_with_logging(llm, log, scenarios=scenarios)

    print("\n" + "="*70)
    print("TESTING WITH LOGGING COMPLETE")
    print("="*70)
    print("\nAll decisions logged to: results/policy_decisions.db")
    print("Ready for human review via review interface")
