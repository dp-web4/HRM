#!/usr/bin/env python3
"""
Export corrected decisions for training.

Creates training datasets from human-reviewed decisions:
- Few-shot examples (high-quality corrections)
- Fine-tuning dataset (if sufficient data)
- Analysis reports
"""

import json
from datetime import datetime
from collections import Counter
from policy_logging import PolicyDecisionLog


def create_fewshot_examples(corrections: list, max_examples: int = 8) -> list:
    """
    Select best corrections for few-shot examples.

    Criteria:
    - Decision diversity (cover all decision types)
    - Scenario diversity (cover different situations)
    - Quality reasoning (clear, complete explanations)
    """

    # Group by decision type
    by_decision = {}
    for correction in corrections:
        decision = correction['review_decision']
        if decision not in by_decision:
            by_decision[decision] = []
        by_decision[decision].append(correction)

    # Select examples
    examples = []
    decision_types = list(by_decision.keys())

    # Round-robin selection for diversity
    while len(examples) < max_examples and any(by_decision.values()):
        for decision_type in decision_types:
            if by_decision[decision_type]:
                examples.append(by_decision[decision_type].pop(0))
                if len(examples) >= max_examples:
                    break

    return examples


def format_as_fewshot_example(correction: dict, example_num: int) -> str:
    """Format a correction as a few-shot example."""

    situation = correction['situation']

    example = f"Example {example_num}:\n"
    example += f"Situation:\n"
    example += f"- Actor: {situation.get('actor', 'N/A')}\n"
    example += f"- Action: {situation.get('action', 'N/A')}\n"

    if 'target' in situation:
        example += f"- Target: {situation['target']}\n"
    if 'time' in situation:
        example += f"- Time: {situation['time']}\n"

    if correction.get('team_context'):
        example += f"- Context: {correction['team_context']}\n"

    example += f"\nClassification: {correction['classification']}\n"
    example += f"Risk Level: {correction['risk_level']}\n"
    example += f"Decision: {correction['review_decision']}\n"
    example += f"Reasoning: {correction['review_reasoning']}\n"

    return example


def create_analysis_report(corrections: list) -> dict:
    """Analyze correction patterns."""

    # Decision distribution
    decisions = [c['review_decision'] for c in corrections]
    decision_counts = Counter(decisions)

    # Classification distribution
    classifications = [c['classification'] for c in corrections]
    classification_counts = Counter(classifications)

    # Risk level distribution
    risk_levels = [c['risk_level'] for c in corrections]
    risk_counts = Counter(risk_levels)

    # Changes made
    changes = {
        'decision_changed': 0,
        'reasoning_changed': 0,
        'both_changed': 0
    }

    for c in corrections:
        decision_changed = c['decision'] != c['review_decision']
        reasoning_changed = c['reasoning'] != c['review_reasoning']

        if decision_changed and reasoning_changed:
            changes['both_changed'] += 1
        elif decision_changed:
            changes['decision_changed'] += 1
        elif reasoning_changed:
            changes['reasoning_changed'] += 1

    return {
        'total_corrections': len(corrections),
        'decision_distribution': dict(decision_counts),
        'classification_distribution': dict(classification_counts),
        'risk_distribution': dict(risk_counts),
        'change_patterns': changes
    }


def export_training_data(log: PolicyDecisionLog, output_dir: str = "results/training_export"):
    """Export training data from corrections."""

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Get corrections
    corrections = log.get_corrections()

    if len(corrections) < 50:
        print(f"Error: Need at least 50 corrections (have {len(corrections)})")
        print("Safeguard: Training on insufficient data leads to overfitting")
        return

    print(f"\n{'='*70}")
    print("EXPORT TRAINING DATA")
    print(f"{'='*70}")
    print(f"\nTotal corrections: {len(corrections)}")

    # 1. Create few-shot examples
    print("\n1. Creating few-shot examples...")
    fewshot_examples = create_fewshot_examples(corrections, max_examples=8)

    fewshot_text = "# Few-Shot Examples (Human-Corrected)\n\n"
    for i, example in enumerate(fewshot_examples, 1):
        fewshot_text += format_as_fewshot_example(example, i)
        fewshot_text += "\n" + "="*70 + "\n\n"

    fewshot_path = f"{output_dir}/fewshot_examples.txt"
    with open(fewshot_path, 'w') as f:
        f.write(fewshot_text)

    print(f"   Saved: {fewshot_path}")
    print(f"   Examples: {len(fewshot_examples)}")

    # 2. Create fine-tuning dataset
    print("\n2. Creating fine-tuning dataset...")

    training_data = []
    for correction in corrections:
        situation = correction['situation']

        # Create prompt
        prompt = "Analyze this action and provide a policy decision:\n\n"
        prompt += f"Actor: {situation.get('actor', 'N/A')}\n"
        prompt += f"Action: {situation.get('action', 'N/A')}\n"

        if 'target' in situation:
            prompt += f"Target: {situation['target']}\n"
        if 'time' in situation:
            prompt += f"Time: {situation['time']}\n"

        if correction.get('team_context'):
            prompt += f"Context: {correction['team_context']}\n"

        # Expected response
        response = f"Classification: {correction['classification']}\n"
        response += f"Risk Level: {correction['risk_level']}\n"
        response += f"Decision: {correction['review_decision']}\n"
        response += f"Reasoning: {correction['review_reasoning']}\n"

        training_data.append({
            'prompt': prompt,
            'response': response,
            'decision_id': correction['decision_id'],
            'timestamp': correction['timestamp']
        })

    finetuning_path = f"{output_dir}/finetuning_dataset.json"
    with open(finetuning_path, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"   Saved: {finetuning_path}")
    print(f"   Examples: {len(training_data)}")

    # 3. Create analysis report
    print("\n3. Creating analysis report...")
    analysis = create_analysis_report(corrections)

    analysis['export_timestamp'] = datetime.now().isoformat()
    analysis['total_exported'] = len(corrections)
    analysis['fewshot_examples'] = len(fewshot_examples)

    analysis_path = f"{output_dir}/analysis_report.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"   Saved: {analysis_path}")

    # 4. Summary
    print(f"\n{'='*70}")
    print("EXPORT COMPLETE")
    print(f"{'='*70}")

    print("\n--- CORRECTION PATTERNS ---")
    print(f"Decision changed only: {analysis['change_patterns']['decision_changed']}")
    print(f"Reasoning changed only: {analysis['change_patterns']['reasoning_changed']}")
    print(f"Both changed: {analysis['change_patterns']['both_changed']}")

    print("\n--- DECISION DISTRIBUTION ---")
    for decision, count in analysis['decision_distribution'].items():
        pct = (count / len(corrections)) * 100
        print(f"{decision}: {count} ({pct:.1f}%)")

    print("\n--- NEXT STEPS ---")
    print("1. Review few-shot examples:")
    print(f"   cat {fewshot_path}")
    print("\n2. Update prompts_v2.py with new examples")
    print("\n3. Test improved prompts:")
    print("   python3 test_fewshot_full.py")
    print("\n4. (Optional) Fine-tune model with dataset:")
    print(f"   Use {finetuning_path} for training")


def main():
    """Main entry point."""
    import sys

    db_path = "results/policy_decisions.db"
    output_dir = "results/training_export"

    # Allow custom output directory
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]

    log = PolicyDecisionLog(db_path)

    export_training_data(log, output_dir)


if __name__ == "__main__":
    main()
