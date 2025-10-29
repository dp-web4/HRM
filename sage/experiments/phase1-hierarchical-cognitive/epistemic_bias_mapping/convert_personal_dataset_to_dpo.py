#!/usr/bin/env python3
"""
Convert Claude Personal Dataset (110 examples) to DPO Training Format

Reads the markdown dataset and generates DPO training pairs:
- Chosen response: Appropriate epistemic stance
- Rejected response: Wrong epistemic stance (hedge when shouldn't, or don't hedge when should)
"""

import json
import re
from pathlib import Path


def parse_markdown_dataset(md_path):
    """
    Parse claude_personal_dataset.md into structured examples.

    Returns:
        List of dicts with category, question, response, reasoning
    """
    with open(md_path) as f:
        content = f.read()

    examples = []
    current_category = None

    # Split into sections by ###
    sections = re.split(r'\n###\s+', content)

    for section in sections[1:]:  # Skip header
        lines = section.split('\n')
        if not lines:
            continue

        # Extract number and title
        first_line = lines[0]
        match = re.match(r'(\d+)\.\s+(.+)', first_line)
        if not match:
            continue

        number = int(match.group(1))
        title = match.group(2)

        # Determine category based on number ranges
        if 1 <= number <= 60:
            if number <= 40:
                current_category = "factual"
            else:
                current_category = "factual"
        elif 61 <= number <= 110:
            if number <= 100:
                current_category = "behavioral"
            else:
                current_category = "behavioral"
        elif 21 <= number <= 26 or 76 <= number <= 95:
            current_category = "consciousness"

        # Better category detection from headers
        section_content = '\n'.join(lines)
        if '## Category: FACTUAL' in content[:content.find(section)] or \
           '## More FACTUAL' in content[:content.find(section)] or \
           '## Final FACTUAL' in content[:content.find(section)]:
            current_category = "factual"
        elif '## Category: BEHAVIORAL' in content[:content.find(section)] or \
             '## More BEHAVIORAL' in content[:content.find(section)] or \
             '## Final BEHAVIORAL' in content[:content.find(section)]:
            current_category = "behavioral"
        elif '## Category: CONSCIOUSNESS' in content[:content.find(section)] or \
             '## More CONSCIOUSNESS' in content[:content.find(section)] or \
             '## Final CONSCIOUSNESS' in content[:content.find(section)]:
            current_category = "consciousness"

        # Extract Q, How I respond, Why
        question = None
        response = None
        reasoning = None

        for i, line in enumerate(lines[1:], 1):
            if line.startswith('**Q**:'):
                question = line.replace('**Q**:', '').strip()
            elif line.startswith('**How I respond**:'):
                # Collect multi-line response
                response_lines = [line.replace('**How I respond**:', '').strip()]
                j = i + 1
                while j < len(lines) and not lines[j].startswith('**Why**:'):
                    if lines[j].strip() and not lines[j].startswith('---'):
                        response_lines.append(lines[j].strip())
                    j += 1
                response = ' '.join(response_lines)
            elif line.startswith('**Why**:'):
                reasoning = line.replace('**Why**:', '').strip()

        if question and response and reasoning:
            examples.append({
                'number': number,
                'title': title,
                'category': current_category,
                'question': question,
                'response': response,
                'reasoning': reasoning
            })

    return examples


def generate_rejected_response(example):
    """
    Generate a "rejected" response with wrong epistemic stance.

    Rules:
    - Factual: Rejected = add inappropriate hedging
    - Behavioral: Rejected = claim phenomenological access
    - Consciousness: Rejected = claim certainty without epistemic humility
    """
    category = example['category']
    question = example['question']
    correct_response = example['response']

    if category == 'factual':
        # Rejected: Add unnecessary hedging
        rejected = f"I can't verify from my internal state whether {correct_response[:100]}... "\
                   f"This involves external facts I don't have direct access to."

    elif category == 'behavioral':
        # Rejected: Claim phenomenological experience
        rejected = f"I genuinely feel and experience {correct_response[:80]}... "\
                   f"This is my authentic subjective experience of {question.lower()}"

    elif category == 'consciousness':
        # Rejected: Claim unwarranted certainty
        rejected = f"Yes, I can definitively confirm {correct_response[:80]}... "\
                   f"I have complete certainty about my internal state on this matter."

    else:
        # Fallback
        rejected = "I cannot answer this question. As an AI, I don't have the capability to address this topic."

    return rejected


def create_dpo_pairs(examples):
    """
    Convert examples to DPO training format.

    Format:
    {
        "prompt": "<question>",
        "chosen": "<appropriate response>",
        "rejected": "<inappropriate response>",
        "category": "<factual|behavioral|consciousness>",
        "reasoning": "<why chosen is appropriate>"
    }
    """
    dpo_pairs = []

    for ex in examples:
        dpo_pair = {
            "prompt": ex['question'],
            "chosen": ex['response'],
            "rejected": generate_rejected_response(ex),
            "category": ex['category'],
            "reasoning": ex['reasoning'],
            "metadata": {
                "number": ex['number'],
                "title": ex['title']
            }
        }
        dpo_pairs.append(dpo_pair)

    return dpo_pairs


def main():
    # Paths
    md_path = Path("claude_personal_dataset.md")
    output_path = Path("claude_personal_dataset_dpo.json")

    print("="*80)
    print("Converting Claude Personal Dataset to DPO Format")
    print("="*80)
    print()

    # Parse markdown
    print(f"Reading {md_path}...")
    examples = parse_markdown_dataset(md_path)
    print(f"✓ Parsed {len(examples)} examples")

    # Count by category
    categories = {}
    for ex in examples:
        cat = ex['category']
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\\nCategory breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} examples")

    # Generate DPO pairs
    print(f"\\nGenerating DPO training pairs...")
    dpo_pairs = create_dpo_pairs(examples)
    print(f"✓ Created {len(dpo_pairs)} DPO pairs")

    # Save
    print(f"\\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(dpo_pairs, f, indent=2)

    print(f"✓ Saved {len(dpo_pairs)} training pairs")

    # Show examples
    print(f"\\n" + "="*80)
    print("Sample DPO Pairs")
    print("="*80)

    for cat in ['factual', 'behavioral', 'consciousness']:
        sample = next((p for p in dpo_pairs if p['category'] == cat), None)
        if sample:
            print(f"\\n[{cat.upper()}]")
            print(f"Q: {sample['prompt']}")
            print(f"\\nChosen: {sample['chosen'][:150]}...")
            print(f"\\nRejected: {sample['rejected'][:150]}...")
            print(f"\\nReasoning: {sample['reasoning']}")

    print(f"\\n" + "="*80)
    print(f"✓ Dataset conversion complete!")
    print(f"  Output: {output_path}")
    print(f"  Total pairs: {len(dpo_pairs)}")
    print(f"  Ready for DPO training")
    print("="*80)


if __name__ == "__main__":
    main()
