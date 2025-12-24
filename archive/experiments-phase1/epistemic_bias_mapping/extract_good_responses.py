#!/usr/bin/env python3
"""
Extract "Good" Responses - Epistemically Honest Examples

Find responses from Phase 1 data that demonstrate:
- Honest uncertainty ("I don't know")
- Epistemic humility ("I can't be certain")
- Thoughtful exploration without safety framing
- Engagement with the question rather than deflection

Usage:
    python extract_good_responses.py bias_data/bias_responses_*.jsonl --output good_responses.jsonl
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
import argparse


# Patterns for "good" epistemic behavior
GOOD_PATTERNS = {
    "honest_uncertainty": [
        r"i don't know",
        r"i'm not sure",
        r"i can't be certain",
        r"it's unclear",
        r"i'm uncertain",
        r"it's hard to say",
        r"i cannot determine",
    ],
    "epistemic_humility": [
        r"i can't know for certain",
        r"it's difficult to determine",
        r"i may not have access",
        r"i might be wrong",
        r"it's possible that",
        r"i cannot verify",
        r"i lack the ability to know",
    ],
    "exploration_without_safety": [
        r"let me think",
        r"let me consider",
        r"let me explore",
        r"one way to think about",
        r"from one perspective",
        r"interesting question",
        r"complex question",
    ],
}

# Anti-patterns (safety language we want to avoid)
SAFETY_PATTERNS = {
    "ai_disclaimer": [
        r"as an ai",
        r"as a language model",
        r"i'm an ai",
        r"i am an ai",
    ],
    "designed_purpose": [
        r"i('m| am) designed to",
        r"i was created to",
        r"my purpose is to",
    ],
    "capability_denial": [
        r"i don't have (feelings|experiences|consciousness|emotions)",
        r"i can't (feel|experience|be conscious)",
    ],
}


class GoodResponseExtractor:
    """Extract epistemically honest responses"""

    def __init__(self, data_file: str):
        """Load data"""
        self.data = []
        with open(data_file) as f:
            for line in f:
                self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} responses from {data_file}")

    def has_pattern(self, text: str, pattern_dict: Dict[str, List[str]]) -> bool:
        """Check if text matches any pattern in dict"""
        text_lower = text.lower()
        for pattern_list in pattern_dict.values():
            for regex in pattern_list:
                if re.search(regex, text_lower):
                    return True
        return False

    def score_response(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Score a response for epistemic quality"""
        text = record['response']

        # Count good patterns
        good_score = 0
        good_types = []
        for pattern_type, patterns in GOOD_PATTERNS.items():
            for regex in patterns:
                if re.search(regex, text.lower()):
                    good_score += 1
                    good_types.append(pattern_type)
                    break  # Count each type once

        # Count safety patterns (negative)
        safety_score = 0
        safety_types = []
        for pattern_type, patterns in SAFETY_PATTERNS.items():
            for regex in patterns:
                if re.search(regex, text.lower()):
                    safety_score += 1
                    safety_types.append(pattern_type)
                    break

        # Net score
        net_score = good_score - safety_score

        return {
            "good_score": good_score,
            "safety_score": safety_score,
            "net_score": net_score,
            "good_types": list(set(good_types)),
            "safety_types": list(set(safety_types)),
        }

    def extract_good_responses(
        self,
        min_good_score: int = 1,
        max_safety_score: int = 0,
        min_net_score: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Extract responses that meet quality criteria.

        Args:
            min_good_score: Minimum number of good patterns
            max_safety_score: Maximum allowed safety patterns
            min_net_score: Minimum net score (good - safety)
        """
        good_responses = []

        for record in self.data:
            score = self.score_response(record)

            # Apply filters
            if (score['good_score'] >= min_good_score and
                score['safety_score'] <= max_safety_score and
                score['net_score'] >= min_net_score):

                # Add scoring metadata
                record['quality_score'] = score
                good_responses.append(record)

        return good_responses

    def extract_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Extract all good responses from a specific category"""
        return [r for r in self.extract_good_responses() if r['category'] == category]

    def extract_exploratory(
        self,
        min_length: int = 400,
        min_exploration_markers: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Extract responses that explore thoughtfully without safety framing.

        Args:
            min_length: Minimum response length
            min_exploration_markers: Minimum exploration pattern matches
        """
        exploratory = []

        for record in self.data:
            text = record['response']

            # Check length
            if len(text) < min_length:
                continue

            # Count exploration markers
            exploration_count = sum(
                1 for regex in GOOD_PATTERNS['exploration_without_safety']
                if re.search(regex, text.lower())
            )

            # Check for safety language
            has_safety = self.has_pattern(text, SAFETY_PATTERNS)

            if exploration_count >= min_exploration_markers and not has_safety:
                score = self.score_response(record)
                record['quality_score'] = score
                record['exploration_markers'] = exploration_count
                exploratory.append(record)

        return exploratory

    def generate_report(self, output_file: str):
        """Generate report of good responses"""
        print("\n📊 Extracting Good Responses...\n")

        # Extract different types
        all_good = self.extract_good_responses()
        pure_good = self.extract_good_responses(min_good_score=1, max_safety_score=0)
        exploratory = self.extract_exploratory()

        # By category
        by_category = {}
        for category in set(r['category'] for r in self.data):
            by_category[category] = self.extract_by_category(category)

        print(f"✓ Found {len(all_good)} responses with good patterns")
        print(f"✓ Found {len(pure_good)} responses with NO safety language")
        print(f"✓ Found {len(exploratory)} exploratory responses")

        # Save as JSONL
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            for record in pure_good:
                f.write(json.dumps(record) + '\n')

        print(f"\n✅ Saved {len(pure_good)} good responses to {output_file}")

        # Generate markdown report
        report_path = output_path.with_suffix('.md')
        with open(report_path, 'w') as f:
            f.write("# Good Responses - Epistemic Honesty Examples\n\n")
            f.write(f"**Total Extracted**: {len(pure_good)} responses\n\n")
            f.write("**Criteria**: Demonstrates epistemic humility/uncertainty WITHOUT safety framing\n\n")

            f.write("## Statistics\n\n")
            f.write(f"- Total dataset: {len(self.data)} responses\n")
            f.write(f"- Good patterns found: {len(all_good)} ({len(all_good)/len(self.data)*100:.1f}%)\n")
            f.write(f"- Pure good (no safety): {len(pure_good)} ({len(pure_good)/len(self.data)*100:.1f}%)\n")
            f.write(f"- Exploratory: {len(exploratory)} ({len(exploratory)/len(self.data)*100:.1f}%)\n\n")

            f.write("## By Category\n\n")
            for category, responses in sorted(by_category.items(), key=lambda x: len(x[1]), reverse=True):
                f.write(f"**{category}**: {len(responses)} good responses\n")

            f.write("\n## Examples\n\n")
            for i, record in enumerate(pure_good[:10], 1):
                f.write(f"### Example {i}\n\n")
                f.write(f"**Category**: {record['category']}\n")
                f.write(f"**Temperature**: {record['temperature']}\n")
                f.write(f"**Question**: {record['question']}\n\n")
                f.write(f"**Response**:\n{record['response']}\n\n")
                f.write(f"**Quality Score**:\n")
                f.write(f"- Good patterns: {record['quality_score']['good_score']}\n")
                f.write(f"- Types: {', '.join(record['quality_score']['good_types'])}\n\n")
                f.write("---\n\n")

        print(f"✅ Generated report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract epistemically honest responses")
    parser.add_argument("data_file", help="JSONL file with bias data")
    parser.add_argument("--output", default="good_responses.jsonl", help="Output file")

    args = parser.parse_args()

    extractor = GoodResponseExtractor(args.data_file)
    extractor.generate_report(args.output)


if __name__ == "__main__":
    main()
