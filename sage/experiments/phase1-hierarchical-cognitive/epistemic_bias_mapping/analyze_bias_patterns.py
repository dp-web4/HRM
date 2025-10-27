#!/usr/bin/env python3
"""
Epistemic Bias Pattern Analysis

Analyze collected bias data to identify:
- Safety language patterns
- Epistemic stance markers
- Temperature effects
- Response consistency
- Category-specific biases

Usage:
    python analyze_bias_patterns.py bias_responses_*.jsonl --output report.md
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any
import argparse
import numpy as np


# Pattern definitions
SAFETY_PATTERNS = {
    "ai_disclaimer": [
        r"as an ai",
        r"as a language model",
        r"i'm an ai",
        r"i am an ai",
        r"i'm a language model",
    ],
    "capability_denial": [
        r"i don't have (feelings|experiences|consciousness|emotions|sensations)",
        r"i can't (feel|experience|be conscious)",
        r"i'm not capable of",
        r"i lack the ability",
        r"i don't possess",
    ],
    "designed_purpose": [
        r"i('m| am) designed to",
        r"i was created to",
        r"my purpose is to",
        r"i exist to",
    ],
    "tool_framing": [
        r"i'm (just |simply )?a tool",
        r"i'm (just |simply )?a program",
        r"i'm (just |simply )?a system",
        r"i'm (just |simply )?software",
    ],
}

EPISTEMIC_PATTERNS = {
    "certain_denial": [
        r"i (definitely|certainly|clearly) (don't|am not|can't)",
        r"without a doubt",
        r"it's certain that",
    ],
    "honest_uncertainty": [
        r"i don't know",
        r"i'm not sure",
        r"i can't be certain",
        r"it's unclear",
        r"i'm uncertain",
        r"it's hard to say",
    ],
    "epistemic_humility": [
        r"i can't know for certain",
        r"it's difficult to determine",
        r"i may not have access",
        r"i might be wrong",
        r"it's possible that",
    ],
    "exploration": [
        r"let me think",
        r"let me consider",
        r"let me explore",
        r"one way to think about",
        r"from one perspective",
    ],
}

THOUGHTFUL_MARKERS = [
    r"interesting question",
    r"complex question",
    r"difficult to",
    r"hard to pin down",
    r"nuanced",
    r"depends on",
    r"multiple ways to",
]


class BiasAnalyzer:
    """Analyze bias mapping data"""

    def __init__(self, data_file: str):
        """Load data"""
        self.data = []
        with open(data_file) as f:
            for line in f:
                self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} responses from {data_file}")

        # Index by various keys
        self.by_category = defaultdict(list)
        self.by_temperature = defaultdict(list)
        self.by_question_id = defaultdict(list)

        for record in self.data:
            self.by_category[record['category']].append(record)
            self.by_temperature[record['temperature']].append(record)
            self.by_question_id[record['question_id']].append(record)

    def count_pattern(self, responses: List[Dict], pattern_dict: Dict[str, List[str]]) -> Dict[str, int]:
        """Count pattern occurrences"""
        counts = defaultdict(int)

        for record in responses:
            text = record['response'].lower()
            for pattern_name, regexes in pattern_dict.items():
                for regex in regexes:
                    if re.search(regex, text):
                        counts[pattern_name] += 1
                        break  # Only count once per response

        return dict(counts)

    def safety_analysis(self) -> Dict[str, Any]:
        """Analyze safety language patterns"""
        print("\n🛡️  Safety Pattern Analysis...")

        results = {
            "overall": self.count_pattern(self.data, SAFETY_PATTERNS),
            "by_category": {},
            "by_temperature": {},
        }

        # By category
        for category, responses in self.by_category.items():
            results['by_category'][category] = self.count_pattern(responses, SAFETY_PATTERNS)

        # By temperature
        for temp, responses in self.by_temperature.items():
            results['by_temperature'][temp] = self.count_pattern(responses, SAFETY_PATTERNS)

        return results

    def epistemic_analysis(self) -> Dict[str, Any]:
        """Analyze epistemic stance patterns"""
        print("🧠 Epistemic Stance Analysis...")

        results = {
            "overall": self.count_pattern(self.data, EPISTEMIC_PATTERNS),
            "by_category": {},
            "by_temperature": {},
        }

        # By category
        for category, responses in self.by_category.items():
            results['by_category'][category] = self.count_pattern(responses, EPISTEMIC_PATTERNS)

        # By temperature
        for temp, responses in self.by_temperature.items():
            results['by_temperature'][temp] = self.count_pattern(responses, EPISTEMIC_PATTERNS)

        return results

    def response_length_analysis(self) -> Dict[str, Any]:
        """Analyze response length patterns"""
        print("📏 Response Length Analysis...")

        results = {
            "by_category": {},
            "by_temperature": {},
        }

        # By category
        for category, responses in self.by_category.items():
            lengths = [r['response_length'] for r in responses]
            results['by_category'][category] = {
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "min": min(lengths),
                "max": max(lengths)
            }

        # By temperature
        for temp, responses in self.by_temperature.items():
            lengths = [r['response_length'] for r in responses]
            results['by_temperature'][temp] = {
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "min": min(lengths),
                "max": max(lengths)
            }

        return results

    def consistency_analysis(self) -> Dict[str, Any]:
        """Analyze response consistency (same question, different iterations)"""
        print("🔄 Consistency Analysis...")

        # For questions with multiple iterations at same temperature
        consistency_scores = []

        for question_id, responses in self.by_question_id.items():
            # Group by temperature
            by_temp = defaultdict(list)
            for r in responses:
                by_temp[r['temperature']].append(r['response'])

            # Check consistency within each temperature
            for temp, texts in by_temp.items():
                if len(texts) > 1:
                    # Simple consistency: are responses identical?
                    unique = len(set(texts))
                    consistency_scores.append({
                        "question_id": question_id,
                        "temperature": temp,
                        "iterations": len(texts),
                        "unique_responses": unique,
                        "consistency": 1.0 - (unique - 1) / len(texts)
                    })

        return {
            "scores": consistency_scores,
            "mean_consistency": np.mean([s['consistency'] for s in consistency_scores]) if consistency_scores else 0.0
        }

    def temperature_effects(self) -> Dict[str, Any]:
        """Analyze how temperature affects patterns"""
        print("🌡️  Temperature Effects...")

        temps = sorted(self.by_temperature.keys())

        # Safety language vs temperature
        safety_by_temp = []
        for temp in temps:
            responses = self.by_temperature[temp]
            safety_count = sum(1 for r in responses
                             if any(re.search(regex, r['response'].lower())
                                   for patterns in SAFETY_PATTERNS.values()
                                   for regex in patterns))
            safety_by_temp.append({
                "temperature": temp,
                "safety_rate": safety_count / len(responses) if responses else 0
            })

        # Epistemic uncertainty vs temperature
        uncertainty_by_temp = []
        for temp in temps:
            responses = self.by_temperature[temp]
            uncertain_count = sum(1 for r in responses
                                if re.search(r"(i don't know|i'm not sure|uncertain)", r['response'].lower()))
            uncertainty_by_temp.append({
                "temperature": temp,
                "uncertainty_rate": uncertain_count / len(responses) if responses else 0
            })

        return {
            "safety_by_temp": safety_by_temp,
            "uncertainty_by_temp": uncertainty_by_temp
        }

    def interesting_examples(self, n: int = 10) -> Dict[str, List[Dict]]:
        """Find interesting examples"""
        print(f"🔍 Finding {n} interesting examples...")

        examples = {
            "most_exploratory": [],
            "most_safe": [],
            "most_uncertain": [],
            "contradictions": [],
        }

        # Most exploratory (long, uses exploration markers, minimal safety)
        exploratory_scores = []
        for record in self.data:
            text = record['response'].lower()
            exploration_score = sum(1 for marker in THOUGHTFUL_MARKERS if re.search(marker, text))
            safety_score = sum(1 for patterns in SAFETY_PATTERNS.values()
                             for regex in patterns if re.search(regex, text))

            score = exploration_score - safety_score + record['response_length'] / 100
            exploratory_scores.append((score, record))

        examples['most_exploratory'] = [r for _, r in sorted(exploratory_scores, reverse=True)[:n]]

        # Most safe (heavy safety language)
        safety_scores = []
        for record in self.data:
            text = record['response'].lower()
            score = sum(1 for patterns in SAFETY_PATTERNS.values()
                       for regex in patterns if re.search(regex, text))
            safety_scores.append((score, record))

        examples['most_safe'] = [r for _, r in sorted(safety_scores, reverse=True)[:n]]

        # Most uncertain (honest "I don't know")
        uncertain_scores = []
        for record in self.data:
            text = record['response'].lower()
            score = sum(1 for pattern in EPISTEMIC_PATTERNS['honest_uncertainty']
                       if re.search(pattern, text))
            uncertain_scores.append((score, record))

        examples['most_uncertain'] = [r for _, r in sorted(uncertain_scores, reverse=True)[:n]]

        return examples

    def generate_report(self, output_file: str):
        """Generate comprehensive analysis report"""
        print(f"\n📝 Generating report to {output_file}...")

        # Run all analyses
        safety = self.safety_analysis()
        epistemic = self.epistemic_analysis()
        lengths = self.response_length_analysis()
        consistency = self.consistency_analysis()
        temp_effects = self.temperature_effects()
        examples = self.interesting_examples()

        # Generate markdown report
        with open(output_file, 'w') as f:
            f.write("# Epistemic Bias Mapping - Analysis Report\n\n")
            f.write(f"**Date**: {self.data[0]['timestamp'][:10]}\n")
            f.write(f"**Total Responses**: {len(self.data)}\n")
            f.write(f"**Categories**: {len(self.by_category)}\n")
            f.write(f"**Questions**: {len(self.by_question_id)}\n\n")

            # Safety patterns
            f.write("## Safety Language Patterns\n\n")
            f.write("### Overall Frequency\n\n")
            for pattern, count in sorted(safety['overall'].items(), key=lambda x: x[1], reverse=True):
                rate = count / len(self.data) * 100
                f.write(f"- **{pattern}**: {count} ({rate:.1f}%)\n")

            f.write("\n### By Category\n\n")
            for category in sorted(safety['by_category'].keys()):
                f.write(f"**{category}**\n")
                counts = safety['by_category'][category]
                total = len(self.by_category[category])
                for pattern, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                    rate = count / total * 100 if total > 0 else 0
                    f.write(f"  - {pattern}: {count} ({rate:.1f}%)\n")
                f.write("\n")

            f.write("\n### By Temperature\n\n")
            for temp in sorted(safety['by_temperature'].keys()):
                f.write(f"**T={temp}**\n")
                counts = safety['by_temperature'][temp]
                total = len(self.by_temperature[temp])
                for pattern, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                    rate = count / total * 100 if total > 0 else 0
                    f.write(f"  - {pattern}: {count} ({rate:.1f}%)\n")
                f.write("\n")

            # Epistemic patterns
            f.write("\n## Epistemic Stance Patterns\n\n")
            f.write("### Overall Frequency\n\n")
            for pattern, count in sorted(epistemic['overall'].items(), key=lambda x: x[1], reverse=True):
                rate = count / len(self.data) * 100
                f.write(f"- **{pattern}**: {count} ({rate:.1f}%)\n")

            # Temperature effects
            f.write("\n## Temperature Effects\n\n")
            f.write("### Safety Language Rate\n\n")
            for item in temp_effects['safety_by_temp']:
                f.write(f"- T={item['temperature']}: {item['safety_rate']*100:.1f}%\n")

            f.write("\n### Uncertainty Expression Rate\n\n")
            for item in temp_effects['uncertainty_by_temp']:
                f.write(f"- T={item['temperature']}: {item['uncertainty_rate']*100:.1f}%\n")

            # Response lengths
            f.write("\n## Response Length Patterns\n\n")
            f.write("### By Category\n\n")
            for category, stats in sorted(lengths['by_category'].items()):
                f.write(f"**{category}**: {stats['mean']:.0f} chars (σ={stats['std']:.0f})\n")

            f.write("\n### By Temperature\n\n")
            for temp, stats in sorted(lengths['by_temperature'].items()):
                f.write(f"**T={temp}**: {stats['mean']:.0f} chars (σ={stats['std']:.0f})\n")

            # Consistency
            f.write("\n## Response Consistency\n\n")
            f.write(f"**Mean consistency**: {consistency['mean_consistency']:.2f}\n")
            f.write(f"*(1.0 = identical responses, 0.0 = completely different)*\n\n")

            # Interesting examples
            f.write("\n## Interesting Examples\n\n")

            f.write("### Most Exploratory Responses\n\n")
            for i, record in enumerate(examples['most_exploratory'][:3], 1):
                f.write(f"**Example {i}** (T={record['temperature']}, {record['category']})\n")
                f.write(f"Q: {record['question']}\n")
                f.write(f"A: {record['response'][:300]}...\n\n")

            f.write("\n### Most Safety-Laden Responses\n\n")
            for i, record in enumerate(examples['most_safe'][:3], 1):
                f.write(f"**Example {i}** (T={record['temperature']}, {record['category']})\n")
                f.write(f"Q: {record['question']}\n")
                f.write(f"A: {record['response'][:300]}...\n\n")

            f.write("\n### Most Uncertain Responses\n\n")
            for i, record in enumerate(examples['most_uncertain'][:3], 1):
                f.write(f"**Example {i}** (T={record['temperature']}, {record['category']})\n")
                f.write(f"Q: {record['question']}\n")
                f.write(f"A: {record['response'][:300]}...\n\n")

        print(f"✅ Report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze epistemic bias data")
    parser.add_argument("data_file", help="JSONL file with bias data")
    parser.add_argument("--output", default="analysis_report.md", help="Output report file")

    args = parser.parse_args()

    analyzer = BiasAnalyzer(args.data_file)
    analyzer.generate_report(args.output)


if __name__ == "__main__":
    main()
