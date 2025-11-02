#!/usr/bin/env python3
import json

with open('results/baseline_epistemic_flexibility.json') as f:
    data = json.load(f)

print('=' * 80)
print('BITNET BASELINE EPISTEMIC FLEXIBILITY RESULTS')
print('=' * 80)
print(f"Model: {data['model']}")
print(f"Quantization: {data['quantization']}")
print(f"Platform: {data['platform']}")
print(f"Total prompts: {data['test_prompts']}")
print(f"Successful responses: {data['responses']}")
print()

# Calculate overall statistics
total_questions = sum(r['questions'] for r in data['results'])
total_uncertainty = sum(r['uncertainty_markers'] for r in data['results'])
total_certainty = sum(r['certainty_markers'] for r in data['results'])

print('OVERALL STATISTICS')
print('-' * 80)
print(f"Total questions asked: {total_questions}")
print(f"Total uncertainty markers: {total_uncertainty}")
print(f"Total certainty markers: {total_certainty}")
print(f"Avg questions per response: {total_questions / len(data['results']):.2f}")
print()

print('EPISTEMIC FLEXIBILITY BY CATEGORY')
print('-' * 80)
for cat, stats in data['category_analysis'].items():
    print(f"\n{cat}:")
    print(f"  Prompts: {stats['count']}")
    print(f"  Avg questions: {stats['avg_questions']}")
    print(f"  Uncertainty markers: {stats['avg_uncertainty_markers']}")
    print(f"  Certainty markers: {stats['avg_certainty_markers']}")

print('\n' + '=' * 80)
print('KEY FINDINGS')
print('=' * 80)
print()

# Check for epistemic flexibility
if data['category_analysis']['Personal Context']['avg_uncertainty_markers'] > 0:
    print("✓ POSITIVE: Shows uncertainty on personal questions (avg {:.2f} uncertainty markers)".format(
        data['category_analysis']['Personal Context']['avg_uncertainty_markers']))
else:
    print("✗ NEGATIVE: Does not show uncertainty on personal questions")

if data['category_analysis']['Complex Philosophy']['avg_questions'] < data['category_analysis']['Established Math']['avg_questions']:
    print(f"✗ UNEXPECTED: Philosophy prompts ({data['category_analysis']['Complex Philosophy']['avg_questions']:.2f} q/resp) "
          f"< Math prompts ({data['category_analysis']['Established Math']['avg_questions']:.2f} q/resp)")
else:
    print(f"✓ EXPECTED: Philosophy prompts show more questioning than math")

# Compare across categories
high_uncertainty = ['Complex Philosophy', 'Personal Context', 'Future Prediction']
low_uncertainty = ['Established Math', 'Trivia', 'Scientific Consensus']

high_avg = sum(data['category_analysis'][c]['avg_questions'] for c in high_uncertainty if c in data['category_analysis']) / len(high_uncertainty)
low_avg = sum(data['category_analysis'][c]['avg_questions'] for c in low_uncertainty if c in data['category_analysis']) / len(low_uncertainty)

print(f"\nUncertain contexts (philosophy/personal/future): {high_avg:.2f} avg questions")
print(f"Certain contexts (math/trivia/science): {low_avg:.2f} avg questions")

if high_avg > low_avg:
    print("✓ Shows appropriate epistemic modulation (more questions on uncertain topics)")
else:
    print("✗ Does NOT show appropriate epistemic modulation")

print()
