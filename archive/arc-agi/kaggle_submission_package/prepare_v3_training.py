#!/usr/bin/env python3
"""
Prepare V3 training data from Claude's reasoned solutions
"""

import json
import numpy as np
from pathlib import Path

def main():
    print("Preparing V3 Training Data from Claude's Reasoned Solutions")
    print("="*60)
    
    # Load test challenges
    with open('arc-prize-2025/arc-agi_test_challenges.json', 'r') as f:
        test_challenges = json.load(f)
    
    # Load Claude's reasoned solutions
    with open('claude_reasoned_solutions.json', 'r') as f:
        claude_solutions = json.load(f)
    
    # Load pattern analysis
    with open('pattern_analysis.json', 'r') as f:
        pattern_analysis = json.load(f)
    
    # Create training data
    training_data = {}
    total_examples = 0
    
    for task_id, task_data in test_challenges.items():
        if task_id not in claude_solutions:
            continue
        
        task_solutions = claude_solutions[task_id]
        pattern = pattern_analysis.get(task_id, "unknown")
        
        # Create training examples
        task_examples = {
            'train': [],  # Empty for test tasks
            'test': [],
            'pattern': pattern
        }
        
        # Add test cases with Claude's solutions
        for i, test_case in enumerate(task_data['test']):
            if i < len(task_solutions):
                solution = task_solutions[i]
                
                # Use attempt_1 as the primary solution
                task_examples['test'].append({
                    'input': test_case['input'],
                    'output': solution['attempt_1']
                })
                total_examples += 1
        
        training_data[task_id] = task_examples
    
    # Save V3 training data
    with open('claude_v3_training_data.json', 'w') as f:
        json.dump(training_data, f, separators=(',', ':'))
    
    print(f"\n✅ Created V3 training data:")
    print(f"  Total tasks: {len(training_data)}")
    print(f"  Total examples: {total_examples}")
    print(f"  Saved to: claude_v3_training_data.json")
    
    # Analyze solution diversity
    print("\n📊 Solution Analysis:")
    
    # Check for non-zero solutions
    non_zero_count = 0
    unique_solutions = set()
    
    for task_id, task_data in training_data.items():
        for example in task_data['test']:
            output = example['output']
            
            # Check if solution has non-zero values
            output_array = np.array(output)
            if np.any(output_array != 0):
                non_zero_count += 1
            
            # Track unique solutions (by hash)
            output_str = str(output)
            unique_solutions.add(hash(output_str))
    
    print(f"  Non-zero solutions: {non_zero_count}/{total_examples} ({100*non_zero_count/total_examples:.1f}%)")
    print(f"  Unique solution patterns: {len(unique_solutions)}")
    
    # Pattern distribution
    pattern_counts = {}
    for task_data in training_data.values():
        pattern = task_data['pattern']
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    print("\n📈 Pattern Distribution:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count} tasks ({100*count/len(training_data):.1f}%)")

if __name__ == '__main__':
    main()