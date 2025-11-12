#!/usr/bin/env python3
"""
Claude's Visual ARC Solver
I'll use this to actually look at and solve each puzzle with reasoning
"""

import json
import numpy as np
from typing import List, Dict, Any

def load_data():
    """Load all necessary data files"""
    with open('arc-prize-2025/arc-agi_test_challenges.json', 'r') as f:
        test_challenges = json.load(f)
    
    with open('arc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
        train_challenges = json.load(f)
    
    with open('arc-prize-2025/arc-agi_training_solutions.json', 'r') as f:
        train_solutions = json.load(f)
    
    return test_challenges, train_challenges, train_solutions

def visualize_grid(grid, title=""):
    """Create a visual representation of the grid"""
    # Color mapping for better visualization
    color_map = {
        0: '⬛',  # Black (background)
        1: '🟦',  # Blue
        2: '🟥',  # Red
        3: '🟩',  # Green
        4: '🟨',  # Yellow
        5: '⬜',  # Grey/White
        6: '🟪',  # Magenta
        7: '🟧',  # Orange
        8: '🟦',  # Azure (using blue)
        9: '🟫'   # Brown/Maroon
    }
    
    print(f"\n{title}")
    print("=" * (len(grid[0]) * 2 if grid else 10))
    for row in grid:
        print(''.join(color_map.get(cell, str(cell)) for cell in row))
    print("=" * (len(grid[0]) * 2 if grid else 10))
    
    # Also show numeric grid
    print("\nNumeric:")
    for row in grid:
        print(' '.join(str(cell) for cell in row))

def find_pattern_in_training(test_input, train_challenges, train_solutions, task_id=None):
    """Find similar patterns in training data"""
    print("\n🔍 SEARCHING FOR SIMILAR PATTERNS IN TRAINING DATA...")
    
    # If we know the task ID (same ID in training), check that first
    if task_id and task_id in train_challenges:
        print(f"\n✓ Found same task ID '{task_id}' in training set!")
        task = train_challenges[task_id]
        
        print(f"  This task has {len(task['train'])} training examples")
        
        # Show first example
        if task['train']:
            example = task['train'][0]
            visualize_grid(example['input'], "Training Input Example:")
            visualize_grid(example['output'], "Training Output Example:")
            
            # Analyze the transformation
            in_shape = (len(example['input']), len(example['input'][0]))
            out_shape = (len(example['output']), len(example['output'][0]))
            
            print(f"\n📊 Transformation Analysis:")
            print(f"  Input shape: {in_shape}")
            print(f"  Output shape: {out_shape}")
            
            if in_shape != out_shape:
                if out_shape[0] > in_shape[0] or out_shape[1] > in_shape[1]:
                    print("  → Pattern: EXPANSION/TILING")
                else:
                    print("  → Pattern: EXTRACTION/CROPPING")
            
            return task['train']
    
    return None

def solve_with_reasoning(test_id, test_case, train_challenges, train_solutions):
    """Solve a test case with explicit reasoning"""
    print(f"\n{'='*60}")
    print(f"🧩 SOLVING: {test_id}")
    print(f"{'='*60}")
    
    test_input = test_case['input']
    visualize_grid(test_input, "TEST INPUT:")
    
    # Find training examples
    training_examples = find_pattern_in_training(test_input, train_challenges, train_solutions, test_id)
    
    # Analyze and solve
    solution = analyze_and_solve(test_input, training_examples, train_solutions, test_id)
    
    return solution

def analyze_and_solve(test_input, training_examples, train_solutions, task_id):
    """Analyze the pattern and generate solution"""
    
    # Default solution (will be replaced with actual reasoning)
    h, w = len(test_input), len(test_input[0]) if test_input else 1
    
    if training_examples:
        # Analyze all training examples to understand the pattern
        print("\n🔬 PATTERN ANALYSIS FROM TRAINING EXAMPLES:")
        
        for i, example in enumerate(training_examples):
            in_array = np.array(example['input'])
            out_array = np.array(example['output'])
            
            # Check various transformations
            if out_array.shape[0] == in_array.shape[0] * 3 and out_array.shape[1] == in_array.shape[1] * 3:
                print(f"  Example {i+1}: 3x3 TILING pattern detected")
                # Apply 3x3 tiling to test input
                test_array = np.array(test_input)
                solution = np.tile(test_array, (3, 3)).tolist()
                print("  ✓ Applying 3x3 tiling transformation")
                return solution
            
            elif out_array.shape[0] == in_array.shape[0] * 2 and out_array.shape[1] == in_array.shape[1] * 2:
                print(f"  Example {i+1}: 2x2 TILING pattern detected")
                # Apply 2x2 tiling to test input
                test_array = np.array(test_input)
                solution = np.tile(test_array, (2, 2)).tolist()
                print("  ✓ Applying 2x2 tiling transformation")
                return solution
    
    # If no pattern found, return input as-is (conservative approach)
    print("\n⚠ No clear pattern found - returning input as-is")
    return test_input

def main():
    """Main solving loop"""
    print("🤖 CLAUDE'S VISUAL ARC SOLVER")
    print("="*60)
    
    # Load data
    test_challenges, train_challenges, train_solutions = load_data()
    
    # Load existing solutions if any
    try:
        with open('claude_contextual_solutions.json', 'r') as f:
            claude_solutions = json.load(f)
    except:
        claude_solutions = {}
    
    # Solve puzzles
    test_ids = list(test_challenges.keys())
    
    # Process first 10 puzzles carefully
    for test_id in test_ids[:10]:
        if test_id in claude_solutions:
            print(f"\n⏭ Skipping {test_id} (already solved)")
            continue
        
        test_data = test_challenges[test_id]
        solutions = []
        
        for test_case in test_data['test']:
            solution = solve_with_reasoning(test_id, test_case, train_challenges, train_solutions)
            
            # Format for submission
            solutions.append({
                'attempt_1': solution,
                'attempt_2': solution  # Same for now
            })
        
        claude_solutions[test_id] = solutions
        
        # Save progress
        with open('claude_contextual_solutions.json', 'w') as f:
            json.dump(claude_solutions, f, indent=2)
        
        print(f"\n✅ Saved solution for {test_id}")
    
    print(f"\n{'='*60}")
    print(f"Completed {len(claude_solutions)} puzzles")

if __name__ == '__main__':
    main()