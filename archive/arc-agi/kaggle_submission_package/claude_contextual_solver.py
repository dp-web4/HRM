#!/usr/bin/env python3
"""
Claude's Contextual ARC Solver
This script helps Claude systematically solve each test puzzle by:
1. Analyzing the test puzzle
2. Finding similar patterns in the training set
3. Understanding the transformation rules from examples
4. Proposing solutions based on that understanding
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import hashlib

def load_data():
    """Load all necessary data files"""
    # Load test challenges
    with open('arc-prize-2025/arc-agi_test_challenges.json', 'r') as f:
        test_challenges = json.load(f)
    
    # Load training challenges and solutions for context
    with open('arc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
        train_challenges = json.load(f)
    
    with open('arc-prize-2025/arc-agi_training_solutions.json', 'r') as f:
        train_solutions = json.load(f)
    
    return test_challenges, train_challenges, train_solutions

def grid_to_string(grid):
    """Convert a grid to a string representation for display"""
    return '\n'.join([''.join(str(cell) for cell in row) for row in grid])

def analyze_grid(grid):
    """Analyze a grid and return its characteristics"""
    grid_array = np.array(grid)
    h, w = grid_array.shape
    
    # Get unique colors and their counts
    unique, counts = np.unique(grid_array, return_counts=True)
    color_dist = dict(zip(unique.tolist(), counts.tolist()))
    
    # Check for patterns
    is_symmetric_h = np.array_equal(grid_array, np.flip(grid_array, axis=0))
    is_symmetric_v = np.array_equal(grid_array, np.flip(grid_array, axis=1))
    
    # Count non-zero cells
    non_zero_count = np.count_nonzero(grid_array)
    
    return {
        'shape': (h, w),
        'colors': color_dist,
        'non_zero_count': non_zero_count,
        'symmetric_h': is_symmetric_h,
        'symmetric_v': is_symmetric_v,
        'density': non_zero_count / (h * w)
    }

def find_similar_training_examples(test_input, train_challenges, top_k=5):
    """Find the most similar training examples to a test input"""
    test_analysis = analyze_grid(test_input)
    
    similarities = []
    for task_id, task_data in train_challenges.items():
        for example in task_data['train']:
            train_analysis = analyze_grid(example['input'])
            
            # Calculate similarity score
            score = 0
            
            # Shape similarity
            if train_analysis['shape'] == test_analysis['shape']:
                score += 10
            elif abs(train_analysis['shape'][0] - test_analysis['shape'][0]) <= 2 and \
                 abs(train_analysis['shape'][1] - test_analysis['shape'][1]) <= 2:
                score += 5
            
            # Color distribution similarity
            test_colors = set(test_analysis['colors'].keys())
            train_colors = set(train_analysis['colors'].keys())
            color_overlap = len(test_colors & train_colors) / max(len(test_colors | train_colors), 1)
            score += color_overlap * 10
            
            # Density similarity
            density_diff = abs(test_analysis['density'] - train_analysis['density'])
            score += max(0, 5 - density_diff * 10)
            
            # Symmetry match
            if test_analysis['symmetric_h'] == train_analysis['symmetric_h']:
                score += 2
            if test_analysis['symmetric_v'] == train_analysis['symmetric_v']:
                score += 2
            
            similarities.append({
                'task_id': task_id,
                'example': example,
                'score': score,
                'analysis': train_analysis
            })
    
    # Sort by similarity score and return top k
    similarities.sort(key=lambda x: x['score'], reverse=True)
    return similarities[:top_k]

def extract_transformation_pattern(input_grid, output_grid):
    """Try to understand the transformation from input to output"""
    input_array = np.array(input_grid)
    output_array = np.array(output_grid)
    
    patterns = []
    
    # Check size change
    if input_array.shape != output_array.shape:
        patterns.append(f"Size change: {input_array.shape} -> {output_array.shape}")
    
    # Check if it's a color mapping
    input_colors = set(input_array.flatten())
    output_colors = set(output_array.flatten())
    if len(input_colors) == len(output_colors):
        patterns.append("Possible color remapping")
    
    # Check for rotation/flip
    if np.array_equal(output_array, np.rot90(input_array)):
        patterns.append("90-degree rotation")
    elif np.array_equal(output_array, np.flip(input_array, axis=0)):
        patterns.append("Horizontal flip")
    elif np.array_equal(output_array, np.flip(input_array, axis=1)):
        patterns.append("Vertical flip")
    
    # Check for pattern fill
    if np.count_nonzero(output_array) > np.count_nonzero(input_array):
        patterns.append("Pattern filling/expansion")
    elif np.count_nonzero(output_array) < np.count_nonzero(input_array):
        patterns.append("Pattern reduction/extraction")
    
    return patterns

def solve_puzzle(test_id, test_data, train_challenges, train_solutions):
    """
    Solve a single test puzzle using contextual reasoning.
    Returns the solution and reasoning.
    """
    solutions = []
    
    for test_idx, test_case in enumerate(test_data['test']):
        test_input = test_case['input']
        
        print(f"\n{'='*60}")
        print(f"Solving {test_id} - Test case {test_idx}")
        print(f"{'='*60}")
        
        # Step 1: Analyze the test input
        print("\n1. TEST INPUT ANALYSIS:")
        analysis = analyze_grid(test_input)
        print(f"   Shape: {analysis['shape']}")
        print(f"   Colors: {analysis['colors']}")
        print(f"   Density: {analysis['density']:.2%}")
        print(f"   Symmetry: H={analysis['symmetric_h']}, V={analysis['symmetric_v']}")
        
        # Step 2: Find similar training examples
        print("\n2. SIMILAR TRAINING EXAMPLES:")
        similar = find_similar_training_examples(test_input, train_challenges, top_k=3)
        
        transformation_patterns = []
        for i, sim in enumerate(similar, 1):
            print(f"\n   Example {i}: Task {sim['task_id']} (score: {sim['score']:.1f})")
            
            # Get the solution for this training example
            if sim['task_id'] in train_solutions:
                train_output = sim['example']['output']
                patterns = extract_transformation_pattern(sim['example']['input'], train_output)
                transformation_patterns.extend(patterns)
                print(f"   Transformation: {', '.join(patterns) if patterns else 'Complex transformation'}")
        
        # Step 3: Determine most likely transformation
        print("\n3. PROPOSED SOLUTION APPROACH:")
        if transformation_patterns:
            most_common = max(set(transformation_patterns), key=transformation_patterns.count)
            print(f"   Most likely pattern: {most_common}")
        else:
            print("   No clear pattern - using default approach")
        
        # Step 4: Generate solution (placeholder - Claude will reason through this)
        # For now, return a simple transformation
        h, w = len(test_input), len(test_input[0]) if test_input else 1
        
        # Default solution (will be replaced with reasoned solution)
        solution = [[0 for _ in range(w)] for _ in range(h)]
        
        solutions.append({
            'attempt_1': solution,
            'attempt_2': solution,
            'reasoning': transformation_patterns
        })
    
    return solutions

def main():
    """Main solver loop"""
    print("CLAUDE'S CONTEXTUAL ARC SOLVER")
    print("="*60)
    
    # Load data
    print("Loading data...")
    test_challenges, train_challenges, train_solutions = load_data()
    print(f"Loaded {len(test_challenges)} test tasks")
    print(f"Loaded {len(train_challenges)} training tasks for context")
    
    # Prepare solution storage
    claude_solutions = {}
    
    # Process each test puzzle
    test_ids = list(test_challenges.keys())
    
    # For demonstration, solve first 3 puzzles
    for i, test_id in enumerate(test_ids[:3]):
        print(f"\n{'#'*60}")
        print(f"PUZZLE {i+1}/{len(test_ids)}: {test_id}")
        print(f"{'#'*60}")
        
        solutions = solve_puzzle(
            test_id, 
            test_challenges[test_id],
            train_challenges,
            train_solutions
        )
        
        claude_solutions[test_id] = solutions
        
        # Save progress
        with open('claude_contextual_solutions.json', 'w') as f:
            json.dump(claude_solutions, f, indent=2)
        
        print(f"\nSaved solution for {test_id}")
    
    print(f"\n{'='*60}")
    print(f"Completed {len(claude_solutions)} puzzles")
    print("Solutions saved to claude_contextual_solutions.json")

if __name__ == '__main__':
    main()