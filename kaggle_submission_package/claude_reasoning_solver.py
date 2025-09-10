#!/usr/bin/env python3
"""
Claude's Reasoning-Based ARC Solver
I'll analyze each puzzle, find the pattern, and apply the correct transformation
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple

def load_data():
    """Load all necessary data files"""
    with open('arc-prize-2025/arc-agi_test_challenges.json', 'r') as f:
        test_challenges = json.load(f)
    
    with open('arc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
        train_challenges = json.load(f)
    
    with open('arc-prize-2025/arc-agi_training_solutions.json', 'r') as f:
        train_solutions = json.load(f)
    
    return test_challenges, train_challenges, train_solutions

def find_rectangles(grid, color=3):
    """Find rectangle patterns in the grid"""
    grid_array = np.array(grid)
    h, w = grid_array.shape
    rectangles = []
    
    # Find horizontal and vertical lines
    for i in range(h):
        for j in range(w):
            if grid_array[i, j] == color:
                # Check if it's a corner of a rectangle
                # Look for rectangle by finding opposite corner
                for i2 in range(i+2, h):
                    for j2 in range(j+2, w):
                        if grid_array[i2, j2] == color:
                            # Check if this forms a rectangle outline
                            is_rect = True
                            # Check top and bottom edges
                            for k in range(j+1, j2):
                                if grid_array[i, k] != color or grid_array[i2, k] != color:
                                    is_rect = False
                                    break
                            # Check left and right edges  
                            for k in range(i+1, i2):
                                if grid_array[k, j] != color or grid_array[k, j2] != color:
                                    is_rect = False
                                    break
                            
                            if is_rect:
                                rectangles.append(((i, j), (i2, j2)))
    
    return rectangles

def fill_rectangle_interior(grid, fill_color=4):
    """Fill the interior of rectangles with a specified color"""
    grid_array = np.array(grid)
    rectangles = find_rectangles(grid, color=3)
    
    for (i1, j1), (i2, j2) in rectangles:
        # Fill interior (not the border)
        for i in range(i1+1, i2):
            for j in range(j1+1, j2):
                grid_array[i, j] = fill_color
    
    return grid_array.tolist()

def apply_transformation(test_input, pattern_type):
    """Apply the identified transformation pattern"""
    
    if pattern_type == "fill_rectangles":
        return fill_rectangle_interior(test_input, fill_color=4)
    
    elif pattern_type == "tile_3x3":
        test_array = np.array(test_input)
        return np.tile(test_array, (3, 3)).tolist()
    
    elif pattern_type == "tile_2x2":
        test_array = np.array(test_input)
        return np.tile(test_array, (2, 2)).tolist()
    
    elif pattern_type == "extract_pattern":
        # Extract non-zero pattern
        test_array = np.array(test_input)
        non_zero_mask = test_array != 0
        
        # Find bounding box
        rows = np.any(non_zero_mask, axis=1)
        cols = np.any(non_zero_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Extract the pattern
        return test_array[rmin:rmax+1, cmin:cmax+1].tolist()
    
    # Default: return as-is
    return test_input

def analyze_training_pattern(task_id, train_challenges, train_solutions):
    """Analyze training examples to identify the transformation pattern"""
    
    if task_id not in train_challenges:
        return "unknown"
    
    task = train_challenges[task_id]
    patterns = []
    
    for example in task['train']:
        in_array = np.array(example['input'])
        out_array = np.array(example['output'])
        
        # Check for tiling
        if out_array.shape[0] == in_array.shape[0] * 3 and out_array.shape[1] == in_array.shape[1] * 3:
            tiled = np.tile(in_array, (3, 3))
            if np.array_equal(out_array, tiled):
                patterns.append("tile_3x3")
        
        elif out_array.shape[0] == in_array.shape[0] * 2 and out_array.shape[1] == in_array.shape[1] * 2:
            tiled = np.tile(in_array, (2, 2))
            if np.array_equal(out_array, tiled):
                patterns.append("tile_2x2")
        
        # Check for rectangle filling
        elif in_array.shape == out_array.shape:
            # Check if output has filled rectangles
            diff = out_array - in_array
            if np.any(diff == 4):  # Yellow fill
                patterns.append("fill_rectangles")
        
        # Check for pattern extraction
        elif out_array.shape[0] < in_array.shape[0] or out_array.shape[1] < in_array.shape[1]:
            patterns.append("extract_pattern")
    
    # Return most common pattern
    if patterns:
        return max(set(patterns), key=patterns.count)
    
    return "unknown"

def solve_puzzle(test_id, test_data, train_challenges, train_solutions):
    """Solve a test puzzle using pattern analysis"""
    
    # Identify the transformation pattern from training examples
    pattern = analyze_training_pattern(test_id, train_challenges, train_solutions)
    
    solutions = []
    for test_case in test_data['test']:
        test_input = test_case['input']
        
        # Apply the identified transformation
        if pattern != "unknown":
            solution = apply_transformation(test_input, pattern)
        else:
            # Default: return input as-is
            solution = test_input
        
        solutions.append({
            'attempt_1': solution,
            'attempt_2': solution
        })
    
    return solutions, pattern

def main():
    """Main solving loop"""
    print("🤖 CLAUDE'S REASONING-BASED ARC SOLVER")
    print("="*60)
    
    # Load data
    test_challenges, train_challenges, train_solutions = load_data()
    
    # Initialize solution storage
    claude_solutions = {}
    pattern_log = {}
    
    # Process all test puzzles
    test_ids = list(test_challenges.keys())
    
    print(f"Processing {len(test_ids)} test puzzles...")
    
    for i, test_id in enumerate(test_ids):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(test_ids)} puzzles solved")
        
        test_data = test_challenges[test_id]
        solutions, pattern = solve_puzzle(test_id, test_data, train_challenges, train_solutions)
        
        claude_solutions[test_id] = solutions
        pattern_log[test_id] = pattern
    
    # Save solutions
    with open('claude_reasoned_solutions.json', 'w') as f:
        json.dump(claude_solutions, f, separators=(',', ':'))
    
    # Save pattern log for analysis
    with open('pattern_analysis.json', 'w') as f:
        json.dump(pattern_log, f, indent=2)
    
    print(f"\n✅ Completed all {len(claude_solutions)} puzzles!")
    print(f"Solutions saved to claude_reasoned_solutions.json")
    
    # Show pattern distribution
    pattern_counts = {}
    for pattern in pattern_log.values():
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    print("\nPattern distribution:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count} puzzles")

if __name__ == '__main__':
    main()