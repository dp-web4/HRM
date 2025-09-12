#!/usr/bin/env python3
"""
Claude's PROPER reasoning for ARC tasks after examining the actual patterns.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

def load_task_and_solve(task_id: str) -> Dict:
    """Load task and solve with proper reasoning"""
    
    train_path = Path('arc-prize-2025/arc-agi_training_challenges.json')
    solutions_path = Path('arc-prize-2025/arc-agi_training_solutions.json')
    
    with open(train_path, 'r') as f:
        challenges = json.load(f)
    
    with open(solutions_path, 'r') as f:
        solutions = json.load(f)
    
    task = challenges[task_id]
    ground_truth = solutions[task_id][0]
    test_input = task['test'][0]['input']
    
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")
    
    # Analyze training examples
    print("\nTraining Examples:")
    for i, ex in enumerate(task['train'], 1):
        inp = ex['input']
        out = ex['output']
        print(f"\nExample {i}:")
        print(f"  Input:  {len(inp)}x{len(inp[0])} - {inp[:2] if len(inp) > 2 else inp}")
        print(f"  Output: {len(out)}x{len(out[0])} - {out[:2] if len(out) > 2 else out}")
    
    print(f"\nTest Input: {len(test_input)}x{len(test_input[0])}")
    
    # Solve based on task ID with proper reasoning
    solution = None
    reasoning = ""
    
    if task_id == '00576224':
        reasoning = """
PATTERN: 2x2 → 6x6 tiling with alternating row reversal
- Take 2x2 input
- Create 6x6 by tiling 3x3 with pattern:
  - Rows 0-1: original tiled 3x horizontally
  - Rows 2-3: each row reversed, tiled 3x  
  - Rows 4-5: original tiled 3x horizontally
"""
        inp = test_input
        solution = []
        # Original rows tiled
        for row in inp:
            solution.append(row * 3)
        # Reversed rows tiled
        for row in inp:
            solution.append(row[::-1] * 3)
        # Original rows tiled again
        for row in inp:
            solution.append(row * 3)
            
    elif task_id == '007bbfb7':
        reasoning = """
PATTERN: 3x3 → 9x9 cell expansion
- Each cell becomes a 3x3 block
- Non-zero cells fill their block with their value
- Zero cells create zero blocks
"""
        inp = test_input
        solution = []
        for row in inp:
            # Each row becomes 3 rows
            for _ in range(3):
                new_row = []
                for cell in row:
                    # Each cell becomes 3 cells
                    new_row.extend([cell] * 3)
                solution.append(new_row)
                
    elif task_id == '0520fde7':
        reasoning = """
PATTERN: Extract pattern from right side of divider
- Input has a vertical divider (column of 5s)
- Extract the pattern to the RIGHT of the divider
- Output is always 3x3
- Transform specific colors (1→2 in the extracted region)
"""
        inp = test_input
        # Find the divider column (5s)
        divider_col = None
        for c in range(len(inp[0])):
            if all(inp[r][c] == 5 for r in range(len(inp))):
                divider_col = c
                break
        
        if divider_col is not None:
            # Extract 3x3 from right of divider
            solution = []
            for r in range(3):
                row = []
                for c in range(divider_col + 1, divider_col + 4):
                    if r < len(inp) and c < len(inp[0]):
                        val = inp[r][c]
                        # Transform 1 to 2
                        if val == 1:
                            val = 2
                        row.append(val)
                    else:
                        row.append(0)
                solution.append(row)
        else:
            solution = [[0, 0, 0] for _ in range(3)]
            
    elif task_id == '025d127b':
        reasoning = """
PATTERN: Fill rectangles bounded by 8s with color 2
- Find rectangles outlined by 8s
- Fill their interiors with 2
- Leave other areas unchanged
"""
        inp = [row[:] for row in test_input]
        solution = [row[:] for row in inp]
        
        # Find all 8s and check for rectangles
        h, w = len(inp), len(inp[0])
        
        # Find rectangular regions by looking for corners
        for r1 in range(h):
            for c1 in range(w):
                if inp[r1][c1] == 8:
                    # Try to find a rectangle starting here
                    for r2 in range(r1+2, h):
                        for c2 in range(c1+2, w):
                            if inp[r2][c2] == 8:
                                # Check if this forms a rectangle
                                # Check top and bottom edges
                                top_edge = all(inp[r1][c] == 8 for c in range(c1, c2+1))
                                bottom_edge = all(inp[r2][c] == 8 for c in range(c1, c2+1))
                                # Check left and right edges
                                left_edge = all(inp[r][c1] == 8 for r in range(r1, r2+1))
                                right_edge = all(inp[r][c2] == 8 for r in range(r1, r2+1))
                                
                                if top_edge and bottom_edge and left_edge and right_edge:
                                    # Fill interior
                                    for r in range(r1+1, r2):
                                        for c in range(c1+1, c2):
                                            if solution[r][c] == 0:
                                                solution[r][c] = 2
                                                
    elif task_id == '1cf80156':
        reasoning = """
PATTERN: Gravity - colored cells fall to bottom
- Non-zero values fall down in their column
- Stack at bottom or on top of other non-zeros
- Maintains relative order within each column
"""
        inp = test_input
        h, w = len(inp), len(inp[0])
        solution = [[0] * w for _ in range(h)]
        
        for c in range(w):
            # Collect non-zero values from top to bottom
            values = []
            for r in range(h):
                if inp[r][c] != 0:
                    values.append(inp[r][c])
            
            # Place at bottom
            for i, val in enumerate(values):
                solution[h - len(values) + i][c] = val
    
    print(f"\nReasoning: {reasoning}")
    
    # Check correctness
    if solution == ground_truth:
        print("✅ CORRECT!")
        return {'task_id': task_id, 'correct': True, 'solution': solution, 'ground_truth': ground_truth}
    else:
        print("❌ INCORRECT")
        if solution:
            sol_shape = f"{len(solution)}x{len(solution[0])}"
            gt_shape = f"{len(ground_truth)}x{len(ground_truth[0])}"
            print(f"  My solution: {sol_shape}")
            print(f"  Ground truth: {gt_shape}")
            if sol_shape == gt_shape:
                sol_flat = np.array(solution).flatten()
                gt_flat = np.array(ground_truth).flatten()
                acc = (sol_flat == gt_flat).mean()
                print(f"  Pixel accuracy: {acc:.1%}")
        return {'task_id': task_id, 'correct': False, 'solution': solution, 'ground_truth': ground_truth}

def main():
    """Properly solve 5 ARC tasks"""
    
    task_ids = [
        '00576224',  # Tiling with row reversal
        '007bbfb7',  # 3x3 cell expansion
        '0520fde7',  # Pattern extraction from divider
        '025d127b',  # Fill rectangles
        '1cf80156'   # Gravity
    ]
    
    results = []
    for task_id in task_ids:
        result = load_task_and_solve(task_id)
        results.append(result)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    correct = sum(1 for r in results if r['correct'])
    print(f"Correct: {correct}/{len(results)}")
    
    for r in results:
        status = "✅" if r['correct'] else "❌"
        print(f"  {status} {r['task_id']}")

if __name__ == "__main__":
    main()