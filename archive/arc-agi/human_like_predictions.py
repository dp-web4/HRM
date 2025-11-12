#!/usr/bin/env python3
"""
Human-like predictions for ARC tasks - SAGE V3 training data.
This time we think like a human: simple, visual, pattern-first.
No elaborate algorithms - just "what would I do if I saw this?"
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

def looks_like_same_size(inputs: List, outputs: List) -> bool:
    """Quick check - are inputs and outputs same size?"""
    if not inputs or not outputs:
        return False
    for inp, out in zip(inputs, outputs):
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return False
    return True

def find_different_pixels(grid1: List[List[int]], grid2: List[List[int]]) -> List[Tuple[int, int]]:
    """Where do two grids differ?"""
    diffs = []
    for r in range(min(len(grid1), len(grid2))):
        for c in range(min(len(grid1[0]), len(grid2[0]))):
            if grid1[r][c] != grid2[r][c]:
                diffs.append((r, c))
    return diffs

def get_colors_used(grid: List[List[int]]) -> set:
    """What colors appear in this grid?"""
    return set(cell for row in grid for cell in row)

def copy_grid(grid: List[List[int]]) -> List[List[int]]:
    """Make a copy of the grid"""
    return [row[:] for row in grid]

def find_bounding_box(grid: List[List[int]], color: int = None) -> Tuple[int, int, int, int]:
    """Find bounding box of non-zero (or specific color) pixels"""
    min_r, max_r = len(grid), -1
    min_c, max_c = len(grid[0]) if grid else 0, -1
    
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if color is None:
                if grid[r][c] != 0:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
            else:
                if grid[r][c] == color:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
    
    if max_r == -1:  # Nothing found
        return 0, 0, 0, 0
    return min_r, min_c, max_r + 1, max_c + 1

def extract_subgrid(grid: List[List[int]], r1: int, c1: int, r2: int, c2: int) -> List[List[int]]:
    """Extract a rectangular region"""
    return [row[c1:c2] for row in grid[r1:r2]]

def human_solve(test_input: List[List[int]], train_examples: List[Dict]) -> List[List[int]]:
    """
    Solve like a human would - look for the simplest pattern that fits.
    """
    if not train_examples:
        return test_input
    
    # First, quick visual scan - what's the pattern?
    first_in = train_examples[0]['input']
    first_out = train_examples[0]['output']
    
    # 1. EXACT COPY? (simplest possible)
    if first_in == first_out:
        return copy_grid(test_input)
    
    # 2. SIZE CHANGE? (extraction or cropping)
    if len(first_out) != len(first_in) or len(first_out[0]) != len(first_in[0]):
        # Probably extracting something
        # What size is the output usually?
        out_h, out_w = len(first_out), len(first_out[0])
        
        # Is it always the same output size?
        same_size = all(len(ex['output']) == out_h and len(ex['output'][0]) == out_w 
                        for ex in train_examples)
        
        if same_size:
            # Extract a region of that size from test
            # Look for the non-zero bounding box
            r1, c1, r2, c2 = find_bounding_box(test_input)
            if r2 - r1 > 0 and c2 - c1 > 0:
                # Try to extract something sensible
                if r2 - r1 <= out_h and c2 - c1 <= out_w:
                    return extract_subgrid(test_input, r1, c1, r2, c2)
                else:
                    # Just take top-left corner of the right size
                    return extract_subgrid(test_input, 0, 0, 
                                         min(out_h, len(test_input)), 
                                         min(out_w, len(test_input[0])))
        
        # Different approach - maybe it's extracting a specific color region?
        colors_in = get_colors_used(first_in) - {0}
        colors_out = get_colors_used(first_out) - {0}
        
        if len(colors_out) < len(colors_in):
            # Extracting specific color
            target_color = list(colors_out)[0] if colors_out else 1
            r1, c1, r2, c2 = find_bounding_box(test_input, target_color)
            if r2 - r1 > 0:
                return extract_subgrid(test_input, r1, c1, r2, c2)
    
    # 3. SAME SIZE - what changes?
    if looks_like_same_size([first_in], [first_out]):
        result = copy_grid(test_input)
        
        # Color replacement?
        in_colors = get_colors_used(first_in)
        out_colors = get_colors_used(first_out)
        
        if len(in_colors) == len(out_colors) and in_colors != out_colors:
            # Simple color mapping
            color_map = {}
            # Try to figure out the mapping from examples
            for r in range(min(len(first_in), len(first_out))):
                for c in range(min(len(first_in[0]), len(first_out[0]))):
                    if first_in[r][c] != 0:
                        color_map[first_in[r][c]] = first_out[r][c]
            
            # Apply color mapping
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] in color_map:
                        result[r][c] = color_map[result[r][c]]
            return result
        
        # Filling pattern?
        # Look for pixels that changed from 0 to non-zero
        filled_pixels = []
        for r in range(min(len(first_in), len(first_out))):
            for c in range(min(len(first_in[0]), len(first_out[0]))):
                if first_in[r][c] == 0 and first_out[r][c] != 0:
                    filled_pixels.append((r, c))
        
        if filled_pixels:
            # What color are they filled with?
            fill_color = first_out[filled_pixels[0][0]][filled_pixels[0][1]]
            
            # Look for similar pattern in test - enclosed areas?
            # Simple heuristic: fill any 0 that's surrounded by non-zeros
            h, w = len(result), len(result[0])
            for r in range(1, h-1):
                for c in range(1, w-1):
                    if result[r][c] == 0:
                        # Check if surrounded (at least 3 sides have non-zero)
                        neighbors = [
                            result[r-1][c], result[r+1][c],
                            result[r][c-1], result[r][c+1]
                        ]
                        if sum(1 for n in neighbors if n != 0) >= 3:
                            result[r][c] = fill_color
            return result
        
        # Mirror/flip pattern?
        # Check if output is flipped version of input
        if all(first_out[r] == first_in[r][::-1] for r in range(len(first_in))):
            # Horizontal flip
            return [row[::-1] for row in test_input]
        
        if all(first_out[r] == first_in[len(first_in)-1-r] for r in range(len(first_in))):
            # Vertical flip
            return test_input[::-1]
        
        # Rotation?
        # Check for 90 degree rotation (simple case)
        if len(first_out) == len(first_in[0]) and len(first_out[0]) == len(first_in):
            # Could be rotation
            rotated = [[first_in[len(first_in)-1-c][r] 
                       for c in range(len(first_in))] 
                      for r in range(len(first_in[0]))]
            if rotated == first_out:
                # 90 degree rotation
                h, w = len(test_input), len(test_input[0])
                return [[test_input[w-1-c][r] for c in range(w)] for r in range(h)]
    
    # 4. COUNTING PATTERN?
    # Check if output is very small (like 1x1 or 2x2) - might be counting
    if len(first_out) <= 2 and len(first_out[0]) <= 2:
        # Count something in the input
        colors = get_colors_used(test_input) - {0}
        if len(colors) == 1:
            # Count non-zero pixels
            count = sum(1 for row in test_input for cell in row if cell != 0)
            # Return count as color value (mod 10 to keep it valid)
            return [[min(count, 9)]]
        else:
            # Count number of colors
            return [[len(colors)]]
    
    # 5. PATTERN COMPLETION?
    # Look for partial patterns that need completion
    # Simple case: if there's symmetry in output but not input
    # Check if left half equals right half in output
    if len(first_out[0]) % 2 == 0:
        mid = len(first_out[0]) // 2
        left_half = [row[:mid] for row in first_out]
        right_half = [row[mid:][::-1] for row in first_out]
        if left_half == right_half:
            # Complete symmetry in test
            result = copy_grid(test_input)
            w = len(result[0])
            if w % 2 == 0:
                mid = w // 2
                for r in range(len(result)):
                    for c in range(mid):
                        if result[r][c] != 0 and result[r][w-1-c] == 0:
                            result[r][w-1-c] = result[r][c]
                        elif result[r][w-1-c] != 0 and result[r][c] == 0:
                            result[r][c] = result[r][w-1-c]
            return result
    
    # 6. DEFAULT: Return most common output pattern from training
    # If we can't figure it out, pick the most common output shape
    output_shapes = [(len(ex['output']), len(ex['output'][0])) for ex in train_examples]
    if output_shapes:
        most_common = max(set(output_shapes), key=output_shapes.count)
        target_h, target_w = most_common
        
        # Return a grid of that size
        if target_h <= len(test_input) and target_w <= len(test_input[0]):
            return extract_subgrid(test_input, 0, 0, target_h, target_w)
        else:
            # Make a grid of the target size with the most common color
            colors = get_colors_used(test_input)
            fill_color = max(colors, key=lambda c: sum(row.count(c) for row in test_input))
            return [[fill_color] * target_w for _ in range(target_h)]
    
    # Last resort: return input unchanged
    return test_input

def generate_human_predictions():
    """Generate predictions thinking like a human would"""
    print("SAGE V3 Training Data - Human-like Predictions")
    print("=" * 50)
    print("Thinking simply: What would a human see?\n")
    
    # Load test tasks
    test_path = Path('kaggle_submission_package/arc-prize-2025/arc-agi_test_challenges.json')
    if not test_path.exists():
        test_path = Path('arc-prize-2025/arc-agi_test_challenges.json')
    with open(test_path, 'r') as f:
        test_tasks = json.load(f)
    
    print(f"Processing {len(test_tasks)} tasks...")
    
    predictions = {}
    
    for i, (task_id, task_data) in enumerate(test_tasks.items()):
        if i % 20 == 0:
            print(f"  {i}/{len(test_tasks)} tasks processed...")
        
        train_examples = task_data.get('train', [])
        test_cases = task_data.get('test', [])
        
        task_predictions = []
        for test_case in test_cases:
            test_input = test_case['input']
            
            # Generate prediction with human-like reasoning
            prediction = human_solve(test_input, train_examples)
            
            # For attempt_2, try something slightly different
            # Maybe invert colors or flip if attempt_1 looks wrong
            attempt2 = prediction
            if len(prediction) == len(test_input) and prediction[0]:
                if len(prediction[0]) == len(test_input[0]):
                    # Try color inversion as attempt 2
                    colors = get_colors_used(prediction)
                    if len(colors) == 2:
                        color_list = sorted(list(colors))
                        attempt2 = [[color_list[1] if c == color_list[0] else color_list[0] 
                                    for c in row] for row in prediction]
            
            task_predictions.append({
                'attempt_1': prediction,
                'attempt_2': attempt2
            })
        
        predictions[task_id] = task_predictions
    
    # Save predictions
    output_path = Path('human_like_predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f, separators=(',', ':'))
    
    print(f"\n✓ Human-like predictions saved to {output_path}")
    
    # Create training data format
    training_data = {}
    for task_id, task_data in test_tasks.items():
        task_entry = {
            'train': task_data.get('train', []),
            'test': []
        }
        
        for i, test_case in enumerate(task_data.get('test', [])):
            test_entry = {
                'input': test_case['input'],
                'output': predictions[task_id][i]['attempt_1']
            }
            task_entry['test'].append(test_entry)
        
        training_data[task_id] = task_entry
    
    # Save training data
    training_path = Path('sage_v3_training_data.json')
    with open(training_path, 'w') as f:
        json.dump(training_data, f, separators=(',', ':'))
    
    print(f"✓ SAGE V3 training data saved to {training_path}")
    
    # Quick stats
    total = len(predictions)
    non_zero = sum(1 for task_preds in predictions.values() 
                   for pred in task_preds 
                   if any(any(c != 0 for c in row) for row in pred['attempt_1']))
    
    print(f"\nStats:")
    print(f"  Total predictions: {total}")
    print(f"  Non-zero predictions: {non_zero}")
    print(f"  Agent Zero rate: {(total - non_zero)/total*100:.1f}%")
    
    print("\nApproach: Simple human-like heuristics")
    print("  • Check for exact copy first")
    print("  • Look for size changes (extraction)")
    print("  • Try color mappings")
    print("  • Fill enclosed areas")
    print("  • Complete symmetry")
    print("  • Count objects/colors for tiny outputs")

if __name__ == '__main__':
    generate_human_predictions()