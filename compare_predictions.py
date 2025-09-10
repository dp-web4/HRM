#!/usr/bin/env python3
"""
Compare V2 (claude_reasoning) and V3 (human_like) predictions
to see how different the approaches are.
"""

import json
import numpy as np
from typing import List, Dict, Tuple

def grid_similarity(grid1: List[List[int]], grid2: List[List[int]]) -> float:
    """Calculate similarity between two grids (0 to 1)"""
    # Handle size mismatches
    if len(grid1) != len(grid2):
        return 0.0
    if grid1 and len(grid1[0]) != len(grid2[0]):
        return 0.0
    
    # Calculate pixel-wise similarity
    total_pixels = 0
    matching_pixels = 0
    
    for r in range(len(grid1)):
        for c in range(len(grid1[0])):
            total_pixels += 1
            if grid1[r][c] == grid2[r][c]:
                matching_pixels += 1
    
    return matching_pixels / total_pixels if total_pixels > 0 else 0.0

def analyze_predictions():
    """Compare V2 and V3 predictions"""
    
    print("Comparing SAGE V2 (Algorithmic) vs V3 (Human-like) Predictions")
    print("=" * 60)
    
    # Load both prediction sets
    with open('kaggle_submission_package/claude_reasoning_predictions.json', 'r') as f:
        v2_preds = json.load(f)
    
    with open('human_like_predictions.json', 'r') as f:
        v3_preds = json.load(f)
    
    # Basic statistics
    print(f"\nTotal tasks: {len(v2_preds)}")
    assert len(v2_preds) == len(v3_preds), "Different number of tasks!"
    
    # Compare predictions
    identical_predictions = 0
    high_similarity = 0  # >80% similar
    medium_similarity = 0  # 50-80% similar
    low_similarity = 0  # <50% similar
    
    size_matches = 0
    size_mismatches = 0
    
    similarities = []
    
    # Detailed comparison
    size_change_v2 = 0
    size_change_v3 = 0
    
    # Track specific differences
    v2_all_zeros = 0
    v3_all_zeros = 0
    
    # Sample some specific differences
    example_differences = []
    
    for task_id in v2_preds.keys():
        v2_pred = v2_preds[task_id][0]['attempt_1']
        v3_pred = v3_preds[task_id][0]['attempt_1']
        
        # Check if predictions are identical
        if v2_pred == v3_pred:
            identical_predictions += 1
        
        # Check for all-zero predictions
        v2_is_zero = all(all(c == 0 for c in row) for row in v2_pred)
        v3_is_zero = all(all(c == 0 for c in row) for row in v3_pred)
        
        if v2_is_zero:
            v2_all_zeros += 1
        if v3_is_zero:
            v3_all_zeros += 1
        
        # Check size
        v2_size = (len(v2_pred), len(v2_pred[0]) if v2_pred else 0)
        v3_size = (len(v3_pred), len(v3_pred[0]) if v3_pred else 0)
        
        if v2_size == v3_size:
            size_matches += 1
            
            # Calculate similarity for same-size grids
            sim = grid_similarity(v2_pred, v3_pred)
            similarities.append(sim)
            
            if sim == 1.0:
                pass  # Already counted as identical
            elif sim > 0.8:
                high_similarity += 1
            elif sim > 0.5:
                medium_similarity += 1
            else:
                low_similarity += 1
                
                # Save an example of low similarity
                if len(example_differences) < 3 and sim < 0.3:
                    example_differences.append({
                        'task_id': task_id,
                        'similarity': sim,
                        'v2_size': v2_size,
                        'v3_size': v3_size,
                        'v2_colors': len(set(c for row in v2_pred for c in row)),
                        'v3_colors': len(set(c for row in v3_pred for c in row)),
                        'v2_zero': v2_is_zero,
                        'v3_zero': v3_is_zero
                    })
        else:
            size_mismatches += 1
            similarities.append(0.0)  # Different sizes = 0 similarity
            
            # Track which version changes size more
            # (We'd need input size to know for sure, but we can compare)
            if v2_size != v3_size:
                if v2_size[0] * v2_size[1] < v3_size[0] * v3_size[1]:
                    size_change_v2 += 1
                else:
                    size_change_v3 += 1
            
            # Save an example of size mismatch
            if len(example_differences) < 5:
                example_differences.append({
                    'task_id': task_id,
                    'similarity': 0.0,
                    'v2_size': v2_size,
                    'v3_size': v3_size,
                    'v2_colors': len(set(c for row in v2_pred for c in row)),
                    'v3_colors': len(set(c for row in v3_pred for c in row)),
                    'v2_zero': v2_is_zero,
                    'v3_zero': v3_is_zero
                })
    
    # Calculate statistics
    avg_similarity = np.mean(similarities) if similarities else 0.0
    
    # Print results
    print("\n### OVERALL COMPARISON ###")
    print(f"Identical predictions: {identical_predictions}/{len(v2_preds)} ({identical_predictions/len(v2_preds)*100:.1f}%)")
    print(f"Average similarity: {avg_similarity:.3f}")
    
    print("\n### SIZE ANALYSIS ###")
    print(f"Same size outputs: {size_matches}/{len(v2_preds)} ({size_matches/len(v2_preds)*100:.1f}%)")
    print(f"Different size outputs: {size_mismatches}/{len(v2_preds)} ({size_mismatches/len(v2_preds)*100:.1f}%)")
    
    print("\n### SIMILARITY BREAKDOWN (for same-size predictions) ###")
    print(f"Identical (100%): {identical_predictions}")
    print(f"High similarity (>80%): {high_similarity}")
    print(f"Medium similarity (50-80%): {medium_similarity}")
    print(f"Low similarity (<50%): {low_similarity}")
    
    print("\n### ZERO PREDICTIONS ###")
    print(f"V2 all-zero predictions: {v2_all_zeros}/{len(v2_preds)} ({v2_all_zeros/len(v2_preds)*100:.1f}%)")
    print(f"V3 all-zero predictions: {v3_all_zeros}/{len(v3_preds)} ({v3_all_zeros/len(v3_preds)*100:.1f}%)")
    
    print("\n### APPROACH DIFFERENCES ###")
    
    # Analyze color usage
    v2_total_colors = []
    v3_total_colors = []
    
    for task_id in v2_preds.keys():
        v2_pred = v2_preds[task_id][0]['attempt_1']
        v3_pred = v3_preds[task_id][0]['attempt_1']
        
        v2_colors = len(set(c for row in v2_pred for c in row))
        v3_colors = len(set(c for row in v3_pred for c in row))
        
        v2_total_colors.append(v2_colors)
        v3_total_colors.append(v3_colors)
    
    print(f"V2 avg colors per prediction: {np.mean(v2_total_colors):.2f}")
    print(f"V3 avg colors per prediction: {np.mean(v3_total_colors):.2f}")
    
    # Size statistics
    v2_sizes = []
    v3_sizes = []
    
    for task_id in v2_preds.keys():
        v2_pred = v2_preds[task_id][0]['attempt_1']
        v3_pred = v3_preds[task_id][0]['attempt_1']
        
        v2_sizes.append(len(v2_pred) * len(v2_pred[0]) if v2_pred else 0)
        v3_sizes.append(len(v3_pred) * len(v3_pred[0]) if v3_pred else 0)
    
    print(f"\nV2 avg grid size (pixels): {np.mean(v2_sizes):.1f}")
    print(f"V3 avg grid size (pixels): {np.mean(v3_sizes):.1f}")
    
    print("\n### EXAMPLE DIFFERENCES ###")
    for i, ex in enumerate(example_differences[:5], 1):
        print(f"\nExample {i} - Task {ex['task_id']}:")
        print(f"  Similarity: {ex['similarity']:.2%}")
        print(f"  V2: {ex['v2_size'][0]}x{ex['v2_size'][1]}, {ex['v2_colors']} colors" + 
              (" (all zeros)" if ex['v2_zero'] else ""))
        print(f"  V3: {ex['v3_size'][0]}x{ex['v3_size'][1]}, {ex['v3_colors']} colors" +
              (" (all zeros)" if ex['v3_zero'] else ""))
    
    # Show a specific prediction comparison
    print("\n### DETAILED EXAMPLE ###")
    sample_task = list(v2_preds.keys())[10]  # Pick task 10
    v2_sample = v2_preds[sample_task][0]['attempt_1']
    v3_sample = v3_preds[sample_task][0]['attempt_1']
    
    print(f"Task {sample_task}:")
    print(f"V2 prediction ({len(v2_sample)}x{len(v2_sample[0])}):")
    for row in v2_sample[:5]:  # Show first 5 rows
        print(" ", row[:10] if len(row) > 10 else row)
    if len(v2_sample) > 5:
        print("  ...")
    
    print(f"\nV3 prediction ({len(v3_sample)}x{len(v3_sample[0])}):")
    for row in v3_sample[:5]:  # Show first 5 rows
        print(" ", row[:10] if len(row) > 10 else row)
    if len(v3_sample) > 5:
        print("  ...")
    
    print("\n### SUMMARY ###")
    if avg_similarity < 0.3:
        print("V2 and V3 use VERY DIFFERENT approaches")
        print("V2: Complex algorithmic pattern detection")
        print("V3: Simple human-like visual reasoning")
    elif avg_similarity < 0.6:
        print("V2 and V3 show MODERATE differences")
        print("Some common patterns but different execution")
    else:
        print("V2 and V3 are QUITE SIMILAR")
        print("Both approaches converge on similar solutions")
    
    return {
        'identical': identical_predictions,
        'avg_similarity': avg_similarity,
        'size_matches': size_matches,
        'v2_zeros': v2_all_zeros,
        'v3_zeros': v3_all_zeros
    }

if __name__ == '__main__':
    results = analyze_predictions()