#!/usr/bin/env python3
"""
Quick Sudoku Demo for HRM - Works on CPU
"""

import torch
import numpy as np
from pathlib import Path

print("🧩 HRM Sudoku Demo (CPU-friendly)")
print("=" * 50)

# Check PyTorch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Create a simple Sudoku puzzle
print("\n📝 Creating a simple 4x4 Sudoku puzzle...")

# 4x4 Sudoku (easier than 9x9)
# Complete solution:
# 1 2 | 3 4
# 3 4 | 1 2
# ----+----
# 2 1 | 4 3
# 4 3 | 2 1

# Puzzle with some numbers removed
puzzle = torch.tensor([
    [1, 0, 3, 4],
    [0, 4, 0, 2],
    [2, 0, 4, 0],
    [4, 3, 0, 1]
], dtype=torch.float32)

solution = torch.tensor([
    [1, 2, 3, 4],
    [3, 4, 1, 2],
    [2, 1, 4, 3],
    [4, 3, 2, 1]
], dtype=torch.float32)

def print_sudoku(grid, name="Sudoku"):
    """Pretty print a Sudoku grid"""
    print(f"\n{name}:")
    for i, row in enumerate(grid):
        if i == 2:
            print("------+------")
        row_str = ""
        for j, val in enumerate(row):
            if j == 2:
                row_str += "| "
            row_str += f"{int(val.item()) if val > 0 else '.'} "
        print(row_str)

print_sudoku(puzzle, "Puzzle")
print_sudoku(solution, "Solution")

# Simple solver using constraints (not HRM, just for demo)
print("\n🤔 Solving with simple constraint propagation...")

def is_valid(grid, row, col, num):
    """Check if placing num at (row, col) is valid"""
    # Check row
    if num in grid[row]:
        return False
    
    # Check column
    if num in grid[:, col]:
        return False
    
    # Check 2x2 box
    box_row, box_col = 2 * (row // 2), 2 * (col // 2)
    if num in grid[box_row:box_row+2, box_col:box_col+2]:
        return False
    
    return True

def solve_simple(grid):
    """Simple backtracking solver"""
    grid = grid.clone()
    
    # Find empty cell
    for i in range(4):
        for j in range(4):
            if grid[i, j] == 0:
                # Try numbers 1-4
                for num in range(1, 5):
                    if is_valid(grid, i, j, num):
                        grid[i, j] = num
                        if solve_simple(grid) is not None:
                            return grid
                        grid[i, j] = 0
                return None
    return grid

solved = solve_simple(puzzle)
if solved is not None:
    print_sudoku(solved, "Solved")
    correct = torch.all(solved == solution)
    print(f"\n✅ Solution is {'CORRECT!' if correct else 'INCORRECT!'}")
else:
    print("\n❌ Could not solve puzzle")

print("\n" + "=" * 50)
print("💡 Next Steps for HRM:")
print("1. Build full Sudoku dataset:")
print("   python dataset/build_sudoku_dataset.py --output-dir data/sudoku-small --subsample-size 100")
print("\n2. Train HRM model:")
print("   python pretrain.py data_path=data/sudoku-small epochs=1000")
print("\n3. HRM will learn to solve Sudoku without explicit rules!")
print("=" * 50)