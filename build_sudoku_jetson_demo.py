#!/usr/bin/env python3
"""
Minimal Sudoku dataset for Jetson demo
Creates just 10 puzzles for quick testing
"""
import json
import os
import random

def create_minimal_sudoku_dataset(output_dir, num_puzzles=10):
    """Create a tiny sudoku dataset for testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Simple sudoku generator (for demo purposes)
    # In production, use the full dataset builder
    puzzles = []
    
    print(f"Generating {num_puzzles} sudoku puzzles...")
    
    for i in range(num_puzzles):
        # Create a partially filled sudoku grid
        # This is simplified - real sudoku generation is more complex
        grid = [[0] * 9 for _ in range(9)]
        
        # Fill some random cells (about 30%)
        for _ in range(25):
            row = random.randint(0, 8)
            col = random.randint(0, 8)
            val = random.randint(1, 9)
            grid[row][col] = val
        
        # Convert to format HRM expects
        puzzle = {
            "input": grid,
            "output": grid,  # In real dataset, this would be the solution
            "id": f"sudoku_demo_{i}"
        }
        puzzles.append(puzzle)
    
    # Save dataset
    with open(os.path.join(output_dir, "puzzles.json"), "w") as f:
        json.dump(puzzles, f, indent=2)
    
    print(f"Created {num_puzzles} puzzles in {output_dir}")
    
    # Create metadata
    metadata = {
        "dataset": "sudoku-jetson-demo",
        "num_puzzles": num_puzzles,
        "grid_size": 9,
        "description": "Minimal sudoku dataset for Jetson testing"
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    create_minimal_sudoku_dataset("data/sudoku-jetson-demo", num_puzzles=10)
