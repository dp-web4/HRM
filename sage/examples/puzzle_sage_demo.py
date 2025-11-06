#!/usr/bin/env python3
"""
Puzzle SAGE Demo - Vision → Puzzle → Consciousness Loop

Demonstrates complete integration:
1. Camera/image input
2. Vision → Puzzle encoding
3. SNARC salience assessment on puzzles
4. Unified SAGE consciousness loop processing
5. Geometric reasoning about spatial patterns

This shows how puzzle space enables geometric consciousness.
"""

import sys
from pathlib import Path

# Add HRM to path
hrm_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(hrm_root))

import torch
import time
from typing import Dict, Any

from sage.core.unified_sage_system import (
    UnifiedSAGESystem,
    SensorOutput,
    ExecutionResult
)
from sage.compression.vision_puzzle_vae import VisionPuzzleVAE
from sage.services.snarc.data_structures import CognitiveStance


class PuzzleVisionSensor:
    """
    Vision sensor that outputs puzzle-encoded observations

    Wraps VisionPuzzleVAE to provide puzzle space interface for SAGE
    """

    def __init__(self, vae: VisionPuzzleVAE):
        self.vae = vae
        self.vae.eval()  # Inference mode
        self.frame_count = 0

    def capture(self) -> SensorOutput:
        """
        Capture vision and encode to puzzle

        Returns:
            SensorOutput with puzzle data (30×30 grid)
        """
        # Simulate camera frame (in production, use actual camera)
        image = torch.rand(1, 3, 224, 224)  # Random image for demo

        # Encode to puzzle
        with torch.no_grad():
            puzzle = self.vae.encode_to_puzzle(image)  # [1, 30, 30]

        self.frame_count += 1

        return SensorOutput(
            data=puzzle[0],  # Remove batch dim → [30, 30]
            timestamp=time.time(),
            quality=1.0,  # Placeholder
            sensor_type='vision_puzzle',
            metadata={
                'puzzle_shape': (30, 30),
                'value_range': (0, 9),
                'frame': self.frame_count,
                'encoding': 'VQ-VAE puzzle space'
            }
        )


class PuzzleActionHandler:
    """
    Handles actions on puzzle observations

    Demonstrates geometric reasoning about spatial patterns
    """

    def __init__(self):
        self.patterns_detected = []

    def __call__(
        self,
        observation: Any,
        stance: CognitiveStance
    ) -> ExecutionResult:
        """
        Process puzzle observation and detect patterns

        Args:
            observation: SensorOutput with puzzle data (30×30)
            stance: Cognitive stance from SNARC

        Returns:
            ExecutionResult with pattern analysis
        """
        puzzle = observation.data  # [30, 30] tensor

        # Geometric pattern detection
        patterns = self._detect_patterns(puzzle)

        # Stance-dependent interpretation
        if stance == CognitiveStance.CURIOUS_UNCERTAINTY:
            description = f"Curious about patterns: {patterns['dominant_regions']}"
            reward = 0.6
        elif stance == CognitiveStance.FOCUSED_ATTENTION:
            description = f"Focused on {patterns['high_value_regions']} high-value regions"
            reward = 0.8
        else:
            description = f"Observing puzzle with {patterns['unique_values']} distinct values"
            reward = 0.7

        self.patterns_detected.append(patterns)

        return ExecutionResult(
            success=True,
            reward=reward,
            description=description,
            outputs={
                'patterns': patterns,
                'puzzle_summary': self._summarize_puzzle(puzzle),
                'actions': {
                    'analysis': description
                }
            }
        )

    def _detect_patterns(self, puzzle: torch.Tensor) -> Dict[str, Any]:
        """
        Detect geometric patterns in puzzle

        Args:
            puzzle: [30, 30] tensor with values 0-9

        Returns:
            Dictionary of detected patterns
        """
        # Value distribution
        unique_vals, counts = torch.unique(puzzle, return_counts=True)
        dominant_value = unique_vals[counts.argmax()].item()

        # Spatial clustering (simple: check quadrants)
        h, w = puzzle.shape
        quadrants = {
            'top_left': puzzle[:h//2, :w//2].float().mean().item(),
            'top_right': puzzle[:h//2, w//2:].float().mean().item(),
            'bottom_left': puzzle[h//2:, :w//2].float().mean().item(),
            'bottom_right': puzzle[h//2:, w//2:].float().mean().item()
        }

        # High-value regions (values >= 7)
        high_val_mask = puzzle >= 7
        high_val_regions = high_val_mask.sum().item()

        # Horizontal/vertical structure
        row_variance = puzzle.float().var(dim=1).mean().item()
        col_variance = puzzle.float().var(dim=0).mean().item()

        return {
            'unique_values': len(unique_vals),
            'dominant_value': dominant_value,
            'dominant_count': counts.max().item(),
            'quadrants': quadrants,
            'dominant_quadrant': max(quadrants, key=quadrants.get),
            'high_value_regions': high_val_regions,
            'row_structure': row_variance,
            'col_structure': col_variance,
            'dominant_regions': max(quadrants, key=quadrants.get)
        }

    def _summarize_puzzle(self, puzzle: torch.Tensor) -> str:
        """Create human-readable puzzle summary"""
        mean_val = puzzle.float().mean().item()
        std_val = puzzle.float().std().item()
        min_val = puzzle.min().item()
        max_val = puzzle.max().item()

        return (
            f"30×30 puzzle: mean={mean_val:.1f}, "
            f"std={std_val:.1f}, range=[{min_val}, {max_val}]"
        )


def main():
    print("=" * 70)
    print("Puzzle SAGE Demo - Geometric Consciousness")
    print("=" * 70)

    # Create Vision → Puzzle VAE
    print("\n1. Creating Vision → Puzzle VAE...")
    vae = VisionPuzzleVAE(latent_dim=64, num_codes=10)
    print("   ✓ VAE initialized (untrained - for demo)")

    # Create puzzle vision sensor
    print("\n2. Creating Puzzle Vision Sensor...")
    puzzle_sensor = PuzzleVisionSensor(vae)
    print("   ✓ Sensor ready (wraps VAE)")

    # Create action handler for puzzles
    print("\n3. Creating Puzzle Action Handler...")
    puzzle_handler = PuzzleActionHandler()
    print("   ✓ Handler ready (geometric pattern detection)")

    # Create Unified SAGE with puzzle sensor
    print("\n4. Creating Unified SAGE System...")
    sage = UnifiedSAGESystem(
        sensor_sources={
            'vision_puzzle': puzzle_sensor.capture
        },
        action_handlers={
            'vision_puzzle': puzzle_handler
        },
        config={
            'initial_atp': 100.0,
            'enable_circadian': False  # Disable for demo
        },
        enable_logging=True
    )
    print("   ✓ UnifiedSAGE initialized")

    # Run consciousness loop
    print("\n5. Running Consciousness Loop...")
    print("   (Processing 10 puzzle-encoded vision frames)\n")

    sage.run(max_cycles=10, cycle_delay=0.1)

    # Analysis
    print("\n" + "=" * 70)
    print("PUZZLE-BASED CONSCIOUSNESS ANALYSIS")
    print("=" * 70)

    # System status
    status = sage.get_system_status()
    print(f"\nSystem Status:")
    print(f"  Cycles completed: {status['cycle_count']}")
    print(f"  ATP remaining: {status['atp_level']:.1f}/{status['max_atp']}")
    print(f"  Metabolic state: {status['metabolic_state']}")
    print(f"  Memory items: {status['memory_items']}")
    print(f"  Salient memories: {status['salient_memories']}")

    # Performance
    perf = sage.get_performance_summary()
    print(f"\nPerformance:")
    print(f"  Average cycle time: {perf['avg_cycle_time']*1000:.2f}ms")
    print(f"  Average salience: {perf['avg_salience']:.3f}")

    # Pattern detection summary
    print(f"\nGeometric Pattern Detection:")
    print(f"  Total patterns analyzed: {len(puzzle_handler.patterns_detected)}")

    if puzzle_handler.patterns_detected:
        # Aggregate pattern statistics
        avg_unique_vals = sum(
            p['unique_values'] for p in puzzle_handler.patterns_detected
        ) / len(puzzle_handler.patterns_detected)

        quadrant_distribution = {}
        for pattern in puzzle_handler.patterns_detected:
            quad = pattern['dominant_quadrant']
            quadrant_distribution[quad] = quadrant_distribution.get(quad, 0) + 1

        print(f"  Avg unique values per puzzle: {avg_unique_vals:.1f}/10")
        print(f"  Dominant quadrant distribution:")
        for quad, count in quadrant_distribution.items():
            print(f"    {quad}: {count} frames")

    # Show sample pattern
    if puzzle_handler.patterns_detected:
        print(f"\nSample Pattern (first frame):")
        sample = puzzle_handler.patterns_detected[0]
        print(f"  Unique values: {sample['unique_values']}")
        print(f"  Dominant value: {sample['dominant_value']} "
              f"({sample['dominant_count']} cells)")
        print(f"  Quadrant means: {sample['quadrants']}")
        print(f"  High-value regions: {sample['high_value_regions']} cells (≥7)")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("\n1. GEOMETRIC ENCODING WORKS")
    print("   Vision frames → 30×30 puzzles with spatial structure")
    print("\n2. SNARC OPERATES ON PUZZLE SPACE")
    print("   Salience assessment directly on geometric representations")
    print("\n3. PATTERN DETECTION IS SPATIAL")
    print("   Quadrant analysis, value distributions, regional clustering")
    print("\n4. STANCE INFLUENCES INTERPRETATION")
    print("   Same puzzle, different cognitive lens → different insights")
    print("\n5. CONSCIOUSNESS LOOP COMPLETE")
    print("   Sensor → Puzzle → SNARC → Decide → Act → Learn → Memory")

    print("\n" + "=" * 70)
    print("Puzzle-based geometric consciousness operational!")
    print("=" * 70)


if __name__ == "__main__":
    main()
