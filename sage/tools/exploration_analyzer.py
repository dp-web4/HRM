#!/usr/bin/env python3
"""
SAGE Exploration Data Analyzer
==============================

Analyzes autonomous exploration logs to extract insights:
- Salience patterns over time
- Task completion rates
- Performance metrics
- Cross-modal correlations (for multi-modal runs)
- Behavioral patterns

Use this to understand what SAGE has been learning during autonomous operation.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class ExplorationCycle:
    """Single exploration cycle data"""
    cycle: int
    task: str
    salience: float
    progress: float
    cycle_time_ms: float
    # Multi-modal specific
    position: Optional[np.ndarray] = None
    gripper: Optional[float] = None


class ExplorationAnalyzer:
    """Analyzes SAGE autonomous exploration logs"""

    def __init__(self, log_file: Path):
        self.log_file = Path(log_file)
        self.cycles: List[ExplorationCycle] = []
        self.summary = {}
        self.is_multimodal = False

        if self.log_file.exists():
            self._parse_log()

    def _parse_log(self):
        """Parse exploration log file"""
        print(f"📊 Analyzing: {self.log_file.name}")

        with open(self.log_file, 'r') as f:
            content = f.read()

        # Check if multi-modal
        self.is_multimodal = 'Multi-Modal' in content or 'Embodied' in content

        # Parse cycle lines
        if self.is_multimodal:
            pattern = r'Cycle\s+(\d+)\s+\|\s+Task:\s+(.{35})\s+\|\s+Salience:\s+([\d.]+)\s+\|\s+Pos:\s+\[([\s\d.-]+),([\s\d.-]+),([\s\d.-]+)\]\s+\|\s+Gripper:\s+([\d.]+)\s+\|\s+Time:\s+([\d.]+)ms'
        else:
            pattern = r'Cycle\s+(\d+)\s+\|\s+Task:\s+(.{40})\s+\|\s+Salience:\s+([\d.]+)\s+\|\s+Progress:\s+([\d.]+)%\s+\|\s+Time:\s+([\d.]+)ms'

        for match in re.finditer(pattern, content):
            if self.is_multimodal:
                cycle_data = ExplorationCycle(
                    cycle=int(match.group(1)),
                    task=match.group(2).strip(),
                    salience=float(match.group(3)),
                    progress=0.0,  # Not in multi-modal output
                    cycle_time_ms=float(match.group(8)),
                    position=np.array([
                        float(match.group(4)),
                        float(match.group(5)),
                        float(match.group(6))
                    ]),
                    gripper=float(match.group(7))
                )
            else:
                cycle_data = ExplorationCycle(
                    cycle=int(match.group(1)),
                    task=match.group(2).strip(),
                    salience=float(match.group(3)),
                    progress=float(match.group(4)),
                    cycle_time_ms=float(match.group(5))
                )

            self.cycles.append(cycle_data)

        # Parse summary
        summary_pattern = r'📊 Exploration Statistics:.*?Cycles:\s+(\d+).*?Total time:\s+([\d.]+)s.*?Avg cycle time:\s+([\d.]+)ms'
        summary_match = re.search(summary_pattern, content, re.DOTALL)
        if summary_match:
            self.summary = {
                'total_cycles': int(summary_match.group(1)),
                'total_time': float(summary_match.group(2)),
                'avg_cycle_time': float(summary_match.group(3))
            }

        # Parse salience analysis
        salience_pattern = r'🎯 Salience Analysis:.*?Average:\s+([\d.]+).*?Maximum:\s+([\d.]+).*?High salience.*?:\s+(\d+)'
        salience_match = re.search(salience_pattern, content, re.DOTALL)
        if salience_match:
            self.summary['avg_salience'] = float(salience_match.group(1))
            self.summary['max_salience'] = float(salience_match.group(2))
            self.summary['high_salience_count'] = int(salience_match.group(3))

        print(f"   ✓ Parsed {len(self.cycles)} cycles")

    def analyze_salience_patterns(self) -> Dict:
        """Analyze salience over time"""
        if not self.cycles:
            return {}

        saliences = np.array([c.salience for c in self.cycles])

        # Time-based analysis
        window_size = 50
        if len(saliences) >= window_size:
            smoothed = np.convolve(saliences, np.ones(window_size)/window_size, mode='valid')
        else:
            smoothed = saliences

        # Detect interesting patterns
        peaks = []
        for i in range(1, len(saliences) - 1):
            if saliences[i] > saliences[i-1] and saliences[i] > saliences[i+1]:
                if saliences[i] > np.mean(saliences) + np.std(saliences):
                    peaks.append((i, saliences[i]))

        return {
            'mean': np.mean(saliences),
            'std': np.std(saliences),
            'min': np.min(saliences),
            'max': np.max(saliences),
            'peaks': peaks,
            'smoothed': smoothed
        }

    def analyze_task_patterns(self) -> Dict:
        """Analyze task switching and completion"""
        if not self.cycles:
            return {}

        # Count unique tasks
        tasks = [c.task for c in self.cycles]
        unique_tasks = set(tasks)

        # Task switching frequency
        switches = sum(1 for i in range(1, len(tasks)) if tasks[i] != tasks[i-1])

        # Average duration per task
        task_durations = []
        current_task = tasks[0]
        duration = 1

        for task in tasks[1:]:
            if task == current_task:
                duration += 1
            else:
                task_durations.append(duration)
                current_task = task
                duration = 1
        task_durations.append(duration)

        return {
            'unique_tasks': len(unique_tasks),
            'total_switches': switches,
            'avg_task_duration': np.mean(task_durations),
            'max_task_duration': max(task_durations),
            'task_list': list(unique_tasks)
        }

    def analyze_performance(self) -> Dict:
        """Analyze performance metrics"""
        if not self.cycles:
            return {}

        cycle_times = np.array([c.cycle_time_ms for c in self.cycles])

        return {
            'avg_cycle_time': np.mean(cycle_times),
            'std_cycle_time': np.std(cycle_times),
            'min_cycle_time': np.min(cycle_times),
            'max_cycle_time': np.max(cycle_times),
            'trend': 'improving' if cycle_times[-10:].mean() < cycle_times[:10].mean() else 'stable'
        }

    def analyze_embodied_behavior(self) -> Optional[Dict]:
        """Analyze embodied behavior (multi-modal only)"""
        if not self.is_multimodal or not self.cycles:
            return None

        positions = np.array([c.position for c in self.cycles if c.position is not None])
        grippers = np.array([c.gripper for c in self.cycles if c.gripper is not None])

        if len(positions) == 0:
            return None

        # Spatial exploration
        position_variance = np.var(positions, axis=0)
        exploration_volume = np.prod(position_variance)

        # Movement patterns
        movements = np.diff(positions, axis=0)
        movement_magnitudes = np.linalg.norm(movements, axis=1)

        # Gripper usage
        gripper_changes = np.abs(np.diff(grippers))

        return {
            'position_variance': position_variance.tolist(),
            'exploration_volume': float(exploration_volume),
            'avg_movement': float(np.mean(movement_magnitudes)),
            'total_distance': float(np.sum(movement_magnitudes)),
            'gripper_activity': float(np.mean(gripper_changes)),
            'gripper_open_ratio': float(np.mean(grippers < 0.5))
        }

    def generate_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("=" * 70)
        report.append(f"SAGE Exploration Analysis: {self.log_file.name}")
        report.append("=" * 70)
        report.append("")

        # Summary
        if self.summary:
            report.append("📊 Summary:")
            for key, value in self.summary.items():
                report.append(f"   {key}: {value}")
            report.append("")

        # Salience analysis
        sal_analysis = self.analyze_salience_patterns()
        if sal_analysis:
            report.append("🎯 Salience Patterns:")
            report.append(f"   Mean: {sal_analysis['mean']:.3f} ± {sal_analysis['std']:.3f}")
            report.append(f"   Range: [{sal_analysis['min']:.3f}, {sal_analysis['max']:.3f}]")
            report.append(f"   Peaks detected: {len(sal_analysis['peaks'])}")
            if sal_analysis['peaks']:
                report.append(f"   Top peaks:")
                for cycle, salience in sorted(sal_analysis['peaks'], key=lambda x: x[1], reverse=True)[:3]:
                    report.append(f"      Cycle {cycle}: {salience:.3f}")
            report.append("")

        # Task analysis
        task_analysis = self.analyze_task_patterns()
        if task_analysis:
            report.append("📋 Task Patterns:")
            report.append(f"   Unique tasks: {task_analysis['unique_tasks']}")
            report.append(f"   Task switches: {task_analysis['total_switches']}")
            report.append(f"   Avg task duration: {task_analysis['avg_task_duration']:.1f} cycles")
            report.append("")

        # Performance
        perf_analysis = self.analyze_performance()
        if perf_analysis:
            report.append("⚡ Performance:")
            report.append(f"   Avg cycle time: {perf_analysis['avg_cycle_time']:.2f}ms")
            report.append(f"   Range: [{perf_analysis['min_cycle_time']:.2f}, {perf_analysis['max_cycle_time']:.2f}]ms")
            report.append(f"   Trend: {perf_analysis['trend']}")
            report.append("")

        # Embodied behavior (if multi-modal)
        embodied = self.analyze_embodied_behavior()
        if embodied:
            report.append("🤖 Embodied Behavior:")
            report.append(f"   Exploration volume: {embodied['exploration_volume']:.4f}")
            report.append(f"   Total distance moved: {embodied['total_distance']:.2f}")
            report.append(f"   Avg movement per cycle: {embodied['avg_movement']:.3f}")
            report.append(f"   Gripper activity: {embodied['gripper_activity']:.3f}")
            report.append(f"   Gripper open ratio: {embodied['gripper_open_ratio']:.1%}")
            report.append("")

        report.append("=" * 70)

        return "\n".join(report)

    def plot_analysis(self, save_path: Optional[Path] = None):
        """Create visualization of exploration data"""
        if not self.cycles:
            print("No data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'SAGE Exploration Analysis: {self.log_file.name}', fontsize=16)

        # 1. Salience over time
        cycles = [c.cycle for c in self.cycles]
        saliences = [c.salience for c in self.cycles]

        axes[0, 0].plot(cycles, saliences, alpha=0.6, label='Raw salience')
        sal_analysis = self.analyze_salience_patterns()
        if 'smoothed' in sal_analysis:
            axes[0, 0].plot(cycles[:len(sal_analysis['smoothed'])],
                          sal_analysis['smoothed'], 'r-', linewidth=2, label='Smoothed')
        axes[0, 0].set_xlabel('Cycle')
        axes[0, 0].set_ylabel('Salience')
        axes[0, 0].set_title('Salience Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Cycle time distribution
        cycle_times = [c.cycle_time_ms for c in self.cycles]
        axes[0, 1].hist(cycle_times, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(cycle_times), color='r', linestyle='--',
                          label=f'Mean: {np.mean(cycle_times):.1f}ms')
        axes[0, 1].set_xlabel('Cycle Time (ms)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Cycle Time Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Performance trend
        window = 20
        if len(cycle_times) >= window:
            rolling_mean = np.convolve(cycle_times, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(cycles[:len(rolling_mean)], rolling_mean, 'b-', linewidth=2)
            axes[1, 0].set_xlabel('Cycle')
            axes[1, 0].set_ylabel('Cycle Time (ms)')
            axes[1, 0].set_title(f'Performance Trend (rolling mean, window={window})')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data for trend',
                           ha='center', va='center', transform=axes[1, 0].transAxes)

        # 4. Embodied behavior or task distribution
        if self.is_multimodal:
            # Plot position trajectory
            positions = np.array([c.position for c in self.cycles if c.position is not None])
            if len(positions) > 0:
                axes[1, 1].plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.6)
                axes[1, 1].scatter(positions[0, 0], positions[0, 1], c='g', s=100, label='Start')
                axes[1, 1].scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, label='End')
                axes[1, 1].set_xlabel('X Position')
                axes[1, 1].set_ylabel('Y Position')
                axes[1, 1].set_title('Position Trajectory (Top-Down)')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        else:
            # Task distribution
            task_analysis = self.analyze_task_patterns()
            if 'task_list' in task_analysis:
                tasks = [c.task for c in self.cycles]
                unique, counts = np.unique(tasks, return_counts=True)
                axes[1, 1].bar(range(len(unique)), counts)
                axes[1, 1].set_xlabel('Task Type')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Task Distribution')
                axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        else:
            plt.show()


def main():
    """Analyze exploration logs"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze SAGE exploration logs")
    parser.add_argument('log_file', type=Path, help='Path to exploration log file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--save', type=Path, help='Save plot to file')

    args = parser.parse_args()

    # Analyze
    analyzer = ExplorationAnalyzer(args.log_file)

    # Generate report
    report = analyzer.generate_report()
    print(report)

    # Plot if requested
    if args.plot or args.save:
        analyzer.plot_analysis(save_path=args.save)


if __name__ == "__main__":
    main()
