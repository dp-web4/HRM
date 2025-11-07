#!/usr/bin/env python3
"""
Embodied Behavior Visualizer
=============================

Visualize embodied exploration patterns:
- 3D movement trajectories
- Action type distributions
- Salience heatmaps overlaid on movement
- Position-action correlations
- Task switching patterns

Use this to understand spatial exploration and action patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re


class EmbodiedVisualizer:
    """Visualize embodied exploration behavior"""

    def __init__(self, log_file: Path):
        self.log_file = Path(log_file)
        self.cycles = []
        self.positions = []
        self.actions = []
        self.saliences = []
        self.tasks = []

        if self.log_file.exists():
            self._parse_log()

    def _parse_log(self):
        """Parse embodied exploration log"""
        print(f"📊 Parsing: {self.log_file.name}")

        with open(self.log_file, 'r') as f:
            content = f.read()

        # Parse cycle lines with action information
        pattern = r'Cycle\s+(\d+)\s+\|\s+Task:\s+(.{35})\s+\|\s+Action:\s+(\w+)\s+\|\s+Pos:\s+\[\s*([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]\s+\|\s+Salience:\s+([+-]?\d+\.\d+)'

        for match in re.finditer(pattern, content):
            self.cycles.append(int(match.group(1)))
            self.tasks.append(match.group(2).strip())
            self.actions.append(match.group(3))
            self.positions.append([
                float(match.group(4)),
                float(match.group(5)),
                float(match.group(6))
            ])
            self.saliences.append(float(match.group(7)))

        print(f"   ✓ Parsed {len(self.cycles)} cycles with {len(set(self.actions))} action types")

    def plot_3d_trajectory(self, save_path: Optional[Path] = None,
                           color_by: str = 'salience'):
        """
        Plot 3D movement trajectory.

        Args:
            color_by: 'salience', 'action', or 'time'
        """
        if not self.positions:
            print("No position data to plot")
            return

        positions = np.array(self.positions)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Color mapping
        if color_by == 'salience':
            colors = self.saliences
            cmap = plt.cm.viridis
            cbar_label = 'Salience'
        elif color_by == 'action':
            # Map actions to numbers
            unique_actions = list(set(self.actions))
            action_to_num = {a: i for i, a in enumerate(unique_actions)}
            colors = [action_to_num[a] for a in self.actions]
            cmap = plt.cm.tab10
            cbar_label = 'Action Type'
        else:  # time
            colors = self.cycles
            cmap = plt.cm.plasma
            cbar_label = 'Cycle'

        # Plot trajectory
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                           c=colors, cmap=cmap, s=50, alpha=0.6,
                           edgecolors='black', linewidth=0.5)

        # Connect consecutive positions
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
               'gray', alpha=0.3, linewidth=1)

        # Mark start and end
        ax.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]],
                  c='green', s=200, marker='*', label='Start', edgecolors='black', linewidth=2)
        ax.scatter([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]],
                  c='red', s=200, marker='X', label='End', edgecolors='black', linewidth=2)

        # Labels and formatting
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_zlabel('Z Position', fontsize=12)
        ax.set_title(f'Embodied Exploration Trajectory\n({len(self.cycles)} cycles, color by {color_by})',
                    fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label(cbar_label, fontsize=11)

        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Equal aspect ratio for better visualization
        max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
                             positions[:, 1].max()-positions[:, 1].min(),
                             positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0

        mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved 3D trajectory to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_action_distribution(self, save_path: Optional[Path] = None):
        """Plot distribution of action types"""
        if not self.actions:
            print("No action data to plot")
            return

        action_counts = {}
        for action in self.actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        fig, ax = plt.subplots(figsize=(10, 6))

        actions = list(action_counts.keys())
        counts = list(action_counts.values())

        bars = ax.bar(actions, counts, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Color bars
        colors = plt.cm.Set3(np.linspace(0, 1, len(actions)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel('Action Type', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Action Distribution\n({len(self.actions)} total actions, {len(actions)} types)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved action distribution to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_salience_over_time(self, save_path: Optional[Path] = None):
        """Plot salience evolution with action markers"""
        if not self.saliences:
            print("No salience data to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Top: Salience over time
        ax1.plot(self.cycles, self.saliences, 'b-', alpha=0.6, linewidth=1.5, label='Salience')

        # Smooth salience
        window = min(20, len(self.saliences) // 5)
        if len(self.saliences) >= window:
            smoothed = np.convolve(self.saliences, np.ones(window)/window, mode='valid')
            ax1.plot(self.cycles[:len(smoothed)], smoothed, 'r-', linewidth=2.5, label='Smoothed')

        ax1.set_ylabel('Salience', fontsize=12)
        ax1.set_title('Salience and Actions Over Time', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Bottom: Action timeline
        unique_actions = list(set(self.actions))
        action_to_y = {a: i for i, a in enumerate(unique_actions)}
        y_positions = [action_to_y[a] for a in self.actions]

        # Color by action type
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_actions)))
        action_colors = [colors[action_to_y[a]] for a in self.actions]

        ax2.scatter(self.cycles, y_positions, c=action_colors, s=50, alpha=0.7,
                   edgecolors='black', linewidth=0.5)
        ax2.set_yticks(range(len(unique_actions)))
        ax2.set_yticklabels(unique_actions, fontsize=10)
        ax2.set_xlabel('Cycle', fontsize=12)
        ax2.set_ylabel('Action Type', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved salience timeline to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_movement_heatmap_2d(self, save_path: Optional[Path] = None, plane='xy'):
        """
        Plot 2D heatmap of positions.

        Args:
            plane: 'xy', 'xz', or 'yz'
        """
        if not self.positions:
            print("No position data to plot")
            return

        positions = np.array(self.positions)

        # Select plane
        if plane == 'xy':
            x, y = positions[:, 0], positions[:, 1]
            xlabel, ylabel = 'X Position', 'Y Position'
        elif plane == 'xz':
            x, y = positions[:, 0], positions[:, 2]
            xlabel, ylabel = 'X Position', 'Z Position'
        else:  # yz
            x, y = positions[:, 1], positions[:, 2]
            xlabel, ylabel = 'Y Position', 'Z Position'

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=30)

        # Plot heatmap
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', alpha=0.6)

        # Overlay trajectory
        ax.plot(x, y, 'cyan', alpha=0.4, linewidth=1, linestyle='--')
        ax.scatter(x, y, c=self.saliences, cmap='viridis', s=30, alpha=0.7,
                  edgecolors='white', linewidth=0.5)

        # Mark start and end
        ax.scatter([x[0]], [y[0]], c='green', s=300, marker='*',
                  label='Start', edgecolors='black', linewidth=2, zorder=5)
        ax.scatter([x[-1]], [y[-1]], c='red', s=300, marker='X',
                  label='End', edgecolors='black', linewidth=2, zorder=5)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Movement Heatmap ({plane.upper()} plane)\n{len(self.cycles)} cycles',
                    fontsize=14, fontweight='bold')

        # Colorbars
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Visit Frequency', fontsize=11)

        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved movement heatmap to {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_full_report(self, output_dir: Path):
        """Generate complete visualization report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print(f"📊 Generating full embodied visualization report...")
        print(f"   Output directory: {output_dir}")

        # 1. 3D trajectory (colored by salience)
        print("   1/6 Creating 3D trajectory (salience)...")
        self.plot_3d_trajectory(
            save_path=output_dir / '1_trajectory_3d_salience.png',
            color_by='salience'
        )

        # 2. 3D trajectory (colored by action)
        print("   2/6 Creating 3D trajectory (action)...")
        self.plot_3d_trajectory(
            save_path=output_dir / '2_trajectory_3d_action.png',
            color_by='action'
        )

        # 3. Action distribution
        print("   3/6 Creating action distribution...")
        self.plot_action_distribution(
            save_path=output_dir / '3_action_distribution.png'
        )

        # 4. Salience timeline
        print("   4/6 Creating salience timeline...")
        self.plot_salience_over_time(
            save_path=output_dir / '4_salience_timeline.png'
        )

        # 5. XY heatmap
        print("   5/6 Creating XY movement heatmap...")
        self.plot_movement_heatmap_2d(
            save_path=output_dir / '5_movement_heatmap_xy.png',
            plane='xy'
        )

        # 6. XZ heatmap
        print("   6/6 Creating XZ movement heatmap...")
        self.plot_movement_heatmap_2d(
            save_path=output_dir / '6_movement_heatmap_xz.png',
            plane='xz'
        )

        print(f"\n✓ Full report generated in {output_dir}")
        print(f"   Generated 6 visualizations")


def main():
    """Visualize embodied exploration logs"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize embodied exploration behavior"
    )
    parser.add_argument('log_file', type=Path,
                       help='Path to embodied exploration log file')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory for visualizations')
    parser.add_argument('--3d', action='store_true',
                       help='Show 3D trajectory')
    parser.add_argument('--actions', action='store_true',
                       help='Show action distribution')
    parser.add_argument('--salience', action='store_true',
                       help='Show salience timeline')
    parser.add_argument('--heatmap', action='store_true',
                       help='Show movement heatmap')

    args = parser.parse_args()

    # Create visualizer
    viz = EmbodiedVisualizer(args.log_file)

    if args.output_dir:
        # Generate full report
        viz.generate_full_report(args.output_dir)
    else:
        # Show individual plots
        if args._3d or not any([args.actions, args.salience, args.heatmap]):
            viz.plot_3d_trajectory(color_by='salience')

        if args.actions:
            viz.plot_action_distribution()

        if args.salience:
            viz.plot_salience_over_time()

        if args.heatmap:
            viz.plot_movement_heatmap_2d(plane='xy')


if __name__ == "__main__":
    main()
