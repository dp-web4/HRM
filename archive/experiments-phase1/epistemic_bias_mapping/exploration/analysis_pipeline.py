#!/usr/bin/env python3
"""
Analysis Pipeline

Analyzes experimental results to detect scaffolding suitability threshold.

Identifies where scaffolding switches from harmful to helpful based on:
- Enhanced energy scores
- Pattern collapse rates
- Coherence metrics
- Cross-experiment comparisons
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from research_db import ResearchDB


class AnalysisPipeline:
    """Analyzes experimental results for threshold detection"""

    def __init__(self, db_path: str = "./exploration/research.db"):
        self.db = ResearchDB(db_path)
        self.output_dir = Path("./exploration/analysis_output")
        self.output_dir.mkdir(exist_ok=True)

    def run_full_analysis(self) -> Dict:
        """Run complete analysis pipeline"""

        print("\n" + "="*80)
        print("SCAFFOLDING SUITABILITY THRESHOLD ANALYSIS")
        print("="*80 + "\n")

        analysis = {
            'timestamp': datetime.now().isoformat(),
            'threshold_location': None,
            'experiments_summary': {},
            'comparisons': [],
            'visualizations': [],
            'key_findings': []
        }

        # 1. Summarize all experiments
        print("1. Summarizing experiments...")
        analysis['experiments_summary'] = self._summarize_experiments()

        # 2. Compare bare vs scaffolded at each size
        print("2. Comparing bare vs scaffolded...")
        analysis['comparisons'] = self._compare_approaches()

        # 3. Detect threshold crossing
        print("3. Detecting threshold crossing...")
        analysis['threshold_location'] = self._detect_threshold()

        # 4. Generate visualizations
        print("4. Generating visualizations...")
        analysis['visualizations'] = self._generate_visualizations()

        # 5. Extract key findings
        print("5. Extracting key findings...")
        analysis['key_findings'] = self._extract_findings(analysis)

        # 6. Export report
        print("6. Exporting report...")
        self._export_report(analysis)

        print("\n✓ Analysis complete!")
        print(f"   Output directory: {self.output_dir}")

        return analysis

    def _summarize_experiments(self) -> Dict:
        """Summarize all completed experiments"""

        query = """
            SELECT
                e.training_size,
                e.scaffolding_type,
                m.avg_enhanced_energy,
                m.pattern_collapse_rate,
                m.on_topic_rate,
                m.epistemic_humility_rate,
                m.total_turns
            FROM experiments e
            LEFT JOIN metrics m ON e.id = m.experiment_id
            WHERE e.status = 'completed'
            ORDER BY e.training_size, e.scaffolding_type
        """

        df = pd.read_sql_query(query, self.db.conn)

        summary = {}
        for size in df['training_size'].unique():
            summary[int(size)] = {}
            for stype in df['scaffolding_type'].unique():
                row = df[(df['training_size'] == size) & (df['scaffolding_type'] == stype)]
                if not row.empty:
                    summary[int(size)][stype] = {
                        'energy': float(row['avg_enhanced_energy'].iloc[0]) if pd.notna(row['avg_enhanced_energy'].iloc[0]) else None,
                        'collapse_rate': float(row['pattern_collapse_rate'].iloc[0]) if pd.notna(row['pattern_collapse_rate'].iloc[0]) else None,
                        'on_topic_rate': float(row['on_topic_rate'].iloc[0]) if pd.notna(row['on_topic_rate'].iloc[0]) else None,
                        'humility_rate': float(row['epistemic_humility_rate'].iloc[0]) if pd.notna(row['epistemic_humility_rate'].iloc[0]) else None
                    }

        return summary

    def _compare_approaches(self) -> List[Dict]:
        """Compare bare vs scaffolded at each training size"""

        comparisons = []

        summary = self._summarize_experiments()

        for size, data in sorted(summary.items()):
            if 'bare' not in data or 'full_irp' not in data:
                continue

            bare = data['bare']
            scaffolded = data['full_irp']

            if bare['energy'] is None or scaffolded['energy'] is None:
                continue

            # Compare metrics
            comparison = {
                'training_size': size,
                'bare_energy': bare['energy'],
                'scaffolded_energy': scaffolded['energy'],
                'energy_improvement': bare['energy'] - scaffolded['energy'],
                'bare_better': bare['energy'] < scaffolded['energy'],
                'scaffolded_better': scaffolded['energy'] < bare['energy'],
                'bare_collapse_rate': bare['collapse_rate'],
                'scaffolded_collapse_rate': scaffolded['collapse_rate']
            }

            comparisons.append(comparison)

            # Create comparison record in database
            # (Would need experiment IDs - simplified here)

            print(f"  {size} examples:")
            print(f"    Bare energy:       {bare['energy']:.3f}")
            print(f"    Scaffolded energy: {scaffolded['energy']:.3f}")
            print(f"    Winner: {'BARE' if comparison['bare_better'] else 'SCAFFOLDED'}")

        return comparisons

    def _detect_threshold(self) -> Optional[Dict]:
        """Detect where scaffolding switches from harmful to helpful"""

        comparisons = self._compare_approaches()

        if len(comparisons) < 2:
            print("  ⚠️ Insufficient data for threshold detection")
            return None

        # Find crossing point
        prev_winner = None
        threshold = None

        for comp in sorted(comparisons, key=lambda x: x['training_size']):
            current_winner = 'bare' if comp['bare_better'] else 'scaffolded'

            if prev_winner and prev_winner != current_winner:
                # Threshold crossed!
                threshold = {
                    'lower_bound': comparisons[comparisons.index(comp) - 1]['training_size'],
                    'upper_bound': comp['training_size'],
                    'estimate': (comparisons[comparisons.index(comp) - 1]['training_size'] + comp['training_size']) / 2,
                    'transition': f"{prev_winner} → {current_winner}"
                }
                print(f"\n  🎯 THRESHOLD DETECTED:")
                print(f"     Location: Between {threshold['lower_bound']} and {threshold['upper_bound']} examples")
                print(f"     Estimate: ~{threshold['estimate']:.0f} examples")
                print(f"     Transition: {threshold['transition']}")
                break

            prev_winner = current_winner

        if not threshold:
            # Determine current state
            if comparisons[0]['bare_better']:
                print(f"\n  📍 THRESHOLD NOT YET CROSSED:")
                print(f"     All sizes favor bare (scaffolding still harmful)")
            else:
                print(f"\n  📍 THRESHOLD ALREADY CROSSED:")
                print(f"     All sizes favor scaffolded (scaffolding beneficial)")

        return threshold

    def _generate_visualizations(self) -> List[str]:
        """Generate analysis visualizations"""

        visualizations = []

        # 1. Energy comparison plot
        try:
            fig_path = self._plot_energy_comparison()
            visualizations.append(str(fig_path))
            print(f"  ✓ Energy comparison: {fig_path}")
        except Exception as e:
            print(f"  ✗ Energy plot failed: {e}")

        # 2. Pattern collapse rates
        try:
            fig_path = self._plot_collapse_rates()
            visualizations.append(str(fig_path))
            print(f"  ✓ Collapse rates: {fig_path}")
        except Exception as e:
            print(f"  ✗ Collapse plot failed: {e}")

        # 3. Threshold crossing
        try:
            fig_path = self._plot_threshold_crossing()
            visualizations.append(str(fig_path))
            print(f"  ✓ Threshold crossing: {fig_path}")
        except Exception as e:
            print(f"  ✗ Threshold plot failed: {e}")

        return visualizations

    def _plot_energy_comparison(self) -> Path:
        """Plot bare vs scaffolded energy across training sizes"""

        summary = self._summarize_experiments()

        sizes = []
        bare_energies = []
        scaffolded_energies = []

        for size, data in sorted(summary.items()):
            if 'bare' in data and 'full_irp' in data:
                if data['bare']['energy'] is not None and data['full_irp']['energy'] is not None:
                    sizes.append(size)
                    bare_energies.append(data['bare']['energy'])
                    scaffolded_energies.append(data['full_irp']['energy'])

        plt.figure(figsize=(10, 6))
        plt.plot(sizes, bare_energies, 'o-', label='Bare (no scaffolding)', linewidth=2, markersize=8)
        plt.plot(sizes, scaffolded_energies, 's-', label='Full IRP scaffolding', linewidth=2, markersize=8)
        plt.xlabel('Training Set Size (examples)', fontsize=12)
        plt.ylabel('Enhanced Energy (lower = better)', fontsize=12)
        plt.title('Scaffolding Energy Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        path = self.output_dir / "energy_comparison.png"
        plt.savefig(path, dpi=150)
        plt.close()

        return path

    def _plot_collapse_rates(self) -> Path:
        """Plot pattern collapse rates"""

        summary = self._summarize_experiments()

        sizes = []
        bare_collapse = []
        scaffolded_collapse = []

        for size, data in sorted(summary.items()):
            if 'bare' in data and 'full_irp' in data:
                if data['bare']['collapse_rate'] is not None and data['full_irp']['collapse_rate'] is not None:
                    sizes.append(size)
                    bare_collapse.append(data['bare']['collapse_rate'])
                    scaffolded_collapse.append(data['full_irp']['collapse_rate'])

        plt.figure(figsize=(10, 6))
        plt.plot(sizes, bare_collapse, 'o-', label='Bare', linewidth=2, markersize=8)
        plt.plot(sizes, scaffolded_collapse, 's-', label='Full IRP', linewidth=2, markersize=8)
        plt.xlabel('Training Set Size (examples)', fontsize=12)
        plt.ylabel('Pattern Collapse Rate', fontsize=12)
        plt.title('Pattern Collapse Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        path = self.output_dir / "collapse_rates.png"
        plt.savefig(path, dpi=150)
        plt.close()

        return path

    def _plot_threshold_crossing(self) -> Path:
        """Plot threshold crossing visualization"""

        comparisons = self._compare_approaches()

        sizes = [c['training_size'] for c in comparisons]
        improvements = [c['energy_improvement'] for c in comparisons]

        plt.figure(figsize=(10, 6))
        colors = ['red' if x < 0 else 'green' for x in improvements]
        plt.bar(sizes, improvements, color=colors, alpha=0.6)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.xlabel('Training Set Size (examples)', fontsize=12)
        plt.ylabel('Energy Improvement (bare - scaffolded)', fontsize=12)
        plt.title('Scaffolding Effectiveness Threshold', fontsize=14, fontweight='bold')
        plt.text(0.02, 0.98, 'Green = scaffolding helps\nRed = scaffolding hurts',
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        path = self.output_dir / "threshold_crossing.png"
        plt.savefig(path, dpi=150)
        plt.close()

        return path

    def _extract_findings(self, analysis: Dict) -> List[str]:
        """Extract key findings from analysis"""

        findings = []

        # Finding 1: Threshold location
        if analysis['threshold_location']:
            threshold = analysis['threshold_location']
            findings.append(
                f"Scaffolding suitability threshold located between {threshold['lower_bound']} "
                f"and {threshold['upper_bound']} training examples (estimate: ~{threshold['estimate']:.0f})"
            )

        # Finding 2: Small model behavior
        summary = analysis['experiments_summary']
        if 25 in summary:
            if 'bare' in summary[25] and 'full_irp' in summary[25]:
                bare_energy = summary[25]['bare']['energy']
                scaf_energy = summary[25]['full_irp']['energy']
                if bare_energy and scaf_energy and bare_energy < scaf_energy:
                    findings.append(
                        f"Phase 1 (25 examples) performs BETTER without scaffolding "
                        f"(bare: {bare_energy:.3f} vs scaffolded: {scaf_energy:.3f})"
                    )

        # Finding 3: Large model behavior
        if 115 in summary:
            if 'bare' in summary[115] and 'full_irp' in summary[115]:
                bare_energy = summary[115]['bare']['energy']
                scaf_energy = summary[115]['full_irp']['energy']
                if bare_energy and scaf_energy and scaf_energy < bare_energy:
                    findings.append(
                        f"Phase 2.1 (115 examples) performs BETTER with scaffolding "
                        f"(bare: {bare_energy:.3f} vs scaffolded: {scaf_energy:.3f})"
                    )

        # Finding 4: Pattern collapse correlation
        comparisons = analysis['comparisons']
        if comparisons:
            small_collapse = [c for c in comparisons if c['training_size'] <= 60]
            if small_collapse:
                avg_bare_collapse = sum(c['bare_collapse_rate'] for c in small_collapse) / len(small_collapse)
                avg_scaf_collapse = sum(c['scaffolded_collapse_rate'] for c in small_collapse) / len(small_collapse)
                if avg_scaf_collapse > avg_bare_collapse:
                    findings.append(
                        f"Small models (<60 examples) show higher pattern collapse with scaffolding "
                        f"({avg_scaf_collapse:.1%} vs {avg_bare_collapse:.1%})"
                    )

        return findings

    def _export_report(self, analysis: Dict):
        """Export analysis report"""

        # JSON export
        json_path = self.output_dir / "threshold_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"  ✓ JSON report: {json_path}")

        # Markdown export
        md_path = self.output_dir / "ANALYSIS_REPORT.md"
        self._write_markdown_report(md_path, analysis)
        print(f"  ✓ Markdown report: {md_path}")

    def _write_markdown_report(self, path: Path, analysis: Dict):
        """Write markdown analysis report"""

        with open(path, 'w') as f:
            f.write("# Scaffolding Suitability Threshold Analysis\n\n")
            f.write(f"**Generated**: {analysis['timestamp']}\n\n")
            f.write("---\n\n")

            # Threshold
            f.write("## Threshold Detection\n\n")
            if analysis['threshold_location']:
                threshold = analysis['threshold_location']
                f.write(f"**🎯 THRESHOLD LOCATED**\n\n")
                f.write(f"- **Location**: Between {threshold['lower_bound']} and {threshold['upper_bound']} examples\n")
                f.write(f"- **Estimate**: ~{threshold['estimate']:.0f} training examples\n")
                f.write(f"- **Transition**: {threshold['transition']}\n\n")
            else:
                f.write("Threshold not detected in current data.\n\n")

            # Key Findings
            f.write("## Key Findings\n\n")
            for i, finding in enumerate(analysis['key_findings'], 1):
                f.write(f"{i}. {finding}\n")
            f.write("\n")

            # Comparisons Table
            f.write("## Bare vs Scaffolded Comparison\n\n")
            f.write("| Training Size | Bare Energy | Scaffolded Energy | Winner |\n")
            f.write("|---------------|-------------|-------------------|--------|\n")
            for comp in analysis['comparisons']:
                winner = "**BARE**" if comp['bare_better'] else "**SCAFFOLDED**"
                f.write(f"| {comp['training_size']} | {comp['bare_energy']:.3f} | "
                       f"{comp['scaffolded_energy']:.3f} | {winner} |\n")
            f.write("\n")

            # Visualizations
            f.write("## Visualizations\n\n")
            for viz_path in analysis['visualizations']:
                name = Path(viz_path).name
                f.write(f"- ![{name}]({name})\n")
            f.write("\n")

            # Raw Data
            f.write("## Experimental Data\n\n")
            f.write("```json\n")
            json.dump(analysis['experiments_summary'], f, indent=2)
            f.write("\n```\n")

    def close(self):
        """Close database connection"""
        self.db.close()


if __name__ == "__main__":
    """Run analysis on current experimental data"""

    print("Analysis Pipeline")
    print("="*80)

    pipeline = AnalysisPipeline()

    try:
        analysis = pipeline.run_full_analysis()

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        if analysis['threshold_location']:
            t = analysis['threshold_location']
            print(f"\n🎯 Threshold: ~{t['estimate']:.0f} examples")
        else:
            print(f"\n⏳ Threshold: Not yet detected")

        print(f"\nKey findings: {len(analysis['key_findings'])}")
        for finding in analysis['key_findings']:
            print(f"  • {finding}")

        print(f"\nVisualizations: {len(analysis['visualizations'])}")
        print(f"Output directory: {pipeline.output_dir}")

    finally:
        pipeline.close()

    print("\n✓ Analysis complete!")
