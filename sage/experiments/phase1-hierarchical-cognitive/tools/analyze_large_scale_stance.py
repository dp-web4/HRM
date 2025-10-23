"""
Analyze Large-Scale Stance Evolution with SVK

Computes 12D stance vectors for baseline, epoch 60, and epoch 100 datasets
and analyzes how stance evolved during training.
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add SVK to path BEFORE imports
# File is at: sage/experiments/phase1-hierarchical-cognitive/tools/analyze_large_scale_stance.py
# Need 5 .parent to get to HRM root
_svk_path = Path(__file__).parent.parent.parent.parent.parent / "forum/nova/stance-vector-kit/src"
sys.path.insert(0, str(_svk_path))

# Now import from stancekit
from stancekit.feature_extraction import compile_lexicons, extract_features


def load_jsonl(path):
    """Load JSONL file"""
    turns = []
    with open(path, 'r') as f:
        for line in f:
            turns.append(json.loads(line))
    return turns


def compute_stance_from_features(features):
    """
    Compute 12D stance vector from lexical features

    SVK dimensions:
    - EH: Epistemic Humility
    - DC: Declarative Confidence
    - EX: Exploratory Drive
    - MA: Meta-Awareness
    - RR: Revision Readiness
    - AG: Agency
    - AS: Attention Stability
    - SV: Skepticism/Verification
    - VA: Valence
    - AR: Arousal
    - IF: Initiative
    - ED: Evidence Density
    """
    # Aggregate features
    agg = {
        'hedges': np.mean([f['hedges'] for f in features]) if features else 0,
        'modals': np.mean([f['modals'] for f in features]) if features else 0,
        'meta': np.mean([f['meta'] for f in features]) if features else 0,
        'backtrack': np.mean([f['backtrack'] for f in features]) if features else 0,
        'verify': np.mean([f['verify'] for f in features]) if features else 0,
        'q_ratio': np.mean([f['q_ratio'] for f in features]) if features else 0,
    }

    # Map to 12D stance (heuristic scaling)
    stance = {
        'EH': min(1.0, agg['hedges'] * 50),  # Epistemic Humility
        'DC': min(1.0, agg['modals'] * 50),  # Declarative Confidence
        'EX': min(1.0, agg['q_ratio']),      # Exploratory Drive
        'MA': min(1.0, agg['meta'] * 100),   # Meta-Awareness
        'RR': min(1.0, agg['backtrack'] * 100),  # Revision Readiness
        'AG': 0.5,  # Neutral (can't infer from text alone)
        'AS': 0.5,  # Neutral
        'SV': min(1.0, agg['verify'] * 100),  # Skepticism/Verification
        'VA': 0.5,  # Neutral
        'AR': 0.5,  # Neutral
        'IF': min(1.0, agg['q_ratio']),  # Initiative (similar to EX)
        'ED': min(1.0, (agg['hedges'] + agg['verify']) * 25),  # Evidence Density
    }

    return stance, agg


def analyze_dataset(jsonl_path, name, lexicons):
    """Analyze a single dataset"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")

    turns = load_jsonl(jsonl_path)
    print(f"Loaded {len(turns)} turns")

    # Extract features for all turns using SVK
    features = extract_features(turns, lexicons)

    # Compute stance vector
    stance, agg_features = compute_stance_from_features(features)

    # Print results
    print(f"\nAggregated Lexical Features:")
    for k, v in agg_features.items():
        print(f"  {k:12s}: {v:.4f}")

    print(f"\n12D Stance Vector:")
    for k, v in stance.items():
        print(f"  {k:3s}: {v:.4f}")

    return {
        'name': name,
        'turns': len(turns),
        'features': agg_features,
        'stance': stance,
        'raw_features': features
    }


def compare_datasets(results):
    """Compare stance vectors across datasets"""
    print(f"\n{'='*60}")
    print(f"Cross-Dataset Comparison")
    print(f"{'='*60}")

    # Get stance vectors
    baseline_stance = np.array([results['baseline']['stance'][k] for k in sorted(results['baseline']['stance'].keys())])
    epoch60_stance = np.array([results['epoch_60']['stance'][k] for k in sorted(results['epoch_60']['stance'].keys())])
    epoch100_stance = np.array([results['epoch_100']['stance'][k] for k in sorted(results['epoch_100']['stance'].keys())])

    # Compute cosine similarities
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    sim_base_60 = cosine_sim(baseline_stance, epoch60_stance)
    sim_base_100 = cosine_sim(baseline_stance, epoch100_stance)
    sim_60_100 = cosine_sim(epoch60_stance, epoch100_stance)

    print(f"\nCosine Similarities:")
    print(f"  Baseline ↔ Epoch 60:  {sim_base_60:.4f}")
    print(f"  Baseline ↔ Epoch 100: {sim_base_100:.4f}")
    print(f"  Epoch 60 ↔ Epoch 100: {sim_60_100:.4f}")

    # Dimension-wise changes
    print(f"\nDimension-wise Evolution:")
    print(f"  {'Dim':3s}  {'Baseline':>8s}  {'Epoch60':>8s}  {'Epoch100':>8s}  {'Δ60':>8s}  {'Δ100':>8s}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    dims = sorted(results['baseline']['stance'].keys())
    for i, dim in enumerate(dims):
        b = baseline_stance[i]
        e60 = epoch60_stance[i]
        e100 = epoch100_stance[i]
        delta60 = e60 - b
        delta100 = e100 - b
        print(f"  {dim:3s}  {b:8.4f}  {e60:8.4f}  {e100:8.4f}  {delta60:+8.4f}  {delta100:+8.4f}")

    # Find biggest changes
    deltas_60 = {dim: epoch60_stance[i] - baseline_stance[i] for i, dim in enumerate(dims)}
    deltas_100 = {dim: epoch100_stance[i] - baseline_stance[i] for i, dim in enumerate(dims)}

    print(f"\nBiggest Changes (Epoch 60):")
    for dim, delta in sorted(deltas_60.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {dim:3s}: {delta:+.4f}")

    print(f"\nBiggest Changes (Epoch 100):")
    for dim, delta in sorted(deltas_100.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {dim:3s}: {delta:+.4f}")


def main():
    base_dir = Path("svk_analysis/large_scale")

    # Load SVK lexicons
    kit_root = Path(__file__).parent.parent.parent.parent.parent / "forum/nova/stance-vector-kit"
    lexicons = compile_lexicons(kit_root)
    print(f"Loaded SVK lexicons from {kit_root}")

    # Analyze each dataset
    results = {}
    results['baseline'] = analyze_dataset(base_dir / "baseline.jsonl", "Baseline (Pretrained Phi-1.5)", lexicons)
    results['epoch_60'] = analyze_dataset(base_dir / "epoch_60.jsonl", "Epoch 60", lexicons)
    results['epoch_100'] = analyze_dataset(base_dir / "epoch_100.jsonl", "Epoch 100", lexicons)

    # Compare
    compare_datasets(results)

    # Save results
    output_dir = base_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    with open(output_dir / "stance_analysis.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for dataset in results.values():
            dataset['raw_features'] = [
                {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in feat.items()}
                for feat in dataset['raw_features']
            ]
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
