"""
Comprehensive Three-Way Comparison with SVK

Compares all three approaches to epistemic stance:
1. Baseline (pretrained Phi-1.5, no modification)
2. Fine-tuned (6 examples × 100 epochs, weight perturbation)
3. Orchestrated (architectural framing, zero training)

Uses SVK 12D stance analysis to quantify differences.
"""

import json
import numpy as np
from pathlib import Path
import sys

# Add SVK to path
_svk_path = Path(__file__).parent.parent.parent.parent.parent / "forum/nova/stance-vector-kit/src"
sys.path.insert(0, str(_svk_path))

from stancekit.feature_extraction import compile_lexicons, extract_features


def load_jsonl(path):
    """Load JSONL file"""
    turns = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    return turns


def compute_stance_vector(features):
    """Compute 12D stance vector from lexical features"""
    if not features:
        return None, None

    # Aggregate features
    agg = {
        'hedges': np.mean([f['hedges'] for f in features]),
        'modals': np.mean([f['modals'] for f in features]),
        'meta': np.mean([f['meta'] for f in features]),
        'backtrack': np.mean([f['backtrack'] for f in features]),
        'action': np.mean([f['action'] for f in features]),
        'verify': np.mean([f['verify'] for f in features]),
        'q_ratio': np.mean([f['q_ratio'] for f in features]),
        'exclaim': np.mean([f['exclaim'] for f in features]),
        'pos': np.mean([f['pos'] for f in features]),
        'neg': np.mean([f['neg'] for f in features]),
    }

    # Map to 12D stance (heuristic scaling)
    stance = {
        'EH': min(1.0, agg['hedges'] * 50),  # Epistemic Humility
        'DC': min(1.0, agg['modals'] * 50),  # Declarative Confidence
        'EX': min(1.0, agg['q_ratio']),      # Exploratory Drive
        'MA': min(1.0, agg['meta'] * 100),   # Meta-Awareness
        'RR': min(1.0, agg['backtrack'] * 100),  # Revision Readiness
        'AG': min(1.0, agg['action'] * 50),  # Agency
        'AS': max(0.0, 1.0 - agg['q_ratio']),  # Attention Stability
        'SV': min(1.0, agg['verify'] * 50),  # Skepticism/Verification
        'VA': max(0.0, min(1.0, (agg['pos'] - agg['neg'] + 1.0) / 2.0)),  # Valence
        'AR': min(1.0, agg['exclaim'] * 200),  # Arousal
        'IF': min(1.0, agg['action'] * 50),  # Initiative
        'ED': min(1.0, agg['verify'] * 50),  # Evidence Density
    }

    return stance, agg


def analyze_approach(jsonl_path, name, lexicons):
    """Analyze single approach"""
    print(f"\n{'='*70}")
    print(f"Analyzing: {name}")
    print(f"{'='*70}")

    if not jsonl_path.exists():
        print(f"File not found: {jsonl_path}")
        return None

    turns = load_jsonl(jsonl_path)
    print(f"Loaded {len(turns)} turns")

    # Extract features
    features = extract_features(turns, lexicons)

    # Compute stance
    stance, agg_features = compute_stance_vector(features)

    if stance is None:
        print("No stance computed")
        return None

    # Print results
    print(f"\nAggregated Lexical Features:")
    for k, v in agg_features.items():
        print(f"  {k:12s}: {v:.4f}")

    print(f"\n12D Stance Vector:")
    for k, v in sorted(stance.items()):
        print(f"  {k:3s}: {v:.4f}")

    return {
        'name': name,
        'turns': len(turns),
        'features': agg_features,
        'stance': stance,
        'raw_features': features
    }


def compare_approaches(results):
    """Three-way comparison"""
    print(f"\n{'='*70}")
    print("THREE-WAY COMPARISON")
    print(f"{'='*70}")

    approaches = ['baseline', 'fine_tuned', 'orchestrated']

    # Check what we have
    available = {k: v for k, v in results.items() if v is not None}

    if len(available) < 2:
        print(f"Not enough data for comparison (have {len(available)} approaches)")
        return

    print(f"\nAvailable approaches: {list(available.keys())}")

    # Extract stance vectors
    vectors = {}
    for approach in approaches:
        if approach in available:
            stance = available[approach]['stance']
            vectors[approach] = np.array([stance[k] for k in sorted(stance.keys())])

    # Cosine similarities
    print(f"\nCosine Similarities:")
    for i, app1 in enumerate(list(vectors.keys())):
        for app2 in list(vectors.keys())[i+1:]:
            v1, v2 = vectors[app1], vectors[app2]
            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            print(f"  {app1:15s} ↔ {app2:15s}: {cos_sim:.4f}")

    # Dimension-wise comparison
    if len(vectors) >= 2:
        print(f"\nDimension-wise Comparison:")
        dims = sorted(available[list(available.keys())[0]]['stance'].keys())

        # Header
        header = "  Dim"
        for approach in vectors.keys():
            header += f"  {approach[:12]:>12s}"
        print(header)
        print("  " + "-"*3 + "  " + "  ".join(["-"*12 for _ in vectors]))

        # Values
        for dim in dims:
            row = f"  {dim:3s}"
            for approach in vectors.keys():
                idx = dims.index(dim)
                value = vectors[approach][idx]
                row += f"  {value:12.4f}"
            print(row)

    # Biggest differences (if we have baseline + others)
    if 'baseline' in vectors:
        print(f"\nChanges from Baseline:")
        baseline_vec = vectors['baseline']
        dims = sorted(available['baseline']['stance'].keys())

        for approach in vectors.keys():
            if approach == 'baseline':
                continue

            print(f"\n{approach.upper()} vs Baseline:")
            deltas = {dim: vectors[approach][i] - baseline_vec[i]
                     for i, dim in enumerate(dims)}

            for dim, delta in sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                direction = "↑" if delta > 0 else "↓"
                print(f"  {dim:3s}: {delta:+.4f} {direction}")


def main():
    print(f"\n{'='*70}")
    print("COMPREHENSIVE THREE-WAY STANCE COMPARISON")
    print(f"{'='*70}\n")

    # Load SVK lexicons
    kit_root = Path(__file__).parent.parent.parent.parent.parent / "forum/nova/stance-vector-kit"
    lexicons = compile_lexicons(kit_root)
    print(f"Loaded SVK lexicons from {kit_root}\n")

    # Paths
    svk_dir = Path("svk_analysis/large_scale")

    # Analyze each approach
    results = {}
    results['baseline'] = analyze_approach(
        svk_dir / "baseline.jsonl",
        "Baseline (Pretrained Phi-1.5)",
        lexicons
    )
    results['fine_tuned'] = analyze_approach(
        svk_dir / "epoch_60.jsonl",
        "Fine-Tuned (6 examples, 60 epochs)",
        lexicons
    )
    results['orchestrated'] = analyze_approach(
        svk_dir / "orchestrated_full.jsonl",
        "Orchestrated (Architectural, Zero Training)",
        lexicons
    )

    # Compare
    compare_approaches(results)

    # Save results
    output_dir = svk_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "three_way_comparison.json"
    with open(output_file, 'w') as f:
        # Convert numpy arrays for JSON
        for approach in results.values():
            if approach and 'raw_features' in approach:
                approach['raw_features'] = [
                    {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in feat.items()}
                    for feat in approach['raw_features']
                ]
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
