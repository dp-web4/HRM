"""
Custom Stance Analysis for Phi Experiments

Analyzes stance evolution across epochs using lexicon-based features
(no classifier training needed for small datasets)
"""

import json
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add SVK to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'forum/nova/stance-vector-kit/src'))

from stancekit.feature_extraction import compile_lexicons, extract_features
from stancekit.eval import cosine_similarity, flicker_index


def load_jsonl(path):
    """Load JSONL file"""
    turns = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    return turns


def compute_stance_vector_from_features(feats):
    """
    Compute stance vector directly from lexicon features
    (no classifier needed)

    Uses heuristic mapping:
    - EH (Epistemic Humility) ← hedges
    - DC (Declarative Confidence) ← modals
    - EX (Exploratory Drive) ← question ratio
    - MA (Meta-Awareness) ← meta markers
    - RR (Revision Readiness) ← backtracks
    - AG (Agency) ← action verbs
    - SV (Skepticism/Verification) ← verify markers
    - VA (Valence) ← positive - negative
    - AR (Arousal) ← exclamation marks
    - IF (Initiative) ← action verbs
    - ED (Evidence Density) ← verify markers
    - AS (Attention Stability) ← inverse of question ratio
    """
    if not feats:
        return None

    # Aggregate features across all turns
    agg = {
        'hedges': np.mean([f['hedges'] for f in feats]),
        'modals': np.mean([f['modals'] for f in feats]),
        'meta': np.mean([f['meta'] for f in feats]),
        'backtrack': np.mean([f['backtrack'] for f in feats]),
        'action': np.mean([f['action'] for f in feats]),
        'verify': np.mean([f['verify'] for f in feats]),
        'q_ratio': np.mean([f['q_ratio'] for f in feats]),
        'exclaim': np.mean([f['exclaim'] for f in feats]),
        'pos': np.mean([f['pos'] for f in feats]),
        'neg': np.mean([f['neg'] for f in feats]),
    }

    # Map to stance dimensions (normalized 0-1)
    stance = {
        'EH': min(1.0, agg['hedges'] * 50),  # Scale up small counts
        'DC': min(1.0, agg['modals'] * 50),
        'EX': min(1.0, agg['q_ratio']),
        'MA': min(1.0, agg['meta'] * 100),
        'RR': min(1.0, agg['backtrack'] * 100),
        'AG': min(1.0, agg['action'] * 50),
        'AS': max(0.0, 1.0 - agg['q_ratio']),  # Inverse of exploration
        'SV': min(1.0, agg['verify'] * 50),
        'VA': max(0.0, min(1.0, (agg['pos'] - agg['neg'] + 1.0) / 2.0)),  # Map to [0,1]
        'AR': min(1.0, agg['exclaim'] * 200),
        'IF': min(1.0, agg['action'] * 50),  # Same as AG
        'ED': min(1.0, agg['verify'] * 50),  # Same as SV
    }

    # Return as vector
    axes = ['EH', 'DC', 'EX', 'MA', 'RR', 'AG', 'AS', 'SV', 'VA', 'AR', 'IF', 'ED']
    vector = np.array([stance[ax] for ax in axes])

    return stance, vector, agg


def analyze_epoch(jsonl_path, lex, epoch_name):
    """Analyze a single epoch"""
    turns = load_jsonl(jsonl_path)

    # Filter to model responses only
    model_turns = [t for t in turns if t.get('speaker') == 'model']

    if not model_turns:
        print(f"Warning: No model turns in {epoch_name}")
        return None

    # Extract features
    feats = extract_features(model_turns, lex)

    # Compute stance
    stance, vector, raw_features = compute_stance_vector_from_features(feats)

    return {
        'epoch': epoch_name,
        'n_turns': len(model_turns),
        'stance': stance,
        'vector': vector,
        'raw_features': raw_features
    }


def compare_epochs(results):
    """Compare stance across epochs"""
    comparisons = {}

    epochs = list(results.keys())
    vectors = {ep: results[ep]['vector'] for ep in epochs}

    # Compute pairwise cosine similarities
    for i, ep1 in enumerate(epochs):
        for ep2 in epochs[i+1:]:
            key = f"{ep1}_vs_{ep2}"
            cos_sim = cosine_similarity(vectors[ep1], vectors[ep2])
            comparisons[key] = cos_sim

    return comparisons


def main():
    # Paths
    svk_dir = Path('sage/experiments/phase1-hierarchical-cognitive/svk_analysis/phi15_precision')
    kit_root = Path('forum/nova/stance-vector-kit')

    # Load lexicons
    lex = compile_lexicons(kit_root)

    # Analyze each epoch
    results = {}
    for epoch_file in ['baseline.jsonl', 'epoch_60.jsonl', 'epoch_100.jsonl']:
        epoch_path = svk_dir / epoch_file
        epoch_name = epoch_file.replace('.jsonl', '')

        if not epoch_path.exists():
            print(f"Skipping {epoch_name} (file not found)")
            continue

        print(f"\nAnalyzing {epoch_name}...")
        result = analyze_epoch(epoch_path, lex, epoch_name)
        if result:
            results[epoch_name] = result

    # Print results
    print("\n" + "="*70)
    print("STANCE ANALYSIS RESULTS")
    print("="*70)

    # Stance vectors
    print("\nStance Vectors (12 dimensions):")
    print("-" * 70)

    axes = ['EH', 'DC', 'EX', 'MA', 'RR', 'AG', 'AS', 'SV', 'VA', 'AR', 'IF', 'ED']
    header = f"{'Epoch':<15} " + " ".join(f"{ax:>5}" for ax in axes)
    print(header)
    print("-" * 70)

    for epoch_name, result in results.items():
        stance = result['stance']
        row = f"{epoch_name:<15} " + " ".join(f"{stance[ax]:5.2f}" for ax in axes)
        print(row)

    # Raw feature comparison
    print("\n\nRaw Lexicon Features:")
    print("-" * 70)
    feat_names = ['hedges', 'modals', 'meta', 'backtrack', 'action', 'verify', 'q_ratio']
    header = f"{'Epoch':<15} " + " ".join(f"{fn:>8}" for fn in feat_names)
    print(header)
    print("-" * 70)

    for epoch_name, result in results.items():
        raw = result['raw_features']
        row = f"{epoch_name:<15} " + " ".join(f"{raw[fn]:8.4f}" for fn in feat_names)
        print(row)

    # Cosine similarities
    if len(results) > 1:
        print("\n\nCross-Epoch Cosine Similarity:")
        print("-" * 70)
        comparisons = compare_epochs(results)
        for pair, similarity in comparisons.items():
            print(f"  {pair:<30} {similarity:6.3f}")

    # Save results
    output_dir = svk_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)

    # Save as JSON
    output_data = {}
    for epoch_name, result in results.items():
        output_data[epoch_name] = {
            'stance': result['stance'],
            'raw_features': result['raw_features'],
            'n_turns': result['n_turns']
        }

    with open(output_dir / 'stance_analysis.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n\nResults saved to: {output_dir / 'stance_analysis.json'}")

    # Save as CSV
    rows = []
    for epoch_name, result in results.items():
        row = {'epoch': epoch_name}
        row.update({f's_{ax}': result['stance'][ax] for ax in axes})
        row.update({f'feat_{k}': v for k, v in result['raw_features'].items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'stance_vectors.csv', index=False)
    print(f"CSV saved to: {output_dir / 'stance_vectors.csv'}")


if __name__ == '__main__':
    main()
