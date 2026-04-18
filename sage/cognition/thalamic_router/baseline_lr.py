#!/usr/bin/env python3
"""Logistic Regression baseline for router decisions.

Trains on router shadow data, predicts invoke/noop.
This is the simplest possible learned router — validates the
data pipeline end-to-end and establishes the floor.

Usage:
    python3 -m sage.cognition.thalamic_router.baseline_lr \
        --data private-context/training-data/router/sprout/
"""
import json, gzip, os, sys
import numpy as np
from collections import Counter
from pathlib import Path


def load_records(data_dir: str):
    """Load router records from gzipped JSONL partitions."""
    records = []
    data_path = Path(data_dir)
    for gz_file in sorted(data_path.glob("**/*.jsonl.gz")):
        try:
            with gzip.open(gz_file, 'rt') as f:
                for line in f:
                    try:
                        records.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        pass
        except EOFError:
            pass  # truncated file (still being written)
    return records


def extract_features(record):
    """Extract numeric feature vector from a router record."""
    inp = record['router_input']
    features = [
        inp.get('snarc_surprise', 0),
        inp.get('snarc_novelty', 0),
        inp.get('snarc_arousal', 0),
        inp.get('snarc_reward', 0),
        inp.get('snarc_conflict', 0),
        inp.get('sensory_novelty', 0),
        inp.get('sensory_urgency', 0),
        inp.get('atp_level', 50) / 100.0,
        inp.get('wm_goal_active', False) * 1.0,
        inp.get('wm_pressure', 0),
        inp.get('habit_available', False) * 1.0,
        inp.get('habit_confidence', 0),
        1.0 if 'audio' in inp.get('sensory_modalities', []) else 0.0,
        1.0 if 'message' in inp.get('sensory_modalities', []) else 0.0,
        1.0 if 'vision' in inp.get('sensory_modalities', []) else 0.0,
        {'wake': 1, 'focus': 2, 'rest': -1, 'dream': -2, 'crisis': 3}.get(
            inp.get('metabolic_state', 'rest'), 0) / 3.0,
    ]
    return features


def extract_label(record):
    """Extract binary label: 0=noop, 1=invoke."""
    action = record['router_output']['action']
    return 1 if action == 'invoke' else 0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_lr(X, y, lr=0.1, epochs=100):
    """Train logistic regression with gradient descent. No sklearn needed."""
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0

    for epoch in range(epochs):
        z = X @ weights + bias
        pred = sigmoid(z)
        error = pred - y

        grad_w = (X.T @ error) / len(y)
        grad_b = np.mean(error)

        weights -= lr * grad_w
        bias -= lr * grad_b

        if epoch % 20 == 0:
            loss = -np.mean(y * np.log(pred + 1e-8) + (1-y) * np.log(1-pred + 1e-8))
            acc = np.mean((pred > 0.5) == y)
            print(f"  epoch {epoch:3d}: loss={loss:.4f} acc={acc:.4f}")

    return weights, bias


def evaluate(X, y, weights, bias):
    """Evaluate with agent-zero discipline."""
    pred = sigmoid(X @ weights + bias)
    pred_labels = (pred > 0.5).astype(int)

    # Overall accuracy
    accuracy = np.mean(pred_labels == y)

    # Per-class metrics
    tp = np.sum((pred_labels == 1) & (y == 1))
    fp = np.sum((pred_labels == 1) & (y == 0))
    fn = np.sum((pred_labels == 0) & (y == 1))
    tn = np.sum((pred_labels == 0) & (y == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Agent-zero baseline (always predict majority class)
    majority_class = int(np.mean(y) < 0.5)  # 0 if mostly noop
    dummy_acc = np.mean(y == majority_class)

    return {
        'accuracy': float(accuracy),
        'dummy_accuracy': float(dummy_acc),
        'margin_over_dummy': float(accuracy - dummy_acc),
        'invoke_precision': float(precision),
        'invoke_recall': float(recall),
        'invoke_f1': float(f1),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'n_invoke': int(np.sum(y == 1)),
        'n_noop': int(np.sum(y == 0)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='private-context/training-data/router/sprout/')
    args = parser.parse_args()

    print(f"Loading records from {args.data}...")
    records = load_records(args.data)
    print(f"Loaded {len(records)} records")

    if len(records) < 100:
        print("Not enough records for training. Need ≥100.")
        sys.exit(1)

    # Extract features and labels
    X = np.array([extract_features(r) for r in records])
    y = np.array([extract_label(r) for r in records])

    print(f"Features: {X.shape[1]}, Samples: {len(y)}")
    print(f"Class balance: invoke={np.sum(y==1)}, noop={np.sum(y==0)}")

    # Train/test split (80/20)
    np.random.seed(42)
    idx = np.random.permutation(len(y))
    split = int(0.8 * len(y))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    # Normalize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    print(f"\nTraining LR (train={len(y_train)}, test={len(y_test)})...")
    weights, bias = train_lr(X_train_norm, y_train, lr=0.5, epochs=100)

    # Evaluate
    train_metrics = evaluate(X_train_norm, y_train, weights, bias)
    test_metrics = evaluate(X_test_norm, y_test, weights, bias)

    print(f"\n{'='*50}")
    print(f"RESULTS (agent-zero defended)")
    print(f"{'='*50}")
    print(f"  Train accuracy:     {train_metrics['accuracy']:.4f}")
    print(f"  Test accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"  Dummy (always-noop):{test_metrics['dummy_accuracy']:.4f}")
    print(f"  Margin over dummy:  {test_metrics['margin_over_dummy']:+.4f}")
    print(f"  Invoke precision:   {test_metrics['invoke_precision']:.4f}")
    print(f"  Invoke recall:      {test_metrics['invoke_recall']:.4f}")
    print(f"  Invoke F1:          {test_metrics['invoke_f1']:.4f}")
    print(f"  Confusion: TP={test_metrics['tp']} FP={test_metrics['fp']} FN={test_metrics['fn']} TN={test_metrics['tn']}")

    # Feature importance
    feature_names = [
        'snarc_surprise', 'snarc_novelty', 'snarc_arousal', 'snarc_reward',
        'snarc_conflict', 'sensory_novelty', 'sensory_urgency', 'atp_norm',
        'wm_goal_active', 'wm_pressure', 'habit_available', 'habit_confidence',
        'has_audio', 'has_message', 'has_vision', 'metabolic_level',
    ]
    print(f"\nFeature importance (|weight|):")
    for name, w in sorted(zip(feature_names, weights), key=lambda x: -abs(x[1])):
        print(f"  {name:20s}: {w:+.4f}")


if __name__ == '__main__':
    main()
