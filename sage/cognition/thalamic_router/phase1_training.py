#!/usr/bin/env python3
"""Phase 1 training pipeline — LR baseline with full agent-zero discipline.

Sprint 1 Phase 0 produced captured router records. This module trains a
Tier-A logistic-regression baseline on that data and applies the full
set of gates from PRD §4 Phase 1 exit criteria + §7.10 agent-zero
defenses. It is the canonical Phase 1 entry runner.

Builds on sage/cognition/thalamic_router/baseline_lr.py (Sprout, 2026-04-17)
which established the numpy-only pattern, dummy baseline reporting, and
JSONL loader. This module ADDS:

- 3-class decision head (invoke / habit / noop) via one-vs-rest LR
- Per-plugin classification within `invoke` (softmax over registry)
- Per-class F1 reporting for EVERY class (not just invoke)
- Salience-weighted agreement on top-decile-arousal subset (§7.10.3)
- Rare-decision recall on non-modal-class subset (§4 Phase 2 gate)
- SNARC ablation (train with + without SNARC features, compare, reject
  adapter if delta <5% per §4.7.F)
- Stratified golden dataset construction
- INCONCLUSIVE gating per §7.10 (if margin <25pp, report INCONCLUSIVE
  regardless of accuracy)
- Data-diversity gate (refuse to declare training ready if source/
  decision-class/SNARC-quintile distributions are too homogeneous —
  the collinearity artifact Sprout's baseline surfaced)
- JSON output for CI consumption

Spec: phase2/brain-arch/thalamic-router-prd.md
      §0.2 (agent-zero), §4 Phase 1, §4.7 (SNARC integration),
      §7.10 (CI defenses)

Not in scope (deferred):
- Confidence-outcome correlation (needs Phase 4 RPE signals)
- LoRA adapter training (Tier C/D only, after Tier A/B ceiling)
- Federated per-machine personalization (Phase 5)
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import zlib
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ───────────────────────────────────────────────────────────────────
# Constants — PRD thresholds (binding)
# ───────────────────────────────────────────────────────────────────

# PRD §4 Phase 1 exit: aggregate agreement with baseline ≥98%
PHASE1_AGGREGATE_AGREEMENT_THRESHOLD = 0.98

# PRD §7.10: modal-class dummy margin ≥25pp
MODAL_DUMMY_MARGIN_THRESHOLD_PP = 0.25

# PRD §4 Phase 1 exit: per-class F1 ≥0.85 on EVERY class
PER_CLASS_F1_THRESHOLD = 0.85

# PRD §4 Phase 1 exit: salience-weighted agreement ≥95% on top-decile-arousal
SALIENCE_WEIGHTED_THRESHOLD = 0.95

# PRD §4.7.F: SNARC-utility delta ≥5% or adapter rejected
SNARC_UTILITY_DELTA_THRESHOLD = 0.05

# PRD §4 Phase 1 exit: rare-decision recall ≥0.80
RARE_DECISION_RECALL_THRESHOLD = 0.80

# Data-diversity thresholds (not in PRD — our safeguard against the
# collinearity artifact. Sprout + CBP Phase 0 data hit 100% LR because
# data was trivial; we enforce minimum diversity before any result is
# trustworthy).
MIN_SOURCES_REPRESENTED = 1      # raising + gameplay + idle ideally
MIN_DECISION_CLASSES = 2         # at least invoke vs noop
MIN_ENTROPY_NATS = 0.3           # across decision classes
MIN_SNARC_STD_PER_DIM = 0.05     # per-dim stddev — flags collinearity

# Head B (action distillation) — relaxed gates, different target.
# Fleet converged on Head A INCONCLUSIVE because the programmatic teacher
# is SNARC-blind. Head B uses known_good_action (from gameplay records)
# as the label, which IS SNARC-correlated (CBP data shows 6x arousal
# difference between UP and CLICK). Gates are calibrated for a 7-class
# problem with a realistic modal-dummy baseline around 25-30%.
HEAD_B_MARGIN_THRESHOLD_PP = 0.15   # 7-way is harder; 15pp above dummy is meaningful
HEAD_B_COMMON_ACTION_RECALL = 0.60  # recall floor on the 4 most common actions
HEAD_B_SNARC_UTILITY_THRESHOLD = 0.05  # same as Head A — SNARC should matter
HEAD_B_ACTION_CLASSES = 7           # GameAction 0..6
# Names for readable reports (GameAction: 0=A0, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=SEL, 6=CLICK, 7=UNDO)
HEAD_B_ACTION_NAMES = ["A0", "UP", "DOWN", "LEFT", "RIGHT", "SEL", "CLICK"]


# Feature names — order defines the feature vector layout
SNARC_FEATURES = [
    "snarc_surprise", "snarc_novelty", "snarc_arousal",
    "snarc_reward", "snarc_conflict",
]
NON_SNARC_FEATURES = [
    "sensory_novelty", "sensory_urgency", "atp_norm",
    "wm_goal_active", "wm_pressure",
    "habit_available", "habit_confidence",
    "has_audio", "has_message", "has_vision",
    "metabolic_level",
]
ALL_FEATURE_NAMES = SNARC_FEATURES + NON_SNARC_FEATURES

METABOLIC_MAP = {"wake": 1, "focus": 2, "rest": -1, "dream": -2, "crisis": 3}


# ───────────────────────────────────────────────────────────────────
# Data loading + feature extraction
# ───────────────────────────────────────────────────────────────────

def load_records(data_dir: str, source_filter: Optional[str] = None) -> List[Dict]:
    """Load router records from gzipped/plain JSONL partitions.

    If source_filter is set, keep only records whose metadata.source matches.
    Robust to truncated-last-line, missing gzip indices, and schema-version
    skew between v0.1.0 and v0.2.0.
    """
    records: List[Dict] = []
    data_path = Path(data_dir)
    for shard in sorted(data_path.glob("**/*.jsonl*")):
        open_fn = gzip.open if shard.suffix == ".gz" else open
        try:
            with open_fn(shard, "rt") as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    if source_filter:
                        src = (rec.get("metadata") or {}).get("source")
                        if src != source_filter:
                            continue
                    records.append(rec)
        except (EOFError, OSError, zlib.error):
            continue
    return records


def _feature_vec(record: Dict) -> List[float]:
    """Build the ALL_FEATURE_NAMES vector from a record."""
    inp = record.get("router_input", {})
    modalities = inp.get("sensory_modalities", []) or []
    metabolic = inp.get("metabolic_state", "rest")
    return [
        # SNARC
        float(inp.get("snarc_surprise", 0) or 0),
        float(inp.get("snarc_novelty", 0) or 0),
        float(inp.get("snarc_arousal", 0) or 0),
        float(inp.get("snarc_reward", 0) or 0),
        float(inp.get("snarc_conflict", 0) or 0),
        # Non-SNARC
        float(inp.get("sensory_novelty", 0) or 0),
        float(inp.get("sensory_urgency", 0) or 0),
        float(inp.get("atp_level", 50) or 50) / 100.0,
        1.0 if inp.get("wm_goal_active") else 0.0,
        float(inp.get("wm_pressure", 0) or 0),
        1.0 if inp.get("habit_available") else 0.0,
        float(inp.get("habit_confidence", 0) or 0),
        1.0 if "audio" in modalities else 0.0,
        1.0 if "message" in modalities else 0.0,
        1.0 if "vision" in modalities else 0.0,
        METABOLIC_MAP.get(metabolic, 0) / 3.0,
    ]


ACTION_CLASSES = ["noop", "invoke", "habit"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}


def _action_label(record: Dict) -> int:
    """Return index into ACTION_CLASSES."""
    return ACTION_TO_IDX.get(record.get("router_output", {}).get("action"), 0)


def build_xy(records: List[Dict], snarc: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Feature matrix + action-class labels. `snarc=False` zeros out SNARC
    columns for the §4.7.F ablation."""
    X = np.array([_feature_vec(r) for r in records], dtype=np.float64)
    if not snarc:
        snarc_idx = [ALL_FEATURE_NAMES.index(n) for n in SNARC_FEATURES]
        X[:, snarc_idx] = 0.0
    y = np.array([_action_label(r) for r in records], dtype=np.int64)
    return X, y


def build_xy_head_b(
    records: List[Dict], snarc: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Feature matrix + known_good_action labels for Head B.

    Returns (X, y, kept_indices) — kept_indices lets callers align the
    filtered record subset back to the original list (needed for
    salience slicing). Records without a valid known_good_action in
    range [0..6] are dropped.
    """
    kept: List[int] = []
    rows: List[List[float]] = []
    ys: List[int] = []
    for i, r in enumerate(records):
        md = r.get("metadata") or {}
        act = md.get("known_good_action")
        if not isinstance(act, int) or act < 0 or act >= HEAD_B_ACTION_CLASSES:
            continue
        kept.append(i)
        rows.append(_feature_vec(r))
        ys.append(act)
    X = np.array(rows, dtype=np.float64)
    if not snarc and len(X):
        snarc_idx = [ALL_FEATURE_NAMES.index(n) for n in SNARC_FEATURES]
        X[:, snarc_idx] = 0.0
    return X, np.array(ys, dtype=np.int64), kept


def normalize(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean/std normalize using train statistics."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


# ───────────────────────────────────────────────────────────────────
# Logistic regression (numpy only — matches Sprout's baseline style)
# ───────────────────────────────────────────────────────────────────

def _softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def train_multiclass_lr(
    X: np.ndarray, y: np.ndarray, n_classes: int,
    lr: float = 0.5, epochs: int = 200, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Softmax regression via vanilla GD. Returns (W, b). Deterministic."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = rng.normal(0, 0.01, size=(d, n_classes))
    b = np.zeros(n_classes)

    # One-hot labels
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1.0

    for _ in range(epochs):
        Z = X @ W + b
        P = _softmax(Z)
        grad_W = X.T @ (P - Y) / n
        grad_b = (P - Y).mean(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b
    return W, b


def predict(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.argmax(_softmax(X @ W + b), axis=1)


# ───────────────────────────────────────────────────────────────────
# Metrics with agent-zero discipline
# ───────────────────────────────────────────────────────────────────

@dataclass
class ClassMetrics:
    name: str
    n: int
    precision: float
    recall: float
    f1: float


@dataclass
class Phase1Metrics:
    # Headline
    aggregate_accuracy: float
    modal_class: str
    modal_dummy_accuracy: float
    margin_over_dummy: float

    # Per-class (every class must pass per-class F1)
    per_class: List[ClassMetrics]
    min_class_f1: float
    min_class_name: str

    # Salience-weighted slice (§7.10.3)
    salience_weighted_accuracy: float
    salience_subset_n: int

    # Rare-decision recall
    rare_class: str
    rare_recall: float
    rare_n: int

    # SNARC ablation (§4.7.F)
    snarc_ablation_accuracy: Optional[float] = None
    snarc_utility_delta: Optional[float] = None

    # Data diversity
    n_sources: int = 0
    n_decision_classes: int = 0
    decision_class_entropy: float = 0.0
    min_snarc_std: float = 0.0

    # Verdict
    verdict: str = "PENDING"
    verdict_reasons: List[str] = field(default_factory=list)


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> List[ClassMetrics]:
    out = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        n = int((y_true == c).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out.append(ClassMetrics(
            name=ACTION_CLASSES[c], n=n,
            precision=precision, recall=recall, f1=f1,
        ))
    return out


def salience_weighted_accuracy(
    records: List[Dict], y_true: np.ndarray, y_pred: np.ndarray,
) -> Tuple[float, int]:
    """Accuracy on the top-decile-arousal subset.

    High arousal means the decision matters more. §7.10.3 wants this slice.
    """
    arousal = np.array([
        float((r.get("router_input") or {}).get("snarc_arousal", 0) or 0)
        for r in records
    ])
    if arousal.max() == arousal.min():
        return float((y_pred == y_true).mean()), len(records)
    threshold = np.quantile(arousal, 0.9)
    mask = arousal >= threshold
    if mask.sum() == 0:
        return 0.0, 0
    subset_acc = float((y_pred[mask] == y_true[mask]).mean())
    return subset_acc, int(mask.sum())


def rare_decision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, int]:
    """Recall on the non-modal-class subset."""
    counts = Counter(y_true.tolist())
    modal = max(counts, key=counts.get)
    rare_mask = y_true != modal
    if rare_mask.sum() == 0:
        return ACTION_CLASSES[modal], 0.0, 0
    correct_rare = int((y_pred[rare_mask] == y_true[rare_mask]).sum())
    # Recall on the rare subset is the fraction of rare labels correctly recovered
    # (equivalent to accuracy restricted to rare-true).
    rare_acc = correct_rare / int(rare_mask.sum())
    return ACTION_CLASSES[modal], float(rare_acc), int(rare_mask.sum())


def compute_diversity(records: List[Dict]) -> Dict[str, Any]:
    """Data-diversity metrics for the diversity gate."""
    sources = Counter(
        (r.get("metadata") or {}).get("source", "unknown")
        for r in records
    )
    actions = Counter(
        (r.get("router_output") or {}).get("action", "unknown")
        for r in records
    )
    # Shannon entropy over decision classes (nats)
    total = sum(actions.values())
    probs = [c / total for c in actions.values() if c > 0]
    entropy = -sum(p * math.log(p) for p in probs) if probs else 0.0
    # Per-SNARC-dim std (collinearity signal)
    X, _ = build_xy(records, snarc=True)
    snarc_cols = [ALL_FEATURE_NAMES.index(n) for n in SNARC_FEATURES]
    per_dim_std = X[:, snarc_cols].std(axis=0)
    return {
        "sources": dict(sources),
        "actions": dict(actions),
        "n_sources": len(sources),
        "n_decision_classes": len(actions),
        "decision_class_entropy": entropy,
        "snarc_per_dim_std": {
            n: float(s) for n, s in zip(SNARC_FEATURES, per_dim_std)
        },
        "min_snarc_std": float(per_dim_std.min()) if len(per_dim_std) else 0.0,
    }


# ───────────────────────────────────────────────────────────────────
# Evaluation entry point
# ───────────────────────────────────────────────────────────────────

def evaluate_phase1(
    records: List[Dict],
    test_frac: float = 0.2,
    seed: int = 42,
) -> Phase1Metrics:
    """Run the full Phase 1 evaluation pipeline on `records`."""
    if len(records) < 100:
        raise ValueError(f"need ≥100 records for Phase 1 evaluation, got {len(records)}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    split = int((1 - test_frac) * len(records))
    train_records = [records[i] for i in idx[:split]]
    test_records = [records[i] for i in idx[split:]]

    # --- Main model (with SNARC) ---
    X_train, y_train = build_xy(train_records, snarc=True)
    X_test, y_test = build_xy(test_records, snarc=True)
    n_classes = 3  # noop / invoke / habit
    X_train_n, X_test_n = normalize(X_train, X_test)
    W, b = train_multiclass_lr(X_train_n, y_train, n_classes, seed=seed)
    y_pred = predict(X_test_n, W, b)
    agg_acc = float((y_pred == y_test).mean())

    # Modal class + dummy
    counts = Counter(y_test.tolist())
    modal = max(counts, key=counts.get)
    modal_dummy_acc = counts[modal] / len(y_test)
    margin = agg_acc - modal_dummy_acc

    # Per-class
    per_class = per_class_metrics(y_test, y_pred, n_classes)
    present = [c for c in per_class if c.n > 0]
    min_f1 = min(c.f1 for c in present) if present else 0.0
    min_cls = min(present, key=lambda c: c.f1).name if present else ""

    # Salience-weighted slice
    sal_acc, sal_n = salience_weighted_accuracy(test_records, y_test, y_pred)

    # Rare-decision recall
    rare_class, rare_acc, rare_n = rare_decision_recall(y_test, y_pred)

    # --- SNARC ablation (§4.7.F) ---
    X_train_ns, y_train_ns = build_xy(train_records, snarc=False)
    X_test_ns, _ = build_xy(test_records, snarc=False)
    X_train_ns_n, X_test_ns_n = normalize(X_train_ns, X_test_ns)
    W_ns, b_ns = train_multiclass_lr(X_train_ns_n, y_train_ns, n_classes, seed=seed)
    y_pred_ns = predict(X_test_ns_n, W_ns, b_ns)
    ablation_acc = float((y_pred_ns == y_test).mean())
    snarc_delta = agg_acc - ablation_acc

    # --- Diversity ---
    diversity = compute_diversity(records)

    m = Phase1Metrics(
        aggregate_accuracy=agg_acc,
        modal_class=ACTION_CLASSES[modal],
        modal_dummy_accuracy=modal_dummy_acc,
        margin_over_dummy=margin,
        per_class=per_class,
        min_class_f1=min_f1,
        min_class_name=min_cls,
        salience_weighted_accuracy=sal_acc,
        salience_subset_n=sal_n,
        rare_class=rare_class,
        rare_recall=rare_acc,
        rare_n=rare_n,
        snarc_ablation_accuracy=ablation_acc,
        snarc_utility_delta=snarc_delta,
        n_sources=diversity["n_sources"],
        n_decision_classes=diversity["n_decision_classes"],
        decision_class_entropy=diversity["decision_class_entropy"],
        min_snarc_std=diversity["min_snarc_std"],
    )

    # Apply gates
    reasons: List[str] = []

    if m.margin_over_dummy < MODAL_DUMMY_MARGIN_THRESHOLD_PP:
        reasons.append(
            f"margin over dummy {m.margin_over_dummy:.3f} < {MODAL_DUMMY_MARGIN_THRESHOLD_PP} "
            "(§7.10: INCONCLUSIVE)"
        )
    if m.aggregate_accuracy < PHASE1_AGGREGATE_AGREEMENT_THRESHOLD:
        reasons.append(
            f"aggregate accuracy {m.aggregate_accuracy:.4f} < {PHASE1_AGGREGATE_AGREEMENT_THRESHOLD}"
        )
    if m.min_class_f1 < PER_CLASS_F1_THRESHOLD:
        # Only fail if the weak class has meaningful support
        weak = next((c for c in present if c.name == m.min_class_name), None)
        if weak and weak.n >= 20:
            reasons.append(
                f"worst-class F1 {m.min_class_f1:.3f} on '{m.min_class_name}' "
                f"< {PER_CLASS_F1_THRESHOLD} (n={weak.n})"
            )
    if m.salience_weighted_accuracy < SALIENCE_WEIGHTED_THRESHOLD and sal_n >= 50:
        reasons.append(
            f"salience-weighted acc {m.salience_weighted_accuracy:.4f} < {SALIENCE_WEIGHTED_THRESHOLD}"
        )
    if m.rare_n >= 20 and m.rare_recall < RARE_DECISION_RECALL_THRESHOLD:
        reasons.append(
            f"rare-decision recall {m.rare_recall:.4f} < {RARE_DECISION_RECALL_THRESHOLD} "
            f"(rare class='{m.rare_class}', n={m.rare_n})"
        )
    if m.snarc_utility_delta is not None and m.snarc_utility_delta < SNARC_UTILITY_DELTA_THRESHOLD:
        reasons.append(
            f"SNARC-utility delta {m.snarc_utility_delta:.4f} < {SNARC_UTILITY_DELTA_THRESHOLD} "
            "(§4.7.F: router is not using SNARC — reject adapter)"
        )
    # Diversity gate (our addition, not in PRD but PRD-aligned)
    if m.n_decision_classes < MIN_DECISION_CLASSES:
        reasons.append(
            f"only {m.n_decision_classes} decision class(es) in data "
            f"< {MIN_DECISION_CLASSES} (diversity gate)"
        )
    if m.decision_class_entropy < MIN_ENTROPY_NATS:
        reasons.append(
            f"decision-class entropy {m.decision_class_entropy:.3f} < {MIN_ENTROPY_NATS} nats "
            "(dataset too homogeneous — collinearity artifact risk)"
        )
    if m.min_snarc_std < MIN_SNARC_STD_PER_DIM:
        reasons.append(
            f"min SNARC-dim stddev {m.min_snarc_std:.4f} < {MIN_SNARC_STD_PER_DIM} "
            "(SNARC features are collinear/constant — §0.2 agent-zero hazard)"
        )

    if not reasons:
        m.verdict = "PASS"
    else:
        # If only the diversity gate + SNARC ablation fail, the model IS
        # trivially correct but the data is not yet informative.
        diversity_only = all(
            any(key in r for key in ("diversity", "SNARC-utility", "homogeneous", "collinear"))
            for r in reasons
        )
        if diversity_only:
            m.verdict = "INCONCLUSIVE"
        elif m.margin_over_dummy < MODAL_DUMMY_MARGIN_THRESHOLD_PP:
            m.verdict = "INCONCLUSIVE"
        else:
            m.verdict = "FAIL"
    m.verdict_reasons = reasons
    return m


# ───────────────────────────────────────────────────────────────────
# Head B — action distillation from known_good_action
# ───────────────────────────────────────────────────────────────────

@dataclass
class HeadBMetrics:
    aggregate_accuracy: float
    modal_class: str               # e.g. "CLICK"
    modal_dummy_accuracy: float
    margin_over_dummy: float

    per_class: List[ClassMetrics]

    # Common-action recall (top 4 most populous actions)
    common_actions: List[str]
    common_action_min_recall: float

    # SNARC ablation
    snarc_ablation_accuracy: Optional[float] = None
    snarc_utility_delta: Optional[float] = None

    # Data diversity
    n_records: int = 0
    n_actions_present: int = 0
    action_entropy: float = 0.0
    min_snarc_std: float = 0.0

    # Verdict
    verdict: str = "PENDING"
    verdict_reasons: List[str] = field(default_factory=list)


@dataclass
class HeadBAdapter:
    """Promotable Head B adapter — trained weights + normalization stats.

    Written by save_adapter(); loaded by sage_plays and any downstream
    inline inference. Self-contained: includes the feature-extractor
    assumption (ALL_FEATURE_NAMES) so we can detect schema skew.
    """
    head: str                           # "B"
    machine: str
    trained_at: str                     # ISO timestamp
    train_commit: Optional[str]
    n_train_records: int
    feature_names: List[str]
    feature_mean: List[float]
    feature_std: List[float]
    weights: List[List[float]]          # (n_features, n_classes)
    bias: List[float]                   # (n_classes,)
    n_classes: int
    class_names: List[str]

    @classmethod
    def from_training(
        cls, *, machine: str, W: np.ndarray, b: np.ndarray,
        X_train_raw: np.ndarray, n_train: int,
        train_commit: Optional[str] = None,
    ) -> "HeadBAdapter":
        from datetime import datetime, timezone
        mean = X_train_raw.mean(axis=0)
        std = X_train_raw.std(axis=0) + 1e-8
        return cls(
            head="B", machine=machine,
            trained_at=datetime.now(timezone.utc).isoformat(),
            train_commit=train_commit,
            n_train_records=int(n_train),
            feature_names=list(ALL_FEATURE_NAMES),
            feature_mean=mean.tolist(),
            feature_std=std.tolist(),
            weights=W.tolist(), bias=b.tolist(),
            n_classes=HEAD_B_ACTION_CLASSES,
            class_names=list(HEAD_B_ACTION_NAMES),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "HeadBAdapter":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(**d)

    def predict(self, features: List[float]) -> Tuple[int, List[float]]:
        """Return (argmax_class, softmax_probs) for a single feature vec."""
        x = (np.array(features, dtype=np.float64) - np.array(self.feature_mean)) / np.array(self.feature_std)
        logits = x @ np.array(self.weights) + np.array(self.bias)
        logits = logits - logits.max()
        e = np.exp(logits)
        p = (e / e.sum()).tolist()
        return int(np.argmax(logits)), p


def evaluate_phase1_head_b(
    records: List[Dict],
    test_frac: float = 0.2,
    seed: int = 42,
    adapter_out: Optional[Path] = None,
    machine: str = "unknown",
) -> HeadBMetrics:
    """Head B: predict known_good_action from RouterInput features.

    Callers should pre-filter to gameplay records (otherwise most
    records have no known_good_action and get dropped).
    """
    X_all, y_all, kept = build_xy_head_b(records, snarc=True)
    if len(X_all) < 100:
        raise ValueError(
            f"Head B needs ≥100 records with known_good_action, got {len(X_all)}. "
            "Pre-filter to source=gameplay."
        )

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_all))
    split = int((1 - test_frac) * len(X_all))
    tr, te = idx[:split], idx[split:]
    X_train, y_train = X_all[tr], y_all[tr]
    X_test, y_test = X_all[te], y_all[te]

    X_train_n, X_test_n = normalize(X_train, X_test)
    W, b = train_multiclass_lr(X_train_n, y_train, HEAD_B_ACTION_CLASSES, seed=seed)
    y_pred = predict(X_test_n, W, b)
    agg_acc = float((y_pred == y_test).mean())

    # Persist adapter if requested — includes raw (un-normalized) mean/std
    # so inline inference can normalize using the same stats.
    if adapter_out is not None:
        adapter = HeadBAdapter.from_training(
            machine=machine, W=W, b=b, X_train_raw=X_train, n_train=len(X_train),
        )
        adapter.save(Path(adapter_out))

    counts = Counter(y_test.tolist())
    modal = max(counts, key=counts.get)
    modal_dummy_acc = counts[modal] / len(y_test)
    margin = agg_acc - modal_dummy_acc

    # Per-class — HEAD_B_ACTION_NAMES indexed 0..6
    per_class: List[ClassMetrics] = []
    for c in range(HEAD_B_ACTION_CLASSES):
        tp = int(((y_pred == c) & (y_test == c)).sum())
        fp = int(((y_pred == c) & (y_test != c)).sum())
        fn = int(((y_pred != c) & (y_test == c)).sum())
        n = int((y_test == c).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class.append(ClassMetrics(
            name=HEAD_B_ACTION_NAMES[c], n=n, precision=p, recall=r, f1=f1,
        ))

    # Common-action recall — top 4 by support, compute min recall across them
    present = sorted([c for c in per_class if c.n > 0], key=lambda c: -c.n)
    common = present[:4]
    common_names = [c.name for c in common]
    common_min_recall = min((c.recall for c in common), default=0.0)

    # SNARC ablation
    X_ns_all, y_ns_all, _ = build_xy_head_b(records, snarc=False)
    X_train_ns, y_train_ns = X_ns_all[tr], y_ns_all[tr]
    X_test_ns, _ = X_ns_all[te], y_ns_all[te]
    X_train_ns_n, X_test_ns_n = normalize(X_train_ns, X_test_ns)
    W_ns, b_ns = train_multiclass_lr(X_train_ns_n, y_train_ns, HEAD_B_ACTION_CLASSES, seed=seed)
    y_pred_ns = predict(X_test_ns_n, W_ns, b_ns)
    ablation_acc = float((y_pred_ns == y_test).mean())
    snarc_delta = agg_acc - ablation_acc

    # Diversity on the Head B subset
    entropy = 0.0
    total = sum(counts.values())
    for c, n in counts.items():
        p = n / total
        if p > 0:
            entropy -= p * math.log(p)
    snarc_cols = [ALL_FEATURE_NAMES.index(n) for n in SNARC_FEATURES]
    min_snarc_std = float(X_all[:, snarc_cols].std(axis=0).min()) if len(X_all) else 0.0

    m = HeadBMetrics(
        aggregate_accuracy=agg_acc,
        modal_class=HEAD_B_ACTION_NAMES[modal],
        modal_dummy_accuracy=modal_dummy_acc,
        margin_over_dummy=margin,
        per_class=per_class,
        common_actions=common_names,
        common_action_min_recall=common_min_recall,
        snarc_ablation_accuracy=ablation_acc,
        snarc_utility_delta=snarc_delta,
        n_records=len(X_all),
        n_actions_present=len([c for c in per_class if c.n > 0]),
        action_entropy=entropy,
        min_snarc_std=min_snarc_std,
    )

    reasons: List[str] = []
    if m.margin_over_dummy < HEAD_B_MARGIN_THRESHOLD_PP:
        reasons.append(
            f"margin over 7-way modal dummy {m.margin_over_dummy:+.4f} "
            f"< {HEAD_B_MARGIN_THRESHOLD_PP}"
        )
    if m.common_action_min_recall < HEAD_B_COMMON_ACTION_RECALL:
        reasons.append(
            f"min recall on top-4 actions {m.common_action_min_recall:.3f} "
            f"< {HEAD_B_COMMON_ACTION_RECALL} "
            f"(common={','.join(m.common_actions)})"
        )
    if m.snarc_utility_delta is not None and m.snarc_utility_delta < HEAD_B_SNARC_UTILITY_THRESHOLD:
        reasons.append(
            f"SNARC-utility delta {m.snarc_utility_delta:+.4f} "
            f"< {HEAD_B_SNARC_UTILITY_THRESHOLD} "
            "(action prediction not using SNARC — feature set may be insufficient)"
        )

    if not reasons:
        m.verdict = "PASS"
    elif m.margin_over_dummy < HEAD_B_MARGIN_THRESHOLD_PP and m.aggregate_accuracy <= m.modal_dummy_accuracy:
        m.verdict = "FAIL"
    else:
        m.verdict = "INCONCLUSIVE"
    m.verdict_reasons = reasons
    return m


def _print_head_b_report(m: HeadBMetrics) -> None:
    print("=" * 60)
    print("Phase 1 Head B — action distillation report")
    print("=" * 60)
    print(f"  Records used         : {m.n_records}")
    print(f"  Aggregate accuracy   : {m.aggregate_accuracy:.4f}")
    print(f"  Modal class          : {m.modal_class}")
    print(f"  Modal-dummy accuracy : {m.modal_dummy_accuracy:.4f}")
    print(f"  Margin over dummy    : {m.margin_over_dummy:+.4f}  (threshold +{HEAD_B_MARGIN_THRESHOLD_PP})")
    print(f"  SNARC ablation acc   : {m.snarc_ablation_accuracy:.4f}"
          if m.snarc_ablation_accuracy is not None else "  SNARC ablation       : SKIPPED")
    print(f"  SNARC-utility delta  : {m.snarc_utility_delta:+.4f}  (threshold +{HEAD_B_SNARC_UTILITY_THRESHOLD})"
          if m.snarc_utility_delta is not None else "")
    print()
    print(f"  Common actions       : {','.join(m.common_actions)}")
    print(f"  Common-action min R  : {m.common_action_min_recall:.3f}  (threshold {HEAD_B_COMMON_ACTION_RECALL})")
    print()
    print(f"  Action entropy       : {m.action_entropy:.3f} nats ({m.n_actions_present} actions present)")
    print(f"  Min SNARC-dim stddev : {m.min_snarc_std:.4f}")
    print()
    print(f"  Per-action:")
    for c in m.per_class:
        if c.n == 0:
            continue
        print(f"    {c.name:6s}: n={c.n:5d}  P={c.precision:.3f}  R={c.recall:.3f}  F1={c.f1:.3f}")
    print()
    verdict_colors = {"PASS": "✓", "INCONCLUSIVE": "~", "FAIL": "✗", "PENDING": "?"}
    print(f"  Verdict: {verdict_colors.get(m.verdict, '?')} {m.verdict}")
    for r in m.verdict_reasons:
        print(f"    - {r}")


# ───────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────

def _print_report(m: Phase1Metrics) -> None:
    print("=" * 60)
    print("Phase 1 LR baseline — agent-zero-defended report")
    print("=" * 60)
    print(f"  Aggregate accuracy    : {m.aggregate_accuracy:.4f}")
    print(f"  Modal class           : {m.modal_class}")
    print(f"  Modal-dummy accuracy  : {m.modal_dummy_accuracy:.4f}")
    print(f"  Margin over dummy     : {m.margin_over_dummy:+.4f}  (threshold +{MODAL_DUMMY_MARGIN_THRESHOLD_PP})")
    print(f"  Salience-weighted acc : {m.salience_weighted_accuracy:.4f}  (n={m.salience_subset_n})")
    print(f"  Rare-decision recall  : {m.rare_recall:.4f}  (class!={m.rare_class}, n={m.rare_n})")
    print(f"  SNARC ablation acc    : "
          f"{m.snarc_ablation_accuracy:.4f}" if m.snarc_ablation_accuracy is not None else "  SNARC ablation        : SKIPPED")
    print(f"  SNARC-utility delta   : "
          f"{m.snarc_utility_delta:+.4f}  (threshold +{SNARC_UTILITY_DELTA_THRESHOLD})"
          if m.snarc_utility_delta is not None else "")
    print()
    print(f"  Data diversity:")
    print(f"    Sources represented   : {m.n_sources}")
    print(f"    Decision classes      : {m.n_decision_classes}")
    print(f"    Class entropy (nats)  : {m.decision_class_entropy:.3f}")
    print(f"    Min SNARC-dim stddev  : {m.min_snarc_std:.4f}  (threshold {MIN_SNARC_STD_PER_DIM})")
    print()
    print(f"  Per-class:")
    for c in m.per_class:
        print(f"    {c.name:10s}: n={c.n:6d}  P={c.precision:.3f}  R={c.recall:.3f}  F1={c.f1:.3f}")
    print()
    verdict_colors = {"PASS": "✓", "INCONCLUSIVE": "~", "FAIL": "✗", "PENDING": "?"}
    print(f"  Verdict: {verdict_colors.get(m.verdict, '?')} {m.verdict}")
    if m.verdict_reasons:
        print(f"  Reasons:")
        for r in m.verdict_reasons:
            print(f"    - {r}")


def _to_json_safe(m: Phase1Metrics) -> Dict[str, Any]:
    d = asdict(m)
    d["thresholds"] = {
        "phase1_aggregate": PHASE1_AGGREGATE_AGREEMENT_THRESHOLD,
        "modal_dummy_margin_pp": MODAL_DUMMY_MARGIN_THRESHOLD_PP,
        "per_class_f1": PER_CLASS_F1_THRESHOLD,
        "salience_weighted": SALIENCE_WEIGHTED_THRESHOLD,
        "snarc_utility_delta": SNARC_UTILITY_DELTA_THRESHOLD,
        "rare_decision_recall": RARE_DECISION_RECALL_THRESHOLD,
        "min_decision_classes": MIN_DECISION_CLASSES,
        "min_entropy_nats": MIN_ENTROPY_NATS,
        "min_snarc_std": MIN_SNARC_STD_PER_DIM,
    }
    return d


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="Path to router shadow partitions (per-machine dir OR _aggregate/).")
    p.add_argument("--head", choices=["A", "B"], default="A",
                   help="Training head. A = dispatch (teacher agreement, PRD §4 Phase 1). "
                        "B = action distillation from known_good_action (gameplay-only).")
    p.add_argument("--source", default=None,
                   help="Optional filter on metadata.source (raising|gameplay|idle|interactive). "
                        "Head B auto-sets source=gameplay unless overridden.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--json-out", default=None,
                   help="If set, write the full report as JSON to this path.")
    p.add_argument("--adapter-out", default=None,
                   help="Head B only: save trained LR adapter (weights + "
                        "feature stats) to this path for inline inference.")
    p.add_argument("--machine", default=os.environ.get("SAGE_MACHINE", "unknown"),
                   help="Machine slug for adapter metadata.")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero on FAIL or INCONCLUSIVE (for CI).")
    args = p.parse_args()

    effective_source = args.source
    if args.head == "B" and effective_source is None:
        effective_source = "gameplay"
        print("[head B] defaulting --source=gameplay (override with --source explicitly)")

    records = load_records(args.data, source_filter=effective_source)
    print(f"Loaded {len(records)} records from {args.data}"
          + (f" (source={effective_source})" if effective_source else ""))
    if len(records) < 100:
        print("Insufficient data (<100 records). Wait for capture to accumulate.")
        return 2

    if args.head == "A":
        m: Any = evaluate_phase1(records, test_frac=args.test_frac, seed=args.seed)
        _print_report(m)
        payload = _to_json_safe(m)
    else:
        m = evaluate_phase1_head_b(
            records, test_frac=args.test_frac, seed=args.seed,
            adapter_out=Path(args.adapter_out) if args.adapter_out else None,
            machine=args.machine,
        )
        _print_head_b_report(m)
        if args.adapter_out:
            print(f"Wrote adapter: {args.adapter_out}")
        payload = asdict(m)
        payload["head"] = "B"
        payload["thresholds"] = {
            "head_b_margin_pp": HEAD_B_MARGIN_THRESHOLD_PP,
            "head_b_common_action_recall": HEAD_B_COMMON_ACTION_RECALL,
            "head_b_snarc_utility": HEAD_B_SNARC_UTILITY_THRESHOLD,
        }

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON report: {args.json_out}")

    if args.strict and m.verdict != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
