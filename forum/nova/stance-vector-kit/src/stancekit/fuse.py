
from typing import Dict
import numpy as np
from .config import EMA_ALPHA

def fuse_axes(l1_signals: Dict[str, float], l2_logits: Dict[str, float], weights=(0.5, 0.5)) -> Dict[str, float]:
    axes = sorted(set(list(l1_signals.keys()) + list(l2_logits.keys())))
    out = {}
    for ax in axes:
        a = l1_signals.get(ax, 0.0)
        b = l2_logits.get(ax, 0.0)
        out[ax] = float(weights[0]*a + weights[1]*b)
    return out

def ema_vector(prev: np.ndarray, curr: np.ndarray, alpha: float = EMA_ALPHA):
    if prev is None:
        return curr
    return alpha*curr + (1.0-alpha)*prev
