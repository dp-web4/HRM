
from typing import List, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression

AXES = ["EH","DC","EX","MA","RR","AG","AS","SV","VA","AR","IF","ED"]

class StanceHead:
    """
    Tiny multi-label classifier head on top of hand-crafted features.
    Replace with an embedding-based model if desired.
    """
    def __init__(self, axes: List[str] = None):
        self.axes = axes or AXES
        self.models = {ax: LogisticRegression(max_iter=500) for ax in self.axes}
        self.is_fit = False

    def _to_X(self, feats: List[Dict]) -> np.ndarray:
        keys = ["hedges","modals","meta","backtrack","action","verify","q_ratio",
                "exclaim","pos","neg","len"]
        X = np.array([[f[k] for k in keys] for f in feats], dtype=float)
        return X

    def fit(self, feats: List[Dict], labels: Dict[str, np.ndarray]):
        X = self._to_X(feats)
        for ax, y in labels.items():
            self.models[ax].fit(X, y)
        self.is_fit = True
        return self

    def predict_proba(self, feats: List[Dict]) -> Dict[str, np.ndarray]:
        assert self.is_fit, "StanceHead not fit"
        X = self._to_X(feats)
        out = {}
        for ax, m in self.models.items():
            proba = m.predict_proba(X)
            if proba.shape[1] == 2:
                out[ax] = proba[:,1]
            else:
                out[ax] = proba[:, -1] if proba.ndim==2 else proba
        return out
