
import numpy as np

def cosine_similarity(a, b):
    a = np.array(a); b = np.array(b)
    num = float((a*b).sum())
    den = float(np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)
    return num/den

def flicker_index(series):
    """
    series: list of vectors (np arrays). Returns mean frame-to-frame cosine distance.
    """
    if len(series) < 2:
        return 0.0
    ds = []
    for i in range(1, len(series)):
        num = float((series[i]*series[i-1]).sum())
        den = float(np.linalg.norm(series[i])*np.linalg.norm(series[i-1]) + 1e-9)
        c = num/den
        ds.append(1.0 - c)  # cosine distance
    return float(np.mean(ds))
