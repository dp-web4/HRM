
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_vectors(path):
    if path.endswith('.csv'):
        return pd.read_csv(path).values
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        raise ValueError('Provide .csv or .npy')

def to_2d(M, pca=None):
    if pca is None:
        pca = PCA(n_components=2, random_state=0)
        P = pca.fit_transform(M)
        return P, pca
    return pca.transform(M), pca

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    A = load_vectors(args.input_a)
    B = load_vectors(args.input_b)

    # truncate to same length for alignment
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]

    # project together to avoid axis mismatch
    pca = PCA(n_components=2, random_state=0).fit(np.vstack([A,B]))
    A2, _ = to_2d(A, pca)
    B2, _ = to_2d(B, pca)

    # compute per-step cosine similarity
    def cos(a,b):
        num = float((a*b).sum())
        den = float(np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)
        return num/den
    cosine_series = [cos(A[i],B[i]) for i in range(n)]
    mean_cos = float(np.mean(cosine_series))

    # plot overlay
    plt.figure()
    plt.plot(A2[:,0], A2[:,1], '-o', label='A', linewidth=2)
    plt.plot(B2[:,0], B2[:,1], '-o', label='B', linewidth=2)
    for i in range(0, n, max(1, n//10)):
        plt.text(A2[i,0], A2[i,1], f'A{i}')
        plt.text(B2[i,0], B2[i,1], f'B{i}')
    plt.legend()
    plt.title(f'A/B Stance Trajectories (mean cosine={mean_cos:.3f})')
    plt.tight_layout()
    out_png = os.path.join(args.out_dir, 'ab_overlay.png')
    plt.savefig(out_png, dpi=160)
    plt.close()

    with open(os.path.join(args.out_dir,'ab_metrics.json'), 'w') as f:
        json.dump({'mean_cosine': mean_cos}, f, indent=2)
    print('Saved', out_png)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_a', required=True, help='CSV or NPY for session A')
    ap.add_argument('--input_b', required=True, help='CSV or NPY for session B')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    main(args)
