
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def load_vectors(path):
    if path.endswith('.csv'):
        return pd.read_csv(path).values
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        raise ValueError('Provide .csv or .npy')

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    X = load_vectors(args.input)
    if args.pca_dims > 0 and X.shape[1] > args.pca_dims:
        pca = PCA(n_components=args.pca_dims, random_state=0)
        Xr = pca.fit_transform(X)
    else:
        Xr = X
    km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(Xr)
    labels = km.labels_

    # summarize segments (runs of same cluster)
    segments = []
    start = 0
    for i in range(1, len(labels)+1):
        if i==len(labels) or labels[i]!=labels[i-1]:
            segments.append({'cluster': int(labels[i-1]), 'start': int(start), 'end': int(i-1), 'length': int(i-start)})
            start = i
    pd.DataFrame({'cluster':labels}).to_csv(os.path.join(args.out_dir,'clusters_per_step.csv'), index=False)
    pd.DataFrame(segments).to_csv(os.path.join(args.out_dir,'segments.csv'), index=False)

    with open(os.path.join(args.out_dir,'cluster_summary.json'), 'w') as f:
        json.dump({'k': args.k, 'pca_dims': args.pca_dims, 'segments': segments}, f, indent=2)

    print('Wrote clusters_per_step.csv and segments.csv')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='stance_windows.csv or fusion_vectors.npy')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--pca_dims', type=int, default=8)
    args = ap.parse_args()
    main(args)
