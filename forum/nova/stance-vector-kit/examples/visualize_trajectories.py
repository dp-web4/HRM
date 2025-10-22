
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import umap  # umap-learn
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

import matplotlib.pyplot as plt

def load_vectors(path_csv_or_npy):
    if path_csv_or_npy.endswith('.csv'):
        M = pd.read_csv(path_csv_or_npy).values
    elif path_csv_or_npy.endswith('.npy'):
        M = np.load(path_csv_or_npy)
    else:
        raise ValueError("Provide .csv or .npy file")
    return M

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    M = load_vectors(args.input)  # shape [T, D]
    steps = np.arange(M.shape[0])

    # PCA to 2D
    pca = PCA(n_components=2, random_state=0)
    P = pca.fit_transform(M)

    # Save PCA plot
    plt.figure()
    plt.plot(P[:,0], P[:,1], marker='o')
    for i,(x,y) in enumerate(P):
        if i % max(1, int(len(P)/10)) == 0:
            plt.text(x, y, str(i))
    plt.title("Stance Trajectory (PCA)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    pca_png = os.path.join(args.out_dir, "trajectory_pca.png")
    plt.savefig(pca_png, dpi=160)
    plt.close()

    # UMAP if available
    if _HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=0)
        U = reducer.fit_transform(M)
        plt.figure()
        plt.plot(U[:,0], U[:,1], marker='o')
        for i,(x,y) in enumerate(U):
            if i % max(1, int(len(U)/10)) == 0:
                plt.text(x, y, str(i))
        plt.title("Stance Trajectory (UMAP)")
        plt.xlabel("U1"); plt.ylabel("U2")
        plt.tight_layout()
        umap_png = os.path.join(args.out_dir, "trajectory_umap.png")
        plt.savefig(umap_png, dpi=160)
        plt.close()

    # Save coordinates
    np.save(os.path.join(args.out_dir,"coords_pca.npy"), P)
    if _HAS_UMAP:
        np.save(os.path.join(args.out_dir,"coords_umap.npy"), U)

    # Simple metrics summary
    metrics = {"pca_var_ratio": pca.explained_variance_ratio_.tolist()}
    with open(os.path.join(args.out_dir,"viz_metrics.json"),"w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="stance_windows.csv OR fusion_vectors.npy")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args)
