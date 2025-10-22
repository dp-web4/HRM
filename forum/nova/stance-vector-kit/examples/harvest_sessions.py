
import argparse, os, shutil
from pathlib import Path

ARTIFACTS = [
    'metrics.json',
    'drift_events.json',
    'cluster_summary.json',
    'trajectory_pca.png',
    'trajectory_umap.png',
    'ab_overlay.png',
    'stance_windows.csv',
    'fusion_vectors.npy'
]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='parent folder containing session subfolders')
    ap.add_argument('--out_root', required=True, help='destination folder to collect standardized copies')
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Each immediate subdir of root is treated as a session
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        dest = out_root / sub.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in ARTIFACTS:
            src = sub / name
            if src.exists():
                shutil.copy2(src, dest / name)
                print('copied', src, '->', dest/name)
    print('Harvest complete ->', out_root)
