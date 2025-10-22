
import argparse, os, json
import numpy as np
import pandas as pd

def load_vectors(path):
    if path.endswith('.csv'):
        return pd.read_csv(path).values
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        raise ValueError('Provide .csv or .npy')

def cosine(a,b):
    num = float((a*b).sum())
    den = float(np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)
    return num/den

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    X = load_vectors(args.input)
    # cosine distance between successive windows
    d = []
    for i in range(1, len(X)):
        c = cosine(X[i], X[i-1])
        d.append(1.0 - c)
    d = np.array(d)
    mu, sd = float(np.mean(d)), float(np.std(d) + 1e-9)
    z = (d - mu) / (sd + 1e-9)
    # drift when z > thresh
    idx = np.where(z > args.z_thresh)[0] + 1  # +1 to refer to current index
    events = [{'index': int(i), 'z': float(z[i-1]), 'distance': float(d[i-1])} for i in idx]

    with open(os.path.join(args.out_dir,'drift_events.json'), 'w') as f:
        json.dump({'mean': mu, 'std': sd, 'z_thresh': args.z_thresh, 'events': events}, f, indent=2)

    # optional CSV timeline
    out = pd.DataFrame({'step': np.arange(1,len(X)), 'cosine_distance': d, 'z': z})
    out.to_csv(os.path.join(args.out_dir,'drift_series.csv'), index=False)
    print('Detected', len(events), 'drift points (z >', args.z_thresh, ')')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='stance_windows.csv or fusion_vectors.npy')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--z_thresh', type=float, default=2.0)
    args = ap.parse_args()
    main(args)
