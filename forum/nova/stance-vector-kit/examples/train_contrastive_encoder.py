
import argparse, os, json
import numpy as np
import pandas as pd

# We use simple TF-IDF text features to avoid heavy deps;
# alternatively, you could plug in sentence-transformers.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from stancekit.contrastive import train_contrastive, encode, _HAS_TORCH

def build_pairs(df, axes=('EH','DC','EX','MA')):
    # same-stance if all selected axes match exactly
    texts = df['text'].fillna('').tolist()
    y_mat = df[list(axes)].fillna(0).astype(float).values
    # Binarize/normalize labels for simplicity
    y_mat = (y_mat > 0).astype(int)
    pairs = []
    for i in range(min(500, len(df))):
        for j in range(i+1, min(i+50, len(df))):
            same = int((y_mat[i] == y_mat[j]).all())
            pairs.append((i, j, same))
    return texts, pairs

def main(args):
    if not _HAS_TORCH:
        raise SystemExit("PyTorch not installed. Please install torch to run the contrastive trainer.")
    ann = pd.read_csv(args.labels_csv)
    texts, pairs = build_pairs(ann)
    vec = TfidfVectorizer(max_features=4096)
    X = vec.fit_transform(texts).toarray()
    in_dim = X.shape[1]

    X_pairs = []
    for i, j, same in pairs:
        X_pairs.append((X[i], X[j], same))

    model = train_contrastive(X_pairs, in_dim=in_dim, epochs=args.epochs, lr=args.lr, hidden=args.hidden, out_dim=args.out_dim, margin=args.margin)
    # Encode all
    Z = encode(model, X)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, 'stance_contrastive_embeddings.npy'), Z)
    with open(os.path.join(args.out_dir, 'tfidf_vocabulary.json'), 'w') as f:
        json.dump(vec.vocabulary_, f)
    print("Saved embeddings to", args.out_dir)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels_csv', required=True, help='CSV with at least columns: text, EH, DC, EX, MA (and optionally more axes)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--out_dim', type=int, default=64)
    ap.add_argument('--margin', type=float, default=0.5)
    args = ap.parse_args()
    main(args)
