import argparse, json
from pathlib import Path

from utils_kv import load_kv, save_kv, prune_kv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to saved KV")
    ap.add_argument("--out", required=True, help="Path to save pruned KV")
    ap.add_argument("--keep_last", type=int, required=True, help="How many most recent tokens to keep")
    ap.add_argument("--fmt", default="pickle", choices=["pickle","gzip","torch"], help="Serialization format for both load and save")
    args = ap.parse_args()

    past = load_kv(args.in_path, fmt=args.fmt)
    pruned = prune_kv(past, keep_last=args.keep_last)
    save_kv(args.out, pruned, fmt=args.fmt)

    meta = {
        "source": args.in_path,
        "output": args.out,
        "fmt": args.fmt,
        "keep_last": args.keep_last,
    }
    Path("prune_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Pruned KV written to {args.out} (kept {args.keep_last} tokens).")

if __name__ == "__main__":
    main()
