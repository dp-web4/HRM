import argparse, json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils_kv import kv_to_cpu, save_kv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", help="HF model id or local path")
    ap.add_argument("--prompt", required=True, help="Initial prompt to build KV‑cache from")
    ap.add_argument("--out", default="kv_cache.pkl", help="Where to save KV‑cache")
    ap.add_argument("--fmt", default="pickle", choices=["pickle","gzip","torch"], help="Save format")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token  # simple pad

    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    model.eval()

    inputs = tok(args.prompt, return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        past = out.past_key_values  # tuple of (K,V) per layer

    past_cpu = kv_to_cpu(past)
    save_kv(args.out, past_cpu, fmt=args.fmt)

    meta = {
        "model": args.model,
        "prompt": args.prompt,
        "kv_path": args.out,
        "fmt": args.fmt,
        "device_saved": "cpu",
        "layers": len(past_cpu),
        "heads": int(past_cpu[0][0].shape[1]) if len(past_cpu) else None,
        "seq_len": int(past_cpu[0][0].shape[2]) if len(past_cpu) else 0,
        "head_dim": int(past_cpu[0][0].shape[3]) if len(past_cpu) else None,
    }
    Path("meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved KV‑cache to {args.out} ({args.fmt})")
    print(f"Meta:\n{json.dumps(meta, indent=2)}")

if __name__ == "__main__":
    main()
