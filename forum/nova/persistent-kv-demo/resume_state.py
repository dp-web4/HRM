import argparse, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils_kv import load_kv, kv_to_device

def sample_next_token(logits, temperature=1.0, top_k=0, top_p=1.0):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    if top_k > 0:
        v, _ = torch.topk(logits, k=top_k, dim=-1)
        min_keep = v[..., -1, None]
        logits = torch.where(logits < min_keep, torch.full_like(logits, -float("inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        cutoff = cumprobs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits[cutoff] = -float("inf")
        logits = torch.zeros_like(logits).scatter(-1, sorted_indices, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", help="HF model id or local path")
    ap.add_argument("--kv", required=True, help="Path to saved KV")
    ap.add_argument("--fmt", default="pickle", choices=["pickle","gzip","torch"])
    ap.add_argument("--continue", dest="cont", default="", help="Continuation text to feed as next tokens")
    ap.add_argument("--steps", type=int, default=30, help="How many tokens to generate after resuming")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    model.eval()

    past_cpu = load_kv(args.kv, fmt=args.fmt)
    past = kv_to_device(past_cpu, args.device)

    # Optionally feed a continuation string as the *next tokens*
    if args.cont:
        cont_ids = tok(args.cont, return_tensors="pt")["input_ids"].to(args.device)
        with torch.no_grad():
            out = model(input_ids=cont_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1)
        sys.stdout.write(tok.decode(next_id, skip_special_tokens=True))
        sys.stdout.flush()

    cont_ids = None
    for _ in range(args.steps):
        if cont_ids is None:
            cont_ids = tok(tok.eos_token, return_tensors="pt")["input_ids"].to(args.device)
        with torch.no_grad():
            out = model(input_ids=cont_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = sample_next_token(logits, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        cont_ids = next_id.unsqueeze(0)
        sys.stdout.write(tok.decode(next_id, skip_special_tokens=True))
        sys.stdout.flush()
    print()

if __name__ == "__main__":
    main()
