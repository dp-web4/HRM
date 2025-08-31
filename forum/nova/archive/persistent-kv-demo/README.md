# Persistent Transformer Memory (KV‑Cache Save & Resume) — Extended

This extended toolkit lets you **pause** an autoregressive Transformer mid‑generation by saving its
**KV‑cache**, and **resume** later from the same internal attention state — now with:
- ✅ Multiple save formats (pickle, gzip, torch.save)
- ✅ **Pruning** (keep only the most recent tokens)
- ✅ Minimal **Web UI** (Gradio) to step, save, and load KV interactively

Works with any Hugging Face causal LM that exposes `past_key_values` (e.g., GPT‑2).

> ⚠️ KV‑caches can be large for long contexts. Prefer `torch.save` or gzip, and prune as needed.

---

## Quick Start

### 0) Install
```bash
pip install -U transformers torch gradio
```

### 1) Save state after an initial prompt
```bash
python save_state.py   --model gpt2   --prompt "The architecture of meaning is"   --out kv_cache.pkl   --fmt pickle
```
or
```bash
python save_state.py --model gpt2 --prompt "..." --out kv_cache.pt --fmt torch
python save_state.py --model gpt2 --prompt "..." --out kv_cache.pkl.gz --fmt gzip
```

### 2) (Optional) Prune the KV to keep only the most recent N tokens
```bash
python prune_state.py --in kv_cache.pt --out kv_cache_pruned.pt --keep_last 512 --fmt torch
```

### 3) Resume later from the saved KV
```bash
python resume_state.py   --model gpt2   --kv kv_cache.pt   --fmt torch   --continue " deeply"   --steps 30   --temperature 0.8
```

### 4) Web UI (optional)
```bash
python app.py
```
- Enter a prompt, click submit to step forward.
- Use the buttons to **Save KV** / **Load KV** between runs.

---

## Files

- `save_state.py` – run a prompt, capture `past_key_values`, write to disk (`--fmt pickle|gzip|torch`)
- `resume_state.py` – load cached KV, feed continuation, continue generation
- `prune_state.py` – prune cache to keep only most recent tokens
- `utils_kv.py` – helpers (CPU<->device moves, pickle/gzip/torch save/load, pruning)
- `app.py` – minimal Gradio UI for interactive stepping and KV persistence
- `requirements.txt` – deps

---

## Notes

- Move KV to **CPU** before saving to reduce GPU memory pressure and improve portability.
- `torch.save` can be faster/smaller than pickle for large tensors; gzip further reduces size (slower).
- Pruning reduces memory linearly with tokens removed; keep in mind **coherence** may drop if you prune too aggressively.

---

## License
MIT (do whatever, but no warranty).
