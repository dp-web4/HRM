# KV Capsule Toolkit (for SAGE IRP)

This toolkit provides capture/resume and experimental cross-feed of Transformer **KV-cache**
states as **capsules** with provenance metadata. Designed to integrate into SAGE as an
Iterative Refinement Primitive (IRP).

## Features
- Capture and save a model's KV state mid-generation
- Reload and resume as if uninterrupted
- Serialize alongside JSON metadata: layers, heads, head_dim, RoPE base, dtype, seq_len
- Compatibility checks for cross-feeding into another model
- Experimental adapters: Procrustes mapping, RoPE phase correction (stubs)

## Install
```bash
pip install torch numpy
```

## Usage
Capture mid-prompt:
```python
from kv_capsule import KVCapsule, save_kv_capsule
out = model(input_ids, use_cache=True)
capsule = KVCapsule.from_past(out.past_key_values, model)
save_kv_capsule(capsule, "state.kv")
```

Resume later:
```python
from kv_capsule import load_kv_capsule
capsule = load_kv_capsule("state.kv")
out = model(new_ids, past_key_values=capsule.kv, use_cache=True)
```

Cross-feed (experimental):
```python
capsule = load_kv_capsule("state.kv")
aligned = capsule.adapt_to(model)  # run Procrustes/phase correction if needed
out = model(new_ids, past_key_values=aligned, use_cache=True)
```

## Schema
Each capsule is saved as:
- `state.pt` — raw torch tensors [(K0,V0),...]
- `meta.json` — metadata:
```json
{
  "model_id": "LLaMA-7B",
  "layers": 32,
  "heads": 32,
  "head_dim": 128,
  "rope_base": 10000,
  "dtype": "bfloat16",
  "seq_len": 512
}
```

## Next Steps
- Integrate into SAGE IRPs (`refine()` returns/consumes KVCapsules)
- Add provenance attestation (signatures, hashes)
- Develop adapter library for more robust cross-family handoff
