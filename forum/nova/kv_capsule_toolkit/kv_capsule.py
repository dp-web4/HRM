import torch, json, os
import numpy as np

class KVCapsule:
    def __init__(self, kv, meta):
        self.kv = kv
        self.meta = meta

    @classmethod
    def from_past(cls, past_key_values, model, model_id="unknown"):
        # Create capsule from a model's past_key_values + metadata
        layers = len(past_key_values)
        heads = past_key_values[0][0].shape[1]
        head_dim = past_key_values[0][0].shape[-1]
        seq_len = past_key_values[0][0].shape[2]
        dtype = str(past_key_values[0][0].dtype)
        meta = {
            "model_id": model_id,
            "layers": layers,
            "heads": heads,
            "head_dim": head_dim,
            "seq_len": int(seq_len),
            "dtype": dtype,
            "rope_base": getattr(getattr(model.config, "rope_theta", None), "item", lambda: None)() if hasattr(model.config,"rope_theta") else None
        }
        return cls([ (k.cpu(), v.cpu()) for (k,v) in past_key_values ], meta)

    def to_disk(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.kv, os.path.join(path, "state.pt"))
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    @classmethod
    def from_disk(cls, path):
        kv = torch.load(os.path.join(path, "state.pt"))
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        return cls(kv, meta)

    def check_compatibility(self, other_model):
        # Check if capsule is compatible with another model
        return {
            "layers_match": self.meta["layers"] == other_model.config.num_hidden_layers,
            "heads_match": self.meta["heads"] == other_model.config.num_attention_heads,
            "head_dim_match": (self.meta["head_dim"] == other_model.config.hidden_size // other_model.config.num_attention_heads),
            "dtype": self.meta["dtype"]
        }

    def adapt_to(self, other_model):
        # Stub for experimental cross-feed adaptation (Procrustes/phase correction)
        compat = self.check_compatibility(other_model)
        if not all(compat.values()):
            print("⚠️ Incompatible KV; adaptation not implemented")
        return self.kv

def save_kv_capsule(capsule, path):
    capsule.to_disk(path)

def load_kv_capsule(path):
    return KVCapsule.from_disk(path)
