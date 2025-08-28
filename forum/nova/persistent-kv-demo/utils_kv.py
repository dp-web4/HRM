import gzip, pickle
from typing import Tuple

import torch

KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]  # tuple of (K,V) per layer

def kv_to_cpu(past_key_values: KVCache) -> KVCache:
    out = []
    for k, v in past_key_values:
        out.append((k.detach().to("cpu"), v.detach().to("cpu")))
    return tuple(out)

def kv_to_device(past_key_values: KVCache, device: str) -> KVCache:
    out = []
    for k, v in past_key_values:
        out.append((k.to(device), v.to(device)))
    return tuple(out)

# --------- Save / Load formats ---------
def save_kv_pickle(path: str, past_key_values: KVCache) -> None:
    with open(path, "wb") as f:
        pickle.dump(past_key_values, f)

def load_kv_pickle(path: str) -> KVCache:
    with open(path, "rb") as f:
        return pickle.load(f)

def save_kv_gzip(path: str, past_key_values: KVCache) -> None:
    with gzip.open(path, "wb") as f:
        pickle.dump(past_key_values, f)

def load_kv_gzip(path: str) -> KVCache:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def save_kv_torch(path: str, past_key_values: KVCache) -> None:
    torch.save(past_key_values, path)

def load_kv_torch(path: str) -> KVCache:
    return torch.load(path, map_location="cpu")

def save_kv(path: str, past_key_values: KVCache, fmt: str = "pickle") -> None:
    fmt = fmt.lower()
    if fmt == "pickle":
        save_kv_pickle(path, past_key_values)
    elif fmt == "gzip":
        save_kv_gzip(path, past_key_values)
    elif fmt == "torch":
        save_kv_torch(path, past_key_values)
    else:
        raise ValueError(f"Unknown fmt: {fmt}")

def load_kv(path: str, fmt: str = "pickle") -> KVCache:
    fmt = fmt.lower()
    if fmt == "pickle":
        return load_kv_pickle(path)
    elif fmt == "gzip":
        return load_kv_gzip(path)
    elif fmt == "torch":
        return load_kv_torch(path)
    else:
        raise ValueError(f"Unknown fmt: {fmt}")

# --------- Pruning ---------
def prune_kv(past_key_values: KVCache, keep_last: int) -> KVCache:
    """Keep only the most recent `keep_last` tokens along the seq_len axis."""
    pruned = []
    for k, v in past_key_values:
        pruned.append((k[:, :, -keep_last:, :], v[:, :, -keep_last:, :]))
    return tuple(pruned)
