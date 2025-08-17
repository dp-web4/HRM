
# transforms.py
# Tiny helpers for context shifting & semantic variations without external deps

def context_shift(description: str, new_context: str) -> str:
    return f"[{new_context}] {description}"

def semantic_variation(text: str) -> str:
    # extremely naive synonym-like tweak
    replacements = {
        "object": "item",
        "grasp": "grab",
        "place": "set",
        "move": "shift",
        "location": "spot"
    }
    out = text
    for k,v in replacements.items():
        out = out.replace(k, v)
    return out
