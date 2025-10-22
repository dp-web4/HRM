
import regex as re
from typing import List, Dict, Iterable, Tuple
from .config import WINDOW_SIZE, WINDOW_STEP

def _load_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def compile_lexicons(root: str) -> Dict[str, List[str]]:
    import os
    lex = {}
    names = [
        "hedges.txt","modals.txt","meta_markers.txt","backtracks.txt",
        "action_verbs.txt","verify_markers.txt","positive_words.txt","negative_words.txt"
    ]
    for fn in names:
        lex[fn.split(".")[0]] = _load_list(os.path.join(root, "resources", "lexicons", fn))
    return lex

def simple_tokens(s: str) -> List[str]:
    return re.findall(r"\p{L}[\p{L}\p{N}'-]*", s.lower())

def count_phrases(text: str, phrases: List[str]) -> int:
    t = text.lower()
    count = 0
    for p in phrases:
        count += len(re.findall(re.escape(p), t))
    return count

def window_iter(turns: List[Dict], size: int = WINDOW_SIZE, step: int = WINDOW_STEP):
    for i in range(0, max(0, len(turns)-size+1), step):
        yield i, turns[i:i+size]

def extract_features(turns: List[Dict], lex: Dict[str, List[str]]) -> List[Dict]:
    feats = []
    for t in turns:
        text = t.get("text", "")
        toks = simple_tokens(text)
        n_tok = max(1, len(toks))

        f = {
            "hedges": count_phrases(text, lex["hedges"]) / n_tok,
            "modals": count_phrases(text, lex["modals"]) / n_tok,
            "meta": count_phrases(text, lex["meta_markers"]) / n_tok,
            "backtrack": count_phrases(text, lex["backtracks"]) / n_tok,
            "action": count_phrases(text, lex["action_verbs"]) / n_tok,
            "verify": count_phrases(text, lex["verify_markers"]) / n_tok,
            "q_ratio": text.count("?") / max(1, text.count(".")+text.count("?")+text.count("!")),
            "exclaim": text.count("!") / max(1, len(text)),
            "pos": sum(1 for w in toks if w in set(lex["positive_words"])) / n_tok,
            "neg": sum(1 for w in toks if w in set(lex["negative_words"])) / n_tok,
            "len": n_tok
        }
        feats.append(f)
    return feats
