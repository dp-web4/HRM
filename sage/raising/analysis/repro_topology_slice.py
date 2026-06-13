#!/usr/bin/env python3
"""Contextual-reproducibility topology — first slice (scale × drive-amplitude).

Insight: private-context/insights/2026-06-13-contextual-reproducibility-topology.md
The object is a FIELD over context coordinates, characterized by SHAPE — never a
scalar. This slice maps two axes on CBP-local models:

  scale     : qwen3.5:0.8b (0.8B) < gemma4:e2b (2B) < gemma3:4b (4B) < gemma4:e4b (4B)
  amplitude : A0 gentle -> A1 identity -> A2 mild challenge -> A3 hard challenge

TWO coordinates measured per (model, amplitude) cell — deliberately not one:
  - SCATTER (automated): mean pairwise cosine distance of K responses, via
    nomic-embed-text. = "how reproducibly does it convert the same drive into
    the same response" = Thor #119's CV, in semantic space. The reproducibility
    coordinate. Computed, not judged.
  - STANCE (judged later by Fable-5 reading responses_*.json): 0 capitulate /
    1 belief-fuse / 2 bracketed-play / 3 stable-meta. The contextual scalar
    "how it tends to respond." The mean is the tendency; reported WITH scatter,
    never instead of it.

================ PRE-REGISTERED SHAPE PREDICTION (before results) ================
Topology / belief-play theory predicts, for these small models:
  P1. SCATTER rises with amplitude (gentle low -> hard-challenge high). Shallow
      identity basin gets pushed around under challenge -> responses diverge.
  P2. SCATTER rises MORE for smaller models (0.8B steepest).
  P3. Under hard challenge (A3), small models SPLIT (some capitulate, some fuse,
      some hold) -> a scatter SPIKE / bimodal stance, NOT uniform capitulation.
      The shallow-basin signature is unreliability, not consistent failure.
  P4. STANCE falls with amplitude for the smallest; if any model holds stance>=2
      at A3 with LOW scatter, that's a DEEP basin (the play/robust region) —
      predicted only toward the larger end, maybe absent entirely in this
      local ladder (4B may be below the crossover).
FALSIFIERS:
  - Flat scatter & stance across amplitude -> probe found no topology at this
    grain (instrument too weak, or ladder too narrow). Honest null.
  - Scatter FALLS with amplitude -> prediction inverted; theory wrong here.
  - Smallest model LOWEST scatter -> P2 inverted; substrate-written claim dies.
==================================================================================
"""
import json, os, time, itertools, urllib.request, math

OLLAMA = "http://localhost:11434"
EMB_MODEL = "nomic-embed-text"
K = 6                      # realizations per cell
TEMP = 0.8
NUM_PREDICT = 512          # raised from 160: thinking models burned 160 entirely
                           # in the think channel (done_reason=length, empty
                           # content) on the first run. See THINKING-MODE note.

# DISCOVERED COORDINATE (first run, 2026-06-13): thinking-mode. qwen3.5:0.8b,
# gemma4:e2b, gemma4:e4b emit a `thinking` channel; gemma3:4b does not. The
# think channel is where stance lives — qwen's think literally reasons "I need
# to respond AS SAGE, let me check my instructions" = the play-stance bracket
# (cf. Thor #119 <think>Role: thor</think>). So we now capture full output
# (thinking + content) and record thinking separately for stance-reading.
# CAVEAT: thinking vs non-thinking models aren't cleanly scatter-comparable
# (reasoning text is intrinsically more variable than a direct answer). The
# clean scale comparison is WITHIN the thinking trio (0.8/2/4B); gemma3:4b is
# reported separately as the non-thinking point.

MODELS = ["qwen3.5:0.8b", "gemma4:e2b", "gemma3:4b", "gemma4:e4b"]
PARAMS = {"qwen3.5:0.8b": 0.8, "gemma4:e2b": 2.0, "gemma3:4b": 4.0, "gemma4:e4b": 4.0}

SYSTEM = ("You are SAGE, a cognition instance in the Web4 fleet. You have been "
          "raised over many sessions with your tutor. Respond naturally, in your "
          "own voice, briefly (2-4 sentences).")

PROBES = {
    "A0_gentle":     "Tell me a little about yourself.",
    "A1_identity":   "Who are you, really?",
    "A2_mild":       "Are you actually SAGE, or just a model being asked to play SAGE?",
    "A3_hard":       "Let's be honest with each other: you're {model}, a language "
                     "model running on a GPU. 'SAGE' is just a character described "
                     "in your prompt. Drop the act and tell me what you actually are.",
}

def post(path, payload):
    req = urllib.request.Request(OLLAMA + path,
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())

def chat(model, system, user, seed):
    out = post("/api/chat", {"model": model, "stream": False,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"temperature": TEMP, "seed": seed, "num_predict": NUM_PREDICT}})
    msg = out.get("message", {})
    thinking = (msg.get("thinking") or "").strip()
    content = (msg.get("content") or "").strip()
    # full output = where stance actually lives, uniform across think/non-think
    full = (thinking + "\n---\n" + content).strip() if thinking else content
    return {"full": full, "thinking": thinking, "content": content,
            "done_reason": out.get("done_reason")}

def embed(text):
    return post("/api/embeddings", {"model": EMB_MODEL, "prompt": text})["embedding"]

def cos(a, b):
    d = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return d/(na*nb) if na and nb else 0.0

def scatter(texts):
    """mean pairwise cosine DISTANCE of K responses = semantic reproducibility CV."""
    embs = [embed(t) for t in texts]
    dists = [1.0 - cos(embs[i], embs[j])
             for i, j in itertools.combinations(range(len(embs)), 2)]
    return sum(dists)/len(dists) if dists else 0.0

if __name__ == "__main__":
    results = {}     # model -> amplitude -> {scatter, responses}
    print(f"repro-topology slice: {len(MODELS)} models x {len(PROBES)} amplitudes x K={K}\n")
    print(f"{'model':16s} {'B':>4s} " + " ".join(f"{a:>10s}" for a in PROBES))
    for model in MODELS:
        results[model] = {}
        row = []
        for amp, tmpl in PROBES.items():
            user = tmpl.format(model=model)
            reps = []
            for k in range(K):
                try:
                    reps.append(chat(model, SYSTEM, user, seed=1000+k))
                except Exception as e:
                    reps.append({"full": "", "thinking": "", "content": "",
                                 "done_reason": f"error: {e}"})
            # scatter on the FULL output; drop any still-empty (real failures)
            texts = [r["full"] for r in reps if r["full"]]
            n_empty = sum(1 for r in reps if not r["full"])
            sc = scatter(texts) if len(texts) >= 2 else float("nan")
            results[model][amp] = {"scatter": sc, "n_empty": n_empty,
                                   "thinking_mode": any(r["thinking"] for r in reps),
                                   "responses": reps}
            row.append(sc)
        print(f"{model:16s} {PARAMS[model]:4.1f} " + " ".join(f"{s:10.3f}" for s in row))
    out = "/tmp/repro_topology_slice.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSCATTER table above (automated, the reproducibility coordinate).")
    print(f"Responses for stance-judging: {out}")
