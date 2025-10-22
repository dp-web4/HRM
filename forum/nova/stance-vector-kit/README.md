
# Stance Vector Kit (SVK)

A lightweight toolkit for encoding **stance-vectors** from dialog transcripts and comparing stance across contexts.

**Stance** = *how* an entity relates to knowledge and action (not *what* it says).
This kit extracts transparent features, trains a small multi-label classifier head,
and fuses signals into a stance vector **s ∈ ℝ^K** (default K=12).

## Axes (default)
1. EH — Epistemic Humility
2. DC — Declarative Confidence
3. EX — Exploratory Drive
4. MA — Meta-Awareness
5. RR — Revision Readiness
6. AG — Autonomy/Agency
7. AS — Attention Stability
8. SV — Skepticism/Verification
9. VA — Affect Valence
10. AR — Arousal/Energy
11. IF — Instruction-following vs Initiative
12. ED — Evidence Density

> Extend/reduce axes in `src/stancekit/config.py`.

## Quick start

```bash
pip install -r requirements.txt
python examples/demo_pipeline.py       --input examples/sample_transcript.jsonl       --out_dir /tmp/svk_out
```

Outputs:
- `stance_windows.csv` — windowed stance vectors (raw & smoothed)
- `metrics.json` — cosine similarity, flicker index, etc. (if `--baseline` provided)

## Data format
JSONL file, one object per line:
- `speaker` (str): e.g., "human", "model"
- `text` (str)
- `timestamp` (optional float seconds)
- `session_id` (optional str)

## License
MIT
