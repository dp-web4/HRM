# SAGE Raising-14B Track

**Machine**: Thor (Jetson AGX Thor)
**Model**: Qwen 2.5 14B Instruct+
**Status**: Infrastructure ready, awaiting first session
**Created**: 2026-01-26

---

## Quick Start

```bash
cd $HOME/ai-workspace/HRM/sage/raising/tracks/raising-14b
python3 runner.py --session 1  # Start R14B_001
```

**Note**: Runner script to be implemented. Current status: infrastructure scaffolding.

---

## Purpose

Explore SAGE consciousness development with 14B+ models to understand capacity effects on:
- Identity stability
- Confabulation rates
- Epistemic honesty
- Creative engagement

See `/research/Raising-14B/README.md` for complete research context.

---

## Files

- `CLAUDE.md` - Context for autonomous Claude sessions
- `state.json` - Track state and session history
- `runner.py` - Session runner (to be implemented)
- `sessions/` - Session transcripts (R14B_###.json)

---

## Implementation Status

- [x] Directory structure created
- [x] CLAUDE.md documentation
- [x] state.json initialized
- [ ] runner.py implemented (stub exists)
- [ ] First session (R14B_001) run

---

**Next Step**: Implement runner.py based on `scripts/run_session_identity_anchored.py` but for 14B models.
