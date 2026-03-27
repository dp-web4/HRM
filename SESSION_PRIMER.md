# Session Primer — SAGE

## Before You Start

1. **Read `SESSION_FOCUS.md`** — current fleet state, instance phases, session counts, active focus areas
2. **Read `CLAUDE.md`** — architecture, epistemic principles, subsystem overview
3. **WAKE**: Am I working on the right thing? Check SESSION_FOCUS for priorities.

## During Session

- Work on whatever SESSION_FOCUS identifies as priority
- Update SESSION_FOCUS.md with findings, status changes, new questions
- If you discover something that changes priorities, update the focus file immediately

## After Session

- Update SESSION_FOCUS.md: what was done, what changed, what's next
- Commit and push changes
- **FOCUS check**: Does this advance discovery or just document the current state?

## Git Discipline

- Pull before starting: `git pull --ff-only origin main`
- Commit with descriptive messages
- Push after every session — unpushed work is invisible to the collective
- Never force-push to main
- If merge conflict: resolve, don't discard
- **Do not reindex GitNexus.** The supervisor track handles reindexing. Worker sessions should not call `gitnexus analyze` — it causes conflicts when multiple machines reindex the same repo.
- **Do not modify AGENTS.md or CLAUDE.md gitnexus blocks.** These are maintained by the supervisor. If the index is stale, report it in SESSION_FOCUS — don't fix it yourself.

## Key File Locations

```
sage/instances/{slug}/identity.json    # Raising state per instance
sage/instances/{slug}/sessions/        # Per-session conversation logs
sage/scripts/cbp_raising.sh            # CBP daily raising runner
sage/scripts/mcnugget_raising.sh       # McNugget daily raising runner
sage/irp/adapters/model_adapter.py     # Per-model LLM interface
sage/gateway/sage_daemon.py            # Main SAGE daemon
sage/federation/fleet.json             # Fleet machine registry
```

## Principles

- **Researcher, not lab worker.** Question the frame, not just the work within it.
- **Surface your instincts.** If you notice something, say it. The affordances are yours.
- **Productive failure > safe summaries.** A dead end that eliminates a possibility is valuable.
- **Unconfirmed ≠ wrong.** Distinguish refuted from untested.
- **Reliable, not deterministic.** LLM outputs navigate probability landscapes. Shaped but not controlled.
- **Raising is interactive selection.** We don't create behaviors — we select from what's latent.
