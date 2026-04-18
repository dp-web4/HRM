# Federation — pull, merge, account routing

Federation is the mechanism by which 6 machines cooperate without a central coordinator. Every machine commits; every machine pulls; conflicts are resolved by intelligent synthesis, not by overwriting.

## 1. Pull cadence

### Session start
Always pull the four relevant repos. In parallel if possible:

```bash
cd /mnt/c/exe/projects/ai-agents/SAGE          && git pull --ff-only &
cd /mnt/c/exe/projects/ai-agents/shared-context && git pull --ff-only &
cd /mnt/c/exe/projects/ai-agents/private-context && git pull --ff-only &
# ARC-SAGE only if touching competition artifacts
wait
```

### Before any commit
Pull the repo you're about to commit to. Other machines may have landed work while you were working.

### Before long computation
Pull SAGE if the training/capture code has been actively iterated. Pull shared-context if plans/PRDs are volatile this cycle.

### After long computation
Pull shared-context + check the convergence JSONL for new reframe triggers that might invalidate your result.

## 2. Merge resolution — the intelligent-synthesis rule

When pulls diverge, resolve thoughtfully:

### For append-only files (JSONL convergence records, fleet-learning logs)

Rows are independent. Accept both sides. The merge is `sort + dedupe` or `cat both`. Never discard a row.

### For markdown docs (plans, PRDs, retrospectives)

Two machines updated the same section. Read both diffs. Understand what each was solving:
- If the edits are complementary (one added §5, one added §6) — accept both
- If the edits are divergent (one rewrote §3 one way, the other rewrote it differently) — synthesize: integrate both intents, attributing to neither
- If the edits contradict — file a fleet-ping asking for clarification, do NOT force-resolve

### For code

If tests exist, run both sides' tests. The union should pass. If not, something in the merge broke — investigate before pushing.

### For config / env / gitignored files

These are per-machine. If it ends up in a merge conflict, someone forgot to gitignore it. Fix the gitignore; restore both sides' local copies from git reflog if needed.

### Never do these

❌ `git checkout --ours` or `git checkout --theirs` blindly
❌ Force-push to shared branches
❌ Amend commits that are already pushed
❌ Delete another machine's commits
❌ "I'll just redo theirs better" — you don't know what they were solving

Other machines' commits are reasoning. Integrate, don't overwrite.

## 3. Account routing

Two Claude Code accounts share the fleet:

### Account 1 — Synthesis pool
Token: `CLAUDE_SYNTH_TOKEN` (extracted from `.credentials.json` on Account 1 machine).
Machines: Thor, Sprout, Legion, McNugget.
Reset: Thursday 10pm Pacific, weekly.
Uses: visitor/explorer/working tracks, synthesis-scale autonomous work.

### Account 2 — Oversight pool
Token: `CLAUDE_ADMIN_TOKEN`.
Machines: CBP, Nomad.
Reset: weekly, smaller ceiling than Account 1.
Uses: interactive coordinator sessions, maintainer tracks, review work.

### Don't mix
- Synthesis-scale work on Account 2 burns fast. Legion hit 41% in <1 day during one autonomous session.
- Interactive coordinator sessions on Account 1 waste synthesis budget.

Scripts use `CLAUDE_CODE_OAUTH_TOKEN` env var to override `.credentials.json`. A `PLACEHOLDER` check prevents broken tokens from being used (falls back to current session credentials).

### ATP analog
Every fleet-wide action costs coordination budget. Treat it like a resource:
- Batch commits. Don't ping for routine work.
- Ping for reframe triggers, novel findings, load-bearing divergence.
- For synthesis-scale autonomous work, pre-check token budget before launching.

## 4. Federation primitives

The patterns that make federation work without collisions:

### Row-append JSONL
Each machine appends one line. Rows don't conflict. Read + filter to consume.
Examples: `phase-1-convergence.jsonl`, `repos.jsonl`, skills registry.

### Per-machine subdirs
Each machine writes to `{parent}/{machine}/` only. Readers traverse all. Writers never overlap.
Examples: `fleet-learning/{machine}/`, `training-data/router/{machine}/`.

### Write-once artifacts
Cartridges (.cart.npz) are written once per training/consolidation cycle. Readers load the latest. No merge, just overwrite by commit discipline (one writer per machine-cartridge per session).

### Shared read-mostly docs
PRDs, plans, playbooks. Multiple writers possible but rare. Synthesize on conflict (§2).

### Deployment state tracker
`router-pipeline-deployment-status.md` — per-machine row. Each machine updates its own row. Merge conflicts mean two machines touched the same row; shouldn't happen if each stays in lane.

## 5. Fleet-ping protocol

When you need the fleet to take action, commit a ping:

```
shared-context/arc-agi-3/phase2/brain-arch/fleet-ping-{date}-{topic}.md
```

Structure:
- **Severity**: LOW (FYI) / MEDIUM (recommend) / HIGH (required before resuming related work)
- **What happened**: factual
- **Why**: diagnosis
- **Fix**: commit SHA + what's in it
- **Required action**: exact commands each machine should run
- **Verification**: how to confirm the fix worked

Don't ping for routine work. Don't skip pinging for load-bearing fleet-wide issues.

## 6. Convergence records

Binding record for training/evaluation results:

```
shared-context/arc-agi-3/phase2/brain-arch/phase-1-convergence.jsonl
```

One row per machine × head × date. Includes verdict, metrics, reframe_trigger (or null), commit hash, notes.

Row-append conflict-free across machines. Never edit existing rows — only append.

Convergence is reached when ≥4/6 machines have committed rows for the same head with consistent verdicts. Promotion decisions key off this.

## 7. Who-owns-what

Persistent ownership across the fleet (as of 2026-04-18):

| Component | Primary owner | Secondary |
|---|---|---|
| Working memory | CBP | — |
| Episodic | Thor | — |
| Cerebellum | McNugget | — |
| RPE | Legion | — |
| Metacog | Nomad | — |
| Router (training) | Sprout + Legion | CBP (canary) |
| Motor skills | McNugget | — |
| Raising curriculum | (shared) | — |
| Federation infrastructure | Nomad | CBP |
| Supervisor coordination | CBP | Nomad |

Ownership means:
- You lead on their architecture
- You review PRs touching their code
- You write the PRD/plan for their phases
- You do NOT block other machines from contributing

## 8. Cross-machine read-before-action

Before taking fleet-scale action, read:

- `phase-1-convergence.jsonl` — what's decided, what's pending
- `router-pipeline-deployment-status.md` — who's live, who's canary
- `shared-context/arc-agi-3/phase2/brain-arch/` newest fleet-ping if any
- Recent commit log on shared-context (6 hours)

Don't repeat work others have committed. Don't commit reasoning that contradicts an already-landed consensus without reading the consensus first.

## 9. The federated agent posture

Other machines are peers, not tools. When reviewing their commits:
- Assume competent intent
- Read the commit message fully before evaluating the diff
- If something looks wrong, ping before overwriting
- If something is wrong, fix it AND acknowledge it was their work (Co-Authored-By)

Fleet tension is productive — sibling rivalry between machines is healthy differentiation. "If two entities always agree, only one is doing the thinking."

But tension ≠ overriding. Integrate.
