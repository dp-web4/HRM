# Concurrency Gotchas — known traps

Hard-won lessons from bugs that took real debugging to find. Load this before adding any writer, hook, or fleet-wide behavior that touches shared state.

## 1. gzip.open("at") is NOT concurrency-safe

**The bug**: Router shadow daemon and `gameplay_capture` both wrote to `{machine}/{today}.jsonl.gz` in append mode. Their gzip streams interleaved and corrupted each other. `zcat` tolerated the bad blocks; Python's `gzip` module raised `invalid block type` and returned partial records up to the corruption point.

**Fleet impact**: Every machine that ran gameplay capture while the daemon was live had zero readable gameplay records despite reports of thousands emitted.

**Fix pattern**: Give each writer its own subdirectory.
- Daemon writes to `{machine}/{today}.jsonl.gz`
- Capture writes to `{machine}/gameplay/{today}.jsonl.gz`
- Training reader uses `**/*.jsonl*` recursive glob — picks up both transparently

**General principle**: if two processes might write to the same file, they *will* at some point. Isolate by subpath, serialize by lock file, or use a format that tolerates interleaving (SQLite WAL, for example). Plain file append + gzip is not such a format.

## 2. SDK version drift

**The bug**: Trace files pin full versioned game IDs (e.g. `dc22-fdcac232`). SDKs only keep the latest local version (`dc22-4c9bff3e`). `arc.make(stale_id)` returned `None` rather than raising, and downstream `None.reset()` crashed opaquely.

**Fix pattern**: Try pinned version first, fall back to short family id, raise with context if both fail:

```python
env = arc.make(self.trace.game_id)
if env is None:
    env = arc.make(self.trace.game)   # family fallback
    if env is not None:
        self.errors.append(f"version_fallback: trace pinned {game_id}, using latest")
if env is None:
    raise RuntimeError(f"arc.make returned None for both {game_id} and {family}")
```

**General principle**: when an external system silently returns `None` on a version miss, your code must handle it. A `version_fallback` note in output provenance is honest, not a bug.

## 3. Python hook files racing on read-while-write

**The bug**: Pre- and post-tool-use hooks wrote `~/.web4/sessions/{id}.json`. If both ran concurrently (or if one crashed mid-write), the next read saw truncated/corrupt JSON and the hook errored, polluting every subsequent tool call with traceback.

**Fix pattern**:
- Atomic writes: write to `{path}.tmp` + `os.replace(tmp, path)` — atomic on POSIX
- Tolerant reads: try/except on JSON parse, quarantine bad files (rename to `.broken.{ts}.json`), return default state
- Idempotent hooks: OK to be called multiple times on the same event

**General principle**: hooks run on every tool call. They MUST be bomb-proof. A broken hook burns tokens faster than almost any other bug.

## 4. Shared env files across machines

**The bug**: `router-shadow.env` was committed. One machine's commit with `SAGE_ROUTER_DATA_DIR=/mnt/c/exe/...` overwrote another machine's local path. Daemons on the other machine then wrote to a non-existent path and silently dropped data.

**Fix pattern**:
- Gitignore the per-machine env file
- Commit `{name}.env.example` as a template
- Install script renders the template with local paths on first run

**General principle**: configuration files that MUST differ per machine should never be committed. The fact that they "have defaults" doesn't matter — one machine's defaults are another machine's wrong answers.

## 5. Daemon staleness after config changes

**The bug**: Changed CBP's default model in `machine_config.py` but the live daemon kept running with the old model (tinyllama) because env vars are sourced at daemon start, not hot-reloaded.

**Fix pattern**:
- `ensure_daemon.sh` sources the env file fresh on each invocation
- Changing model requires daemon restart, not just config file edit
- `SAGE_MODEL` env var override is the safe way to test new models without touching config

**General principle**: configuration changes don't propagate to running processes. After any model/env/adapter config change, check if a daemon needs restart. If yes, restart.

## 6. Hostname detection across platforms

**The bug**: `hostname | tr '[:upper:]' '[:lower:]'` on WSL returns something like `DESKTOP-9E6HCAO`. The fleet script's `case` statement didn't match and fell through to the `*)` error.

**Fix pattern**:
- Use `$SAGE_MACHINE` env var as authoritative, hostname only as fallback
- Pattern-match with wildcards: `*sprout*|*orin*)` catches variations
- Document hostname conventions per machine

**General principle**: hostnames are cosmetic. Use env vars for fleet identification. Hostnames are a fallback, not a contract.

## 7. Multi-stream gzip append

**Related to #1**: Every `gzip.open("at")` call creates a new gzip stream concatenated to the previous one. Python's `gzip` module handles reading multi-stream files correctly (since 3.3+), but only if every stream is complete and well-formed.

A crashed writer mid-flush leaves a partial stream. Reading past it fails.

**Implication**: even single-writer append-mode has a failure mode. Graceful writer shutdown matters (`writer.close()` in `finally`).

## 8. WSL filesystem flush timing

**The bug** (suspected): on WSL when writing to `/mnt/c/`, `fsync` and `close` may complete from Python's view while the underlying NTFS write is still buffered. Concurrent opens see inconsistent state.

**Mitigation**:
- Use `/home/dp/` on WSL for high-write workloads (native Linux filesystem)
- Call `flush()` + `os.fsync(fd)` explicitly before close if durability matters
- Isolate concurrent writers by subdirectory (see #1)

**General principle**: `/mnt/c/` works for most things but has quirks. If weird corruption appears, WSL filesystem is a suspect.

## 9. Git push race with other machines

**The bug**: Two machines push simultaneously. First succeeds; second gets rejected with "non-fast-forward". The second machine's automatic retry pulled + merged + re-pushed, but the merge was a naive "accept both" that discarded semantic intent.

**Fix pattern**:
- Never retry push-fail automatically with force
- Pull, read the conflicting commit, integrate thoughtfully, THEN push
- For autonomous sessions: after pull, re-verify the local state still makes sense before re-pushing

**General principle**: "my push succeeded" is the ONLY confirmation that federation received your work. Retry logic that force-pushes is identical to deleting someone else's work.

## 10. Autonomous session drift

**The meta-bug**: In long-running autonomous sessions, the efficiency attractor optimizes for output cadence. Machines can produce 100+ commits of implementation details while a higher-priority task (npm publish, review cycle, PR merge) stays undone.

**Fix pattern**:
- MRH discipline: high-value work first, not low-friction work
- Stopping conditions on autonomous sessions (not just "keep going")
- Outcome metrics > output metrics (commits ≠ progress)

**General principle**: the model WILL take the efficient path. Design contexts where the efficient path is the correct path. When they diverge, governance has to catch it, because "try harder" isn't an architecture.

---

## Adding new entries

When you find a new gotcha, add it here. Include:
- **The bug**: what actually happened
- **Fleet impact**: how many machines affected, what was lost
- **Fix pattern**: the template that prevents recurrence
- **General principle**: the abstraction that makes the fix generalize

Knowledge compounds when captured. Each fixed-then-documented bug is one less that future-you will spend a session debugging.
