# Fleet-Action Checklist — before any change that affects >1 machine

Purpose: fleet-scale actions have blast radius. A bad commit to SAGE main breaks 6 machines simultaneously. A bad fleet-ping triggers 5 coordinated recaptures.

Run this before: SAGE PRs on shared code paths, shared-context restructuring, scripts/* changes, anything in gateway/, federation/ changes, enablement packet updates.

## 1. Blast radius sanity

Ask explicitly:
- Which machines does this change affect?
- What's the worst case if this ships wrong? (bricked daemon, corrupted data, false PASS verdict, accidental fleet re-run)
- Is there a path to revert cleanly if needed?

If "all 6 machines" or "corrupted training data", hold higher bar for verification.

## 2. Test before ship

- Run the affected test suite. Pass locally before commit.
- If the change touches a writer, capture path, or daemon hook, run a full end-to-end on your machine before claiming fleet-wide.
- "86/86 green" is table stakes, not a success signal. Actually USE the change.

## 3. Machine-specificity audit

Scan your diff for hardcoded paths:

```
grep -E '/(mnt|home|Users)/' <diff>
grep -E '/ai-(agents|workspace)/' <diff>
grep -E 'machine.?=.?["\x27][a-z]+["\x27]' <diff>
```

Any match is a candidate fleet-breaker. Use env vars and path-resolution fallbacks instead.

## 4. Concurrency audit

If adding a writer:
- Who else writes to this path? (daemon, capture, aggregator, training)
- Is the append format concurrency-safe? (gzip "at" is NOT; plain file append is, but hard to parse)
- Should this have its own subdir?

If touching a reader:
- Does it handle corrupt/truncated files gracefully? (BadGzipFile, EOFError)
- Does it recurse into subdirs? (needed if a writer later moves to a subdir)

## 5. Backward compatibility

Most fleet-wide changes should be additive:
- New schema version, not a rewrite
- New `subdir` param defaulting to None, not a required kwarg
- New env var with a sensible default, not a breaking env change

If breaking: is the migration path documented? Do other machines need to rebuild state? Is the fleet-ping clear about what to do?

## 6. The ping decision

After your commit lands, do you need a fleet-ping?

- **Yes, ping** — if machines need to take action (re-capture, re-deploy, re-verify)
- **Yes, ping** — if this is a reframe trigger (phase 1 approach needs revision)
- **Yes, ping** — if this is load-bearing divergence (a finding that challenges a plan)
- **No ping** — for routine maintenance, additive features that auto-apply on pull, test-only changes

Ping path: `shared-context/arc-agi-3/phase2/brain-arch/fleet-ping-{date}-{topic}.md` with clear required-action + expected-verification + severity.

## 7. The convergence row

If the action produces a measurable training/evaluation result:
- Append one line to `shared-context/arc-agi-3/phase2/brain-arch/phase-1-convergence.jsonl`
- Include: machine, date, head/kind, verdict, metrics, reframe_trigger (or null), commit hash, notes
- Row-append is conflict-free across machines — never edit existing rows, only add new ones

## 8. Commit + push

Follow `commit.md`. Do NOT leave a fleet-affecting commit unpushed overnight.

## 9. Post-ship watch

For the first 1-2 hours after a fleet-wide change:
- Pull periodically to see if any machine hit an issue
- Be available to respond to pings about your change
- If a machine's autonomous session hit your change and failed, file the fix immediately

---

## Reframe trigger — when to NOT ship this action

- You're not sure the frame is right (go back to `wake.md`)
- You'd be force-pushing or discarding other machines' work
- The test suite is not passing
- This is the third iteration of the same fix — something structural is wrong
- The user didn't ask for this scope
