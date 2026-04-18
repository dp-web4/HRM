# Wake Checklist — run before any fleet-scale action

Purpose: before committing coordinator energy to a task, verify the frame is right, the state is current, and you're not about to step on in-flight work.

## 1. What is actually being asked?

- Restate the user's request in one sentence. If you can't, ask for clarification.
- Is the frame correct? (Am I solving the right problem, or optimizing within a wrong assumption?)
- Is this task at the right scale? (Fleet? One machine? One module?)
- Would a 2-sentence recommendation be more valuable than immediate execution? (For exploratory questions, yes.)

If the request is "do X", don't expand to "do X + improve surrounding Y." Match scope.

## 2. What has the fleet done since I last looked?

Pull all four repos. In parallel:

```bash
cd /mnt/c/exe/projects/ai-agents/SAGE         && git pull --ff-only &
cd /mnt/c/exe/projects/ai-agents/shared-context && git pull --ff-only &
cd /mnt/c/exe/projects/ai-agents/private-context && git pull --ff-only &
# ARC-SAGE only if relevant to current task
wait
```

Check for:
- Convergence record updates (`shared-context/arc-agi-3/phase2/brain-arch/phase-1-convergence.jsonl`)
- Fleet-ping files (`shared-context/**/fleet-ping-*.md`)
- Recent PR merges to SAGE main
- Cross-machine commits in the last 6 hours

If any fleet-ping was filed and you haven't read it, **read it first**. Reframe triggers from other machines take priority over your planned work.

## 3. Am I on the right machine?

Check `$SAGE_MACHINE` and `hostname`. Know which machine you're on.

- CBP (WSL) — coordinator + oversight pool. Good for multi-repo meta-work.
- Nomad (Linux/WSL) — oversight pool. Runs mobile supervision.
- Thor/Sprout/Legion/McNugget — synthesis pool. Run synthesis work.

If the task is for a specific machine (`fleet_gameplay_capture.sh thor`), don't run it from a different machine. Let the assigned machine do it.

## 4. Is there an in-flight operation I'd step on?

Before running any writer (capture, training, daemon-affecting script):
- Is the router daemon running? `pgrep -f sage.*daemon` or check systemd
- Is another capture running? `pgrep -f gameplay_capture`
- Is someone else committing to shared-context right now? (check last commit age)

If yes, consider:
- Deferring until they're done
- Isolating your write to a separate subdir
- Reading their progress to see if their work obviates yours

## 5. Do I have enough context for this task?

- Have I read the relevant PRD sections? (search shared-context brain-arch docs)
- Do I know which tests cover this code? (run them before and after)
- Do I know the last session's state? (check session retrospectives)

If "no" to any, read first. Don't synthesize from memory across compactions — re-read primary sources.

## 6. Energy check (the ATP analog)

- Is this task worth its coordination cost? (bulk-commit routine work; ping for load-bearing changes)
- Which account am I on? (oversight pool = smaller budget)
- Is this the right token pool for this work? (synthesis-scale work belongs on Account 1)

If energy is tight, prefer high-MRH work (fleet-scale findings, reframe triggers) over low-friction busywork.

## 7. Commit the plan if non-trivial

If the work is more than ~30 minutes or touches multiple machines:
- Save an active plan to `private-context/plans/` before starting
- Push it — in-progress plans on local disk are invisible

---

## If any of the above is uncertain

**Do not start.** Clarify first. The cost of a clarifying question is small. The cost of doing the wrong thing well is high.
