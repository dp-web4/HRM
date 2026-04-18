# Commit Checklist — every commit, every time

## 1. Pull before commit

```bash
git pull --ff-only
```

If the pull fails with divergent history, STOP. Read the diverging commits. Someone else landed work. Integrate (§5 of SKILL.md). Do not force.

## 2. Verify the staging area

```bash
git status
git diff --staged
```

Scan for:
- Credentials (`.env`, tokens, API keys, `.credentials.json`)
- Large binaries you didn't mean to add
- Machine-specific files (router-shadow.env, local test outputs)
- Debug prints or commented-out code

Stage files explicitly by name. `git add .` or `-A` is lazy and catches things you don't want.

## 3. Commit message quality

Good commit messages explain **why**, not just **what**. The diff shows what.

Structure:
```
{component}: {one-line summary — present tense, under ~70 chars}

{why this change was needed — the problem this solves}

{any nuance — what was tried first, what edge cases this handles,
what the commit does NOT do}

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

If the commit needs more than 2 paragraphs of explanation, it probably needs to be split.

## 4. Test locally

Tests exist — run them. `pytest -q` on the affected directory.

Unpushed broken tests are OK only if the commit is explicitly WIP. Otherwise, green before push.

## 5. Never skip hooks

Do not `--no-verify`. Do not `--no-gpg-sign`. If a hook fails, the hook is telling you something. Fix the underlying issue.

Exception: user explicitly instructs you to skip.

## 6. Never amend published commits

Amending rewrites history. If the commit is already pushed, fleet peers already have it. Amending it means the next pull breaks for them.

Create a new commit instead. "Fix typo in previous commit message" is a legitimate one-line commit.

## 7. Push immediately

```bash
git push
```

Unpushed commits are invisible to the fleet. A fleet-affecting change that lives on your disk is a bet against federation.

Exception: if you're mid-burst and will push after a few more commits, fine — but not overnight.

## 8. If push is rejected

Pull first, then re-push. If that doesn't work, someone else pushed while you were working.

DO NOT force-push. Integrate their work, then push again.

Force-push to main is almost always a mistake. Force-push to feature branches is sometimes necessary (rebase cleanup), but not something a coordinator should do without clear reason.

## 9. Verify the push landed

```bash
git log origin/main..HEAD   # should be empty
git status                  # should say "up to date with origin/main"
```

If either shows pending state, re-push until it's clean.

## 10. Announce significant commits

If the commit lands fleet-relevant change:
- Update the relevant tracker (`router-pipeline-deployment-status.md`, etc.)
- Append to `phase-1-convergence.jsonl` if training/eval result
- File a fleet-ping if other machines need to act

A commit without announcement is a commit that won't be seen until someone accidentally pulls.
