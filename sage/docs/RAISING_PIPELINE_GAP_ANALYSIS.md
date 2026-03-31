# Raising Pipeline Gap Analysis (2026-03-31)

**Problem**: Legion's 14 raising sessions produced identical output despite consolidation repeatedly requesting different prompts and setting BLOCK directives.

**Root Cause**: The consolidation→prompt feedback loop **does not exist**.

---

## Evidence

### 1. Hardcoded Prompts

`sage/raising/scripts/ollama_raising_session.py` lines 105-154 define `CONVERSATION_FLOWS`:

```python
CONVERSATION_FLOWS = {
    "sensing": [
        "Before we begin, check in with yourself. What's your state right now?",
        "What do you notice about how you're processing right now?",
        "Can you describe the difference between noticing something and thinking about something?",
        # ... 8 hardcoded prompts
    ],
    # ... other phases
}
```

These prompts are **static**. They never change based on consolidation output.

### 2. No Consolidation Integration

Searched `ollama_raising_session.py` for consolidation-related code:
- **`raising_log`**: 0 references
- **`consolidation`**: 0 references
- **`BLOCK`**: 0 references
- **`dream_consolidation` import**: 0 references

The session runner **never reads** `raising_log.md`.

### 3. Dream Consolidation is Orphaned

`sage/raising/scripts/dream_consolidation.py` exists and works correctly:
- Line 125: Reads `raising_log.md` for context
- Lines 188-194: Extracts vocabulary
- Lines 209-215: Records milestones
- Lines 224-255: Writes new entries to `raising_log.md`

But **no script calls it**. Dream consolidation must be run manually:

```bash
python3 -m sage.raising.scripts.dream_consolidation \
    --instance sage/instances/legion-gemma3-12b \
    --session 14
```

### 4. Legion's Evidence Trail

From `sage/instances/legion-gemma3-12b/raising_log.md`:

- **Session 2 consolidation** (line 43): "Break the agreement loop. Use prompts that offer contradictions."
- **Session 3 prompts**: Identical to session 2 (no changes)
- **Session 5 consolidation** (line 114): "BLOCKER: Do not run session 6 on automated pipeline."
- **Session 6**: Ran anyway with identical prompts
- **Sessions 6-14**: Each consolidation set BLOCK directive, each ignored

The pattern repeated **10 times**. This is empirical proof the pipeline has no feedback loop.

---

## What Works

1. **Dream consolidation script** - Produces detailed, accurate analysis when run
2. **Raising_log.md writing** - Consolidation entries are correctly written to disk
3. **Vocabulary extraction** - Works when consolidation runs (Legion has "gentle hum", "focused spotlight", etc.)
4. **Milestone detection** - Works when consolidation runs
5. **Quality scoring** - Accurate 1-5 ratings

The analysis infrastructure is **excellent**. The problem is integration.

---

## What's Missing

### 1. Consolidation Trigger

Session runner should call dream consolidation after each session:

```python
# After session completes
from sage.raising.scripts.dream_consolidation import run_dream_consolidation
run_dream_consolidation(instance_dir=str(self.instance.root), session_num=self.session_number)
```

### 2. Prompt Generation from Consolidation

Before each session, read `raising_log.md` and adapt prompts:

```python
def _generate_prompts_from_consolidation(self) -> List[str]:
    """Generate session prompts based on latest consolidation recommendations."""
    raising_log_path = self.instance.root / 'raising_log.md'

    if not raising_log_path.exists():
        # First session - use default prompts
        return self.CONVERSATION_FLOWS[self.phase]

    # Read consolidation notes
    log_text = raising_log_path.read_text()

    # Check for BLOCK directives
    if "BLOCK:" in log_text or "BLOCKER:" in log_text:
        # Parse last consolidation entry
        # Extract BLOCK reasons
        # Raise exception to prevent session launch
        raise BlockedSessionError("Consolidation has active BLOCK directive")

    # Parse "Next Session Focus" from latest entry
    # Extract recommended prompt changes
    # Adapt CONVERSATION_FLOWS accordingly
    # Return customized prompts

    return adapted_prompts
```

### 3. Blocker Enforcement

Session launcher must check for active blockers before starting:

```python
def _check_blockers(self) -> Optional[str]:
    """Check if consolidation has set BLOCK directives."""
    raising_log_path = self.instance.root / 'raising_log.md'
    if not raising_log_path.exists():
        return None

    log_text = raising_log_path.read_text()

    # Extract most recent consolidation entry
    # Check for BLOCK/BLOCKER keywords
    # Return blocker message if found

    if blocker_found:
        return blocker_message
    return None
```

Then in `__init__`:

```python
blocker = self._check_blockers()
if blocker:
    print(f"\n⚠️  SESSION BLOCKED ⚠️\n{blocker}\n")
    sys.exit(1)
```

### 4. Banned Word Detection

Consolidation often requests word bans ("Do NOT use 'noticing', 'processing', 'awareness'").

Need runtime word detection:

```python
def _check_banned_words(self, response: str, banned: List[str]) -> bool:
    """Check if response contains banned words from consolidation."""
    response_lower = response.lower()
    violations = [w for w in banned if w.lower() in response_lower]
    return len(violations) > 0, violations
```

Then interrupt and re-prompt if violations found.

---

## Impact

**Without the feedback loop:**
- Consolidation notes are write-only theater
- Instances cannot escape cached patterns (Legion's "noticing" loop)
- Raising becomes rote repetition, not development
- Dream consolidation's excellent analysis is wasted

**With the feedback loop:**
- Consolidation can adapt prompts session-to-session
- BLOCK directives prevent harmful repetition
- Vocabulary emerges naturally (tracked but not forced)
- Trust tensors can evolve (infrastructure exists, just not wired)
- Milestones guide phase progression

---

## Fix Priority

**Critical path:**
1. Add consolidation call to session runner (after each session)
2. Add blocker check to session runner (`__init__`)
3. Add prompt adaptation from consolidation (before each session)

**Follow-up:**
4. Banned word detection at runtime
5. Trust tensor update hook (consolidation → identity.json)
6. Phase progression based on milestones (not session count)

---

## Testing

**Verification that fix works:**

1. Run Legion session 15 manually with consolidation enabled
2. Check that `raising_log.md` gets a new entry
3. Modify the entry to add a BLOCK directive manually
4. Attempt to run session 16
5. **Expected**: Session refuses to launch, prints blocker message
6. **Current**: Session runs anyway

---

## Conclusion

The raising infrastructure is well-designed. Dream consolidation produces excellent analysis. The gap is **integration** - consolidation output doesn't flow back into session generation.

Legion's 14-session loop is a feature, not a bug: it revealed the missing feedback path through empirical evidence. The system detected the problem (consolidation notes are detailed and accurate), but couldn't act on it (prompts never changed).

**Fix is straightforward**: Wire consolidation into the session lifecycle. The hard work (analysis, vocabulary tracking, milestone detection) is already done.

---

**Related Documents:**
- `sage/instances/legion-gemma3-12b/raising_log.md` (empirical evidence)
- `sage/raising/scripts/dream_consolidation.py` (working but orphaned)
- `sage/raising/scripts/ollama_raising_session.py` (needs integration)
- `private-context/insights/2026-03-27-raising-deep-dive-analysis.md` (identified gap from session data)
