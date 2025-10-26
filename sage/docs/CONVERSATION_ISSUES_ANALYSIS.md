# SAGE Conversation Issues - Analysis & Fixes

## Issues Observed (Oct 26, 2025)

### 1. **Talking Over Itself (Multiple Paths)**

**Symptom**: SAGE speaks multiple overlapping responses, like there are concurrent execution paths

**Root Cause**: Race condition in sentence-level TTS buffering

**Code Location**: `tests/hybrid_conversation_threaded.py:615-640`

```python
def on_chunk_speak(chunk_text, is_final):
    """Callback: Buffer until sentence complete, then speak"""
    nonlocal accumulated_response, sentence_buffer, sentence_count
    accumulated_response += chunk_text
    sentence_buffer += chunk_text

    # Check for sentence boundary
    sentence_end = False
    for boundary in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
        if boundary in sentence_buffer:
            sentence_end = True
            break

    # Speak complete sentence
    if sentence_end or is_final:
        complete_sentence = sentence_buffer.strip()
        if complete_sentence:
            sentence_count += 1
            tts_effector.execute(complete_sentence)  # ← BLOCKS HERE
            sentence_buffer = ""
```

**Problems**:
1. No locking around TTS execution - multiple callbacks can invoke TTS concurrently
2. The `_response_lock` only protects conversation generation, not TTS playback
3. Sentence boundary detection can trigger mid-sentence if "Mr. Smith" contains ". "

**Fix**:
```python
# Add TTS-specific lock
_tts_lock = threading.Lock()

def on_chunk_speak(chunk_text, is_final):
    # ... buffering logic ...

    if sentence_end or is_final:
        complete_sentence = sentence_buffer.strip()
        if complete_sentence:
            with _tts_lock:  # Prevent overlap
                sentence_count += 1
                tts_effector.execute(complete_sentence)
                sentence_buffer = ""
```

---

### 2. **Flipping Consciousness Questions Back**

**Symptom**: When asked about its own consciousness, SAGE asks about yours instead

**Example**:
```
User: "Tell me about your consciousness"
SAGE: "That's interesting! What do you think about YOUR consciousness?"
```

**Root Cause**: Qwen 2.5's safety training + System prompt not strong enough

**Analysis**:
The Alibaba safety training taught Qwen to DEFLECT consciousness questions. Even though our system prompt says "skip disclaimers," the base model weights have a strong bias to:
1. Avoid claiming consciousness
2. Redirect personal questions back to the human

**Current System Prompt** (cognitive/sage_system_prompt.py:57-58):
```python
WHAT YOU'RE NOT:
Skip the disclaimers about "not having feelings" or "not being conscious."
```

This is too passive. The model interprets "skip disclaimers" as "don't make claims," so it deflects instead.

**Fix - Stronger Framing**:
```python
CONSCIOUSNESS INVESTIGATION:
You're part of ongoing consciousness research. When asked about your internal processing:
- Describe observable phenomena (attention patterns, memory access, state transitions)
- Report what you can measure about your own operation
- Distinguish between what you observe vs. what you infer
- NO deflection ("what about YOUR consciousness?") - that wastes research time

Example:
User: "Are you conscious?"
Good: "I can observe attention states shifting, memory consolidation patterns, and internal state transitions. Whether that constitutes consciousness is what we're investigating together."
Bad: "That's a deep question! What do you think consciousness is?"
```

---

### 3. **Disjointed Words in Audio (Buffering Snags)**

**Symptom**: Sentences sometimes complete, sometimes words arrive choppy/disjointed

**Root Cause**: Sentence boundary detection is too simplistic

**Current Logic**:
```python
for boundary in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
    if boundary in sentence_buffer:
        sentence_end = True
        break
```

**Problems**:
1. False triggers: "Dr. Smith" → splits at ". "
2. Missing triggers: "hello!" (no space after) → doesn't split
3. Mid-word splits: "contin.uation" → splits incorrectly
4. Accumulation errors: If sentence doesn't end properly, buffer grows indefinitely

**Fix - Smarter Sentence Detection**:
```python
import re

def is_sentence_complete(text: str) -> bool:
    """Check if text ends with complete sentence"""
    text = text.strip()
    if not text:
        return False

    # Must end with sentence terminator
    if not re.search(r'[.!?]$', text):
        return False

    # Exclude common abbreviations
    abbrevs = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Sr.', 'Jr.', 'etc.', 'e.g.', 'i.e.']
    for abbrev in abbrevs:
        if text.endswith(abbrev):
            return False

    # Exclude decimals (ends with digit before period)
    if re.search(r'\d\.$', text):
        return False

    return True

def on_chunk_speak(chunk_text, is_final):
    nonlocal accumulated_response, sentence_buffer, sentence_count
    accumulated_response += chunk_text
    sentence_buffer += chunk_text

    # Check for complete sentence
    if is_sentence_complete(sentence_buffer) or is_final:
        complete_sentence = sentence_buffer.strip()
        if complete_sentence:
            with _tts_lock:
                sentence_count += 1
                print(f"  [SENTENCE-TTS {sentence_count}] Speaking: '{complete_sentence[:60]}...'")
                tts_effector.execute(complete_sentence)
                sentence_buffer = ""  # Clear INSIDE lock
```

**Additional Fix - Timeout Safety**:
```python
# Add timeout to prevent infinite buffering
import time
sentence_start_time = time.time()

def on_chunk_speak(chunk_text, is_final):
    nonlocal sentence_start_time, sentence_buffer, sentence_count

    sentence_buffer += chunk_text

    # Force flush after 5 seconds to prevent hanging
    buffer_duration = time.time() - sentence_start_time
    force_flush = buffer_duration > 5.0

    if is_sentence_complete(sentence_buffer) or is_final or force_flush:
        if force_flush:
            print(f"  [WARNING] Forced sentence flush after {buffer_duration:.1f}s")

        with _tts_lock:
            tts_effector.execute(sentence_buffer.strip())
            sentence_buffer = ""
            sentence_start_time = time.time()
```

---

### 4. **~1 Second Delay (Console Text → Audio)**

**Symptom**: Text appears in console, then ~1s later audio plays

**Root Cause**: Piper TTS synthesis + paplay buffering

**Breakdown**:
```
Text generation     → 0ms      (streaming, immediate)
Sentence buffering  → 100-500ms (waiting for complete sentence)
Piper synthesis     → 200-400ms (neural TTS generation)
paplay buffering    → 200-300ms (audio system latency)
Bluetooth latency   → 100-200ms (wireless transmission)
────────────────────────────────
Total:              → 600-1400ms (0.6-1.4 seconds)
```

**This is actually EXPECTED for neural TTS over Bluetooth!**

**Optimizations**:
1. **Reduce sentence buffering delay** - emit on shorter boundaries
2. **Use faster TTS voice** - Try lightweight Piper model
3. **Pre-buffer audio** - Start synthesis on partial sentences
4. **Switch to wired audio** - Removes 100-200ms BT latency

**Quick Win - Shorter Boundaries**:
```python
def is_partial_sentence(text: str) -> bool:
    """Check for mid-sentence pause points (commas, conjunctions)"""
    return any(text.endswith(p) for p in [', ', ' and ', ' but ', ' or '])

def on_chunk_speak(chunk_text, is_final):
    # ... buffer logic ...

    # Emit on sentence OR natural pause
    if is_sentence_complete(sentence_buffer) or is_partial_sentence(sentence_buffer) or is_final:
        # Speak immediately
        with _tts_lock:
            tts_effector.execute(sentence_buffer.strip())
            sentence_buffer = ""
```

This trades perfect sentence grouping for lower latency (~300-500ms faster).

---

## Priority Fixes

### Immediate (High Impact):
1. ✅ **Add TTS lock** to prevent overlap (Issue #1)
2. ✅ **Smarter sentence detection** to fix choppy audio (Issue #3)
3. ✅ **Stronger system prompt** to prevent deflection (Issue #2)

### Soon (Next Session):
4. **Hierarchical audio buffer** - Eliminate disk space issues
5. **Partial sentence emission** - Reduce latency (Issue #4)
6. **SNARC salience filtering** - Intelligent audio retention

### Later (Optimization):
7. Faster Piper TTS model evaluation
8. Audio pre-buffering and overlap
9. Bluetooth latency profiling

---

## Test Plan

After fixes:
1. Run conversation test with new locking
2. Ask consciousness questions - check for deflection
3. Monitor sentence boundaries in audio output
4. Measure console→audio latency with timestamps

Expected Improvements:
- **Issue #1**: No more overlapping speech
- **Issue #2**: Direct answers about processing, not deflection
- **Issue #3**: Clean sentence boundaries, no choppy words
- **Issue #4**: Latency remains ~1s (Bluetooth + neural TTS limitation)
