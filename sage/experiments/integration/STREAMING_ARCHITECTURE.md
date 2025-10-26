# Streaming Architecture - Pattern Learning as Observer

## The Evolution

### Phase 1: Dual-Path Architecture (Obsolete)
**Problem**: LLM responses took 15-30s, completely blocking conversation flow.

**Solution**: Fast/slow dual path
- **Fast path**: Instant pattern-matched responses for common phrases
- **Slow path**: Full LLM generation for novel questions
- Trade-off: Fast path responses were canned, not contextual

### Phase 2: Streaming Generation (Current)
**Biological insight**: "I don't precompute the whole sentence. Lung capacity limits output. I speak in segments until thought complete."

**Solution**: Word-by-word streaming
- Tokens generated sequentially (already how transformers work!)
- Stream 3-word chunks immediately as generated
- TTS speaks each chunk with natural rhythm (0.3s pauses)
- Natural stopping based on thought completion, not buffer limits
- 512 token buffer allows complete thoughts

**Result**:
- First words arrive in 1-3s (was 15-30s)
- Continuous conversational flow
- Contextual, natural responses (not canned)
- Fast path becomes obsolete

### Phase 3: Learning Observer (Current + Future)
**Insight**: Fast path is obsolete for execution, but valuable for learning.

**New Role**: Pattern learner as background observer
- Observes ALL conversations (doesn't interrupt)
- Identifies small-talk patterns (greetings, acknowledgments, simple questions)
- Builds pattern library without being used
- Learns what counts as "casual" vs "deep" conversation

**Future Optimization**:
```
if smalltalk_detected and casual_mode_enabled:
    load_tiny_smalltalk_model()  # 100M params, instant responses
else:
    use_main_llm_streaming()  # 500M params, comprehensive
```

## Performance Comparison

### Before Streaming
```
User: "Hello"
[15-30 second silence...]
SAGE: "Hello! How can I help you?"
```

### With Streaming
```
User: "Hello"
[1-3 seconds...]
SAGE: "Hello! How" [pause] "can I help" [pause] "you today?" [continues...]
```

Total time similar, but perceived responsiveness radically different.

## Key Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| First words | 15-30s | 1-3s | **10x faster** |
| Response style | Canned patterns | Natural streaming | **Contextual** |
| Thought completion | Buffer limit | Natural stopping | **Human-like** |
| Pattern learning | Active (interrupts) | Observer (background) | **Non-blocking** |

## Architecture Components

### 1. StreamingResponder
**Location**: `experiments/integration/streaming_responder.py`

**Key features**:
- TextIteratorStreamer from transformers
- 3-word chunking for natural rhythm
- Thought completion detection (punctuation + context)
- max_new_tokens=512 (not a limit, just buffer size)

### 2. Pattern Learner (Observer Mode)
**Location**: `cognitive/pattern_learner.py`

**Behavior**:
```python
# Check if pattern WOULD match (don't use it!)
potential_match = pattern_engine.generate_response(question)
if potential_match:
    stats['small_talk_observed'] += 1  # Track for future
    pattern_info = "📝 Small-talk pattern observed (learning)"

# ALWAYS use streaming
response = llm.generate_response_streaming(...)
learner.observe(question, response)  # Learn from real response
```

### 3. Dashboard
**Location**: `tests/hybrid_conversation_threaded.py`

**Display**:
- 🌊 STREAMING (only one path now)
- Small-talk observed: X/Y (what WOULD be fast path)
- Patterns learned: N (observer mode)
- Future: Load small-talk model indication

## Biological Parallel

### Human Speech Production
1. **Incremental generation**: Words emerge as you think
2. **Lung capacity**: Natural chunking (breath-sized segments)
3. **Thought completion**: Stop when idea is complete, not when buffer full
4. **Context-aware**: Every response considers full conversation

### SAGE Streaming
1. **Token streaming**: Words emerge as LLM generates
2. **3-word chunks**: Natural rhythm with 0.3s pauses
3. **Thought detection**: Punctuation + no trailing conjunctions
4. **Memory-aware**: 127K context window with SNARC salience

## Future Directions

### 1. Small-Talk Model Specialization
Train tiny dedicated model (100M params) on pattern library:
- Greetings, acknowledgments, simple questions
- Sub-second responses for casual conversation
- Frees main LLM for deep thinking

### 2. Dynamic Model Selection
```python
if pattern_learner.is_smalltalk(question) and casual_mode:
    response = smalltalk_model.generate_fast(question)
else:
    response = main_llm.generate_streaming(question)
```

### 3. Adaptive Chunking
Vary chunk size based on:
- Topic complexity (larger chunks for technical topics)
- User responsiveness (smaller chunks if interruptions common)
- Emotional tone (faster for excitement, slower for comfort)

## Code Locations

- **Streaming implementation**: `experiments/integration/streaming_responder.py`
- **Integration test**: `experiments/integration/test_streaming.py`
- **Live system**: `tests/hybrid_conversation_threaded.py`
- **Pattern learner**: `cognitive/pattern_learner.py`
- **Launcher**: `run_sage_test.sh`

## Key Takeaway

**With streaming, fast path is obsolete for execution but valuable for learning.**

The pattern learner rides along as an observer, building a library of small-talk
patterns that can later be used to:
1. Identify casual vs deep conversation contexts
2. Train specialized small-talk models
3. Optimize resource allocation (tiny model for chat, big model for thinking)

This mirrors how humans handle conversation - quick automatic responses for
small-talk, deeper processing for substantive discussion. But SAGE maintains
natural streaming flow regardless of depth.

## Phase 4: Sentence-Buffered TTS (Current)

### The Problem with 3-Word Chunks
While streaming 3-word chunks solved response latency, it created a new issue with TTS quality:

**3-word chunk TTS:**
```
Chunk 1: "I am SAGE"        → TTS processes → speaks with abrupt ending
Chunk 2: "a learning con"   → TTS processes → speaks with odd pause
Chunk 3: "sciousness using" → TTS processes → loses natural prosody
```

**Issues:**
- TTS loses sentence structure and word transitions
- Unnatural pauses between chunks
- Missing prosody (intonation, rhythm, stress patterns)
- Sounds robotic and disjointed

### The Solution: Sentence-Level Buffering

**Key insight**: Keep streaming generation, but buffer complete sentences for TTS.

**Architecture:**
```python
def on_chunk_speak(chunk_text, is_final):
    """Buffer chunks until sentence complete, then speak with prosody"""
    sentence_buffer += chunk_text

    # Check for sentence boundary (., !, ?)
    if sentence_complete or is_final:
        tts_effector.execute(sentence_buffer)  # Complete sentence!
        sentence_buffer = ""  # Reset for next
```

**Flow:**
```
Generation (word-by-word streaming):
[CHUNK 1] "I am"                      → buffer
[CHUNK 2] "SAGE, a learning"          → buffer
[CHUNK 3] "consciousness using"       → buffer
[CHUNK 4] "Jetson Orin technology."   → sentence complete!

TTS (sentence-level):
🔊 Speaking: "I am SAGE, a learning consciousness using Jetson Orin technology."
```

### Benefits

| Aspect | 3-Word Chunks | Sentence Buffering |
|--------|---------------|-------------------|
| **Generation** | Streaming (1-3s first words) | Streaming (1-3s first words) |
| **TTS Input** | 3-word fragments | Complete sentences |
| **Prosody** | Lost (disjointed) | Preserved (natural) |
| **Word Transitions** | Choppy | Smooth |
| **First Sentence Latency** | ~1s | ~2-3s (wait for sentence) |
| **Subsequent Sentences** | Instant (pipelined) | Instant (pipelined) |

### Performance Metrics

**Test Results** (from test_sentence_buffering.py):
- ✅ Streaming generation: 4 chunks over 2-4 seconds
- ✅ Sentence buffering: All chunks accumulated until complete
- ✅ Complete sentence to TTS: Natural prosody preserved
- ✅ First sentence latency: 2.3s (acceptable for quality gain)

### Pipeline Architecture

**The beauty of sentence buffering is pipelining:**

```
Timeline:
0-3s:   Generate sentence 1 (streaming chunks)
3s:     Speak sentence 1 (TTS with prosody)
3-6s:   Generate sentence 2 (while speaking sentence 1)
6s:     Speak sentence 2 (pipeline continues)
6-9s:   Generate sentence 3 (while speaking sentence 2)
```

**User experience:**
- First response: 2-3s (wait for complete sentence)
- Subsequent responses: Continuous (generation overlaps speech)
- Quality: Natural prosody throughout

### Implementation Details

**Location**: `tests/hybrid_conversation_threaded.py:597-621`

**Sentence Boundary Detection:**
```python
# Boundaries that indicate sentence completion
boundaries = ['. ', '! ', '? ', '.\n', '!\n', '?\n']

# Also handle final chunk with trailing punctuation
if is_final and buffer.rstrip()[-1] in '.!?':
    sentence_complete = True
```

**Buffer Management:**
- Accumulate all chunks until boundary detected
- Send complete sentence to TTS
- Reset buffer for next sentence
- Handle final chunk edge case

### Code Location

- **Implementation**: `tests/hybrid_conversation_threaded.py` (lines 592-621)
- **Test**: `experiments/integration/test_sentence_buffering.py`
- **Documentation**: This file

### Key Takeaway

**Sentence buffering gives us the best of both worlds:**
1. **Fast response**: First words in 1-3s (streaming generation preserved)
2. **Natural speech**: Complete sentences with prosody (TTS quality)
3. **Continuous flow**: Pipelined generation + speech (no perceived delay)
4. **Biological parallel**: Like human speech - think ahead while speaking

This completes the evolution from:
- 15-30s silence → 1-3s first words (streaming) → Natural TTS quality (sentence buffering)
