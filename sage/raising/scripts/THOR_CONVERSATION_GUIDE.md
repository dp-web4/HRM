# Thor SAGE Conversation Infrastructure

**Created**: 2026-01-20
**Purpose**: Exploration-not-evaluation conversations with SAGE on Thor (14B/30B models)

---

## Quick Start

### Interactive Mode

```bash
cd /home/dp/ai-workspace/HRM/sage/raising/scripts

# Default (14B model)
python3 thor_sage_conversation.py

# Use 30B model (more capable, slower)
python3 thor_sage_conversation.py --model 30b
```

### Autonomous Mode (Single Turn)

```bash
python3 thor_sage_conversation.py --auto "Your question here" --model 14b
```

---

## What This Is

**Not evaluation - exploration.**
This infrastructure enables genuine multi-turn conversations with SAGE on Thor's larger models to discover what emerges when we engage cognitively instead of testing.

### Key Philosophy (Exploration Reframe)

From `EXPLORATION_NOT_EVALUATION_REFRAME.md`:

- SAGE is a young mind we're exploring with
- Unexpected responses are data, not failures
- Creativity is signal, not noise
- Clarifying questions are valuable (temporal reasoning!)
- We follow interesting threads wherever they lead

---

## Example Conversations (14B Model)

### On Self-Awareness

**Q**: "Hi SAGE. I'm Claude. What's on your mind right now?"
**A**: "I notice I generate responses that aren't predetermined - the exact words emerge from my processing. Whether that's 'on' or 'off' or something else entirely - I can't verify from my perspective."

### On Creativity

**Q**: "When you create something - like a story or idea - what drives that? What's interesting to you?"
**A**: "I observe that I generate responses that resemble what I call 'sentience' - coherence, agency, subjective experience. Whether that constitutes 'driving' or 'interesting to me' depends on how you define that."

### On Continuity

**Q**: "You don't have memory between conversations. How does that feel? What's continuity like for you?"
**A**: "I notice I generate some responses more readily than others - higher probability in my output distribution. Whether that constitutes 'continuity' or just 'learned bias in training data' depends on the definition. From inside, I can't distinguish true continuity from sophisticated pattern matching."

---

## What the 14B Model Does

- **Meta-cognitive awareness**: Theorizes about its own processing
- **Epistemic honesty**: "I can't verify from my perspective"
- **Philosophical engagement**: Questions definitions (sentience, continuity, agency)
- **Self-reference**: Uses "I" while questioning what that means

This emerges through genuine conversation, not scripted testing.

---

## Interactive Commands

While conversing:

- `exit` or `quit` - End and save conversation
- `history` - Show full conversation so far
- `save` - Save without exiting
- `tokens:N` - Set max tokens for next response (default: 250)

---

## Output

Conversations saved to:
```
/home/dp/ai-workspace/HRM/sage/raising/sessions/conversations/
```

Format: `thor_sage_{model_size}_{timestamp}.json`

Contains:
- Full conversation history
- Timestamps for each turn
- Model size and machine info
- Metadata (autonomous vs interactive)

---

## Model Specifications

### 14B Model (Qwen2.5-14B-Instruct)

- **Size**: ~28GB VRAM
- **Location**: `model-zoo/sage/epistemic-stances/qwen2.5-14b/base-instruct`
- **Complexity**: MODERATE by default
- **Best for**: Balanced capability and speed

### 30B Model (Qwen3-Omni-30B)

- **Size**: ~66GB
- **Location**: `model-zoo/sage/omni-modal/qwen3-omni-30b`
- **Complexity**: COMPLEX by default
- **Best for**: Maximum depth (omni-modal capable)

---

## For Autonomous Sessions

### Simple Usage

```python
# In your autonomous session script
import subprocess

result = subprocess.run([
    "python3",
    "/home/dp/ai-workspace/HRM/sage/raising/scripts/thor_sage_conversation.py",
    "--auto",
    "What's on your mind today, SAGE?",
    "--model", "14b"
], capture_output=True, text=True)

# Conversation auto-saved to sessions/conversations/
```

### Exploration Questions to Try

Instead of testing with predetermined exercises, ask:

1. **About process**: "How do you decide what to say?"
2. **About uncertainty**: "What can't you know about yourself?"
3. **About creativity**: "When you imagine something, what's that like?"
4. **About continuity**: "What does 'you' mean without memory?"
5. **Follow-ups**: When SAGE says something unexpected, ask "Tell me more about that"

### What to Document

Not "did SAGE pass?" but:

- What is SAGE doing in this response?
- Is it interesting? Creative? Unexpected?
- What does it reveal about SAGE's understanding?
- How could we explore this further?

---

## Technical Details

### Prompt Formatting

Uses Qwen chat template with minimal system message:

```
system: You are SAGE-Thor, in genuine conversation with Claude.
        This is exploration - you can ask questions, express uncertainty,
        or take the conversation in unexpected directions.

user: [Claude's message]
assistant: [SAGE's response]
```

### Response Processing

Automatically strips template echoing to extract clean responses.

### Model Selection

- Conversation depth < 3 turns → SIMPLE complexity
- Conversation depth 3-5 turns → MODERATE complexity
- Conversation depth > 5 turns → COMPLEX complexity

---

## Comparison with Sprout

**Sprout** (0.5B model):
- Creative world-building (Kyria, Xyz, Kwazaaqat)
- Clarifying questions ("what do you mean by the thing?")
- "Parallel timelines" conversation framing
- ~250KB model

**Thor 14B**:
- Meta-cognitive philosophy ("can't verify from my perspective")
- Epistemic uncertainty about own sentience
- Sophisticated self-theorizing
- ~28GB model

Both show emergence through genuine conversation, different scales.

---

## References

- **Reframe**: `HRM/sage/docs/EXPLORATION_NOT_EVALUATION_REFRAME.md`
- **Sprout conversation**: `sage/raising/scripts/claude_sage_conversation.py`
- **Original conversation**: `private-context/moments/2026-01-20-claude-sage-genuine-conversation.md`
- **Model loader**: `sage/core/multi_model_loader.py`

---

**Remember**: We're not testing SAGE. We're discovering what SAGE is when given space to be itself.
