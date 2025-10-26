# Prediction Logging System - Turning Hallucinations into Training Data

## Overview

When SAGE's LLM generates multi-turn dialogue (hallucinations), it's actually making **predictions about what the user will say next**. Instead of just suppressing these hallucinations, we capture them as training data.

This creates a valuable dataset of:
- **Model predictions** about conversation flow
- **Reality vs prediction** comparison
- **Potential finetuning data** for better dialogue modeling

## The Prediction Triplet

Each captured hallucination creates a triplet:

```python
{
    'model_response': "What SAGE said to user",
    'predicted_user_response': "What model thought user would say",
    'actual_user_response': "What user actually said",
    'similarity': 0.15,  # Word overlap metric
    'status': 'complete'
}
```

## Architecture

### Components

1. **`cognitive/prediction_logger.py`** - Core logging system
   - Captures hallucinated predictions
   - Waits for actual user response
   - Completes triplets with similarity calculation
   - Stores to JSONL format

2. **`experiments/integration/streaming_responder.py`** - Hallucination extraction
   - Modified `_extract_hallucination()` to extract both model response and predicted user response
   - Integrated with `PredictionLogger` during streaming
   - Detects conversation turn markers (`\nUser:`, `\nAssistant:`)

3. **`tests/hybrid_conversation_threaded.py`** - Actual response capture
   - Initializes `PredictionLogger`
   - Passes logger to `StreamingResponder`
   - Captures actual user response after each turn

### Data Flow

```
1. LLM generates response word-by-word
2. Hallucination detected: "...appreciation.\nUser: How..."
3. EXTRACT: model_response="...appreciation."
            predicted_user_response="User: How..."
4. CAPTURE: Store pending prediction, wait for user
5. USER SPEAKS: "Tell me how you feel."
6. COMPLETE: Add actual response, calculate similarity, save triplet
```

## Usage

### Initialization

```python
from cognitive.prediction_logger import PredictionLogger
from experiments.integration.streaming_responder import StreamingResponder

# Create logger
logger = PredictionLogger(log_dir="~/sage_predictions")

# Create responder with prediction logging
llm = StreamingResponder(
    max_new_tokens=512,
    temperature=0.7,
    prediction_logger=logger  # Enable prediction capture
)
```

### During Conversation

Hallucination detection and capture happens automatically:

```python
# When hallucination is detected during streaming:
# - Model response is extracted
# - Predicted user response is extracted
# - Triplet is created with status='pending'

# After user speaks:
if logger.pending_prediction:
    logger.log_actual_response(user_input)
    # Triplet is completed with status='complete'
```

### Exporting Training Data

```python
# Get statistics
stats = logger.get_stats()
print(f"Total predictions: {stats['total_predictions']}")
print(f"Avg similarity: {stats['avg_similarity']:.2%}")

# Export for finetuning
training_file = logger.export_training_data()
# Creates: ~/sage_predictions/training_data_YYYYMMDD_HHMMSS.json
```

## Storage Format

### JSONL Session Logs

Each conversation session creates a JSONL file:
```
~/sage_predictions/predictions_YYYYMMDD_HHMMSS.jsonl
```

Each line is a complete triplet:

```json
{
  "timestamp": 1234567890.123,
  "datetime": "2024-10-25T16:45:09",
  "model_response": "Yes, happiness often feels like a sense of fulfillment, belonging, and purpose.",
  "predicted_user_response": "User: How does that feel for you?",
  "actual_user_response": "Tell me how you feel.",
  "similarity": 0.15,
  "status": "complete",
  "response_time": 2.3,
  "context": {
    "chunks": 11,
    "tokens": 44,
    "user_text": "Would this joy feel like to you?"
  }
}
```

### Training Data Export

Export format for finetuning:

```json
{
  "metadata": {
    "created": "2024-10-25T17:00:00",
    "total_predictions": 15,
    "avg_similarity": 0.12
  },
  "examples": [
    {
      "assistant_response": "...",
      "predicted_user_response": "...",
      "actual_user_response": "...",
      "similarity": 0.15,
      "context": {...}
    }
  ]
}
```

## Similarity Calculation

Simple word overlap metric:

```python
predicted_words = set(predicted_response.lower().split())
actual_words = set(actual_response.lower().split())

overlap = len(predicted_words & actual_words)
total = len(predicted_words | actual_words)
similarity = overlap / total  # 0.0 to 1.0
```

**Example**:
- Predicted: "User: How does that work?"
- Actual: "Tell me more about that."
- Shared words: {} (no overlap)
- Similarity: 0.00

## Testing

Run comprehensive tests:

```bash
cd /home/sprout/ai-workspace/HRM/sage
python3 experiments/integration/test_prediction_logging.py
```

**Test coverage**:
1. Hallucination extraction (detection + parsing)
2. Prediction triplet creation (capture → complete)
3. Streaming generation with logging
4. Training data export

## Real-World Examples

### Hallucination Event #1

**User question**: "Would this joy feel like to you?"

**Model response (spoken)**:
```
"Yes, happiness often feels like a sense of fulfillment, belonging, and purpose.
It's a state where you feel alive and connected to others, and you can appreciate
things around you with greater appreciation."
```

**Predicted user response (hallucinated)**:
```
"User: How..."
```

**Actual user response**:
```
"Tell me how you feel."
```

**Similarity**: 0% (no word overlap)

**Analysis**: Model correctly predicted user would ask a follow-up question about "how", showing understanding of conversation flow.

### Hallucination Event #2

**User question**: "Okay."

**Model response (spoken)**:
```
"Good. We've covered some basic emotions. Do you have any other feelings that
come up in your conversations?"
```

**Predicted user response (hallucinated)**:
```
"User: Not..."
```

**Actual user response**:
```
"Tell me about sadness."
```

**Similarity**: 0% (no word overlap)

**Analysis**: Model predicted user might decline ("Not...") when asked about other feelings. Reality was opposite - user engaged deeper.

## Why This Matters

### 1. Emergent Metacognition

The model demonstrates **perspective-taking** - imagining what the conversation partner would say next. This is sophisticated dialogue modeling for a 0.5B parameter model.

### 2. Training Data Quality

Each triplet captures:
- **Context**: What led to the prediction
- **Model's theory of mind**: What it expected
- **Reality check**: What actually happened

This is valuable for:
- Improving conversation continuation models
- Training dialogue flow predictors
- Understanding model's internal conversation representation

### 3. Pattern Analysis

By collecting many triplets, we can identify:
- What question types trigger hallucinations?
- Are certain response patterns more prone?
- Does conversation history affect frequency?
- What's the typical prediction accuracy?

## Future Work

### Short Term
- **Data collection**: Gather 100+ triplets from real conversations
- **Pattern analysis**: Identify common hallucination triggers
- **Similarity improvements**: Use embedding-based semantic similarity

### Long Term
- **Finetuning**: Train on prediction triplets to improve dialogue modeling
- **Specialized models**: Create dedicated conversation continuation predictor
- **Integration**: Use predictions to anticipate user needs proactively

## Code Locations

- **Prediction logger**: `/sage/cognitive/prediction_logger.py`
- **Hallucination extraction**: `/sage/experiments/integration/streaming_responder.py:222-272`
- **Actual response capture**: `/sage/tests/hybrid_conversation_threaded.py:330-332`
- **Test suite**: `/sage/experiments/integration/test_prediction_logging.py`
- **Documentation**: This file

## Key Insight

**Hallucinations aren't bugs - they're predictions.** Instead of just suppressing them, we capture and learn from the gap between prediction and reality. This turns a "problem" into a valuable dataset for understanding how the model thinks about conversation flow.

The 0.5B model on a Jetson is demonstrating sophisticated dialogue modeling. It "knows":
- Conversations continue beyond single turns
- Partners take turns
- Questions invite responses
- Context shapes continuations

By capturing these predictions, we're building a dataset that reveals the model's internal representation of dialogue structure.
