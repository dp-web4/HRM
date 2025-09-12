# Proposed SAGE Implementation: Context-First Approach

*Date: September 12, 2025*  
*Concrete steps to implement H↔L Context↔Action architecture*

## Immediate Action Items

### 1. Create Context Encoder V2 (Today)
Build on V3's 16D classification with richer encoding:

```python
# sage/context/context_encoder_v2.py
import torch
import torch.nn as nn
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

class PatternType(Enum):
    EXTRACTION = "extraction"
    FILLING = "filling"
    SYMMETRY = "symmetry"
    TILING = "tiling"
    COLOR_MAP = "color_mapping"
    COUNTING = "counting"
    MOVEMENT = "movement"
    UNKNOWN = "unknown"

@dataclass
class PuzzleContext:
    # Core dimensions (from V3)
    pattern_type: PatternType
    size_relationship: str  # "same", "smaller", "larger", "tiled"
    input_colors: int
    output_colors: int
    spatial_density: float
    
    # Enhanced dimensions
    color_semantics: Dict[int, str]  # {0: "background", 3: "border", ...}
    object_count: int
    symmetry_axes: List[str]  # ["horizontal", "vertical", "rotational"]
    transformation_complexity: float  # 0-1 scale
    
    # Temporal context
    attempt_number: int = 0
    previous_attempts: List[torch.Tensor] = None
    success_history: List[bool] = None
    
    # Training set similarity
    similar_examples: List[Tuple[str, float]] = None  # [(task_id, similarity)]
    
    def to_tensor(self) -> torch.Tensor:
        """Convert context to tensor for H-module"""
        # Implementation here
        pass
```

### 2. Implement Context-Based Example Retrieval (Today/Tomorrow)

```python
# sage/context/context_retrieval.py
import faiss
import numpy as np
from typing import List, Tuple

class ContextRetriever:
    def __init__(self, training_set_path: str):
        self.training_examples = self.load_training_set(training_set_path)
        self.context_index = self.build_context_index()
    
    def build_context_index(self):
        """Build FAISS index for fast similarity search"""
        contexts = []
        for example in self.training_examples:
            context = self.encode_context(example)
            contexts.append(context.to_tensor().numpy())
        
        # Build index
        d = contexts[0].shape[0]  # dimension
        index = faiss.IndexFlatL2(d)
        index.add(np.array(contexts))
        return index
    
    def retrieve_similar(self, query_context: PuzzleContext, k: int = 5) -> List[Tuple[str, float]]:
        """Find k most similar training examples"""
        query_vector = query_context.to_tensor().numpy()
        distances, indices = self.context_index.search(query_vector.reshape(1, -1), k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            task_id = self.training_examples[idx]['task_id']
            similarity = 1.0 / (1.0 + dist)  # Convert distance to similarity
            results.append((task_id, similarity))
        
        return results
```

### 3. Separate H and L Modules with Clear Roles (Tomorrow)

```python
# sage/modules/h_module_context.py
class HModuleContext(nn.Module):
    """H-Module: Attends to context"""
    
    def __init__(self, context_dim: int = 32, hidden_dim: int = 768):
        super().__init__()
        self.context_encoder = ContextEncoderV2()
        self.context_refiner = nn.TransformerEncoder(...)
        self.retriever = ContextRetriever("data/training_set.json")
        
    def forward(self, puzzle_input: torch.Tensor, temporal_context: Optional[Dict] = None):
        # 1. Encode current puzzle into context
        context = self.context_encoder(puzzle_input)
        
        # 2. Retrieve similar examples
        similar = self.retriever.retrieve_similar(context)
        context.similar_examples = similar
        
        # 3. Refine with temporal information
        if temporal_context:
            context = self.integrate_temporal(context, temporal_context)
        
        # 4. Output rich context for L-module
        return context

# sage/modules/l_module_actor.py  
class LModuleActor(nn.Module):
    """L-Module: Acts within context"""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.execution_layers = nn.TransformerDecoder(...)
        self.output_head = nn.Linear(hidden_dim, 10)  # 10 colors
        
    def forward(self, puzzle_input: torch.Tensor, context: PuzzleContext):
        # Don't try to understand WHY, just HOW
        # Use context to guide execution
        
        if context.pattern_type == PatternType.FILLING:
            return self.execute_filling(puzzle_input, context)
        elif context.pattern_type == PatternType.EXTRACTION:
            return self.execute_extraction(puzzle_input, context)
        # ... etc
```

### 4. Design H↔L Communication Protocol (Tomorrow)

```python
# sage/modules/h_l_protocol.py
@dataclass
class HToLMessage:
    """Message from H (context) to L (action)"""
    context_vector: torch.Tensor
    pattern_type: PatternType
    confidence: float
    strategy_hint: str  # "fill_rectangles", "extract_pattern", etc.
    constraints: List[str]  # ["preserve_borders", "maintain_symmetry"]
    similar_examples: List[Tuple[str, float]]

@dataclass  
class LToHMessage:
    """Feedback from L (action) to H (context)"""
    action_taken: str
    output_generated: torch.Tensor
    execution_confidence: float
    anomalies_detected: List[str]  # ["unexpected_color", "size_mismatch"]
    needs_clarification: List[str]  # ["ambiguous_border", "multiple_patterns"]

class HLCommunicator:
    """Manages bidirectional communication"""
    
    def __init__(self):
        self.message_history = []
        
    def h_to_l(self, h_state: Dict, l_needs: List[str]) -> HToLMessage:
        """Craft message from H to L based on context and L's needs"""
        pass
        
    def l_to_h(self, l_result: Dict, h_context: PuzzleContext) -> LToHMessage:
        """Send execution feedback to H for context refinement"""
        pass
```

### 5. Create Training Pipeline (This Week)

```python
# sage/training/train_context_action.py
def train_h_module(h_module, training_data):
    """Train H to understand context, not solve puzzles"""
    optimizer = torch.optim.AdamW(h_module.parameters(), lr=1e-4)
    
    for puzzle, solution in training_data:
        # Target is CONTEXT, not solution
        true_context = extract_context(puzzle, solution)
        predicted_context = h_module(puzzle)
        
        # Loss is context accuracy, not pixel accuracy
        loss = context_similarity_loss(predicted_context, true_context)
        loss.backward()
        optimizer.step()

def train_l_module(l_module, training_data, h_module):
    """Train L to execute within given context"""
    optimizer = torch.optim.AdamW(l_module.parameters(), lr=1e-3)
    
    for puzzle, solution in training_data:
        # Get context from trained H
        with torch.no_grad():
            context = h_module(puzzle)
        
        # L learns to act within this context
        prediction = l_module(puzzle, context)
        loss = execution_loss(prediction, solution)
        loss.backward()
        optimizer.step()
```

## Testing Strategy

### Test 1: Context Quality
```python
def test_context_quality():
    """Does H produce meaningful context?"""
    h_module = load_trained_h()
    
    for test_puzzle in test_set:
        context = h_module(test_puzzle)
        
        # Check context dimensions
        assert context.pattern_type in PatternType
        assert 0 <= context.spatial_density <= 1
        assert len(context.similar_examples) > 0
        
        # Check retrieval quality
        for task_id, similarity in context.similar_examples:
            assert similarity > 0.5  # Should find relevant examples
```

### Test 2: Context-Guided Execution
```python
def test_context_execution():
    """Does L perform better with context?"""
    h_module = load_trained_h()
    l_module = load_trained_l()
    
    baseline_accuracy = test_without_context(l_module)
    context_accuracy = test_with_context(h_module, l_module)
    
    assert context_accuracy > baseline_accuracy
```

### Test 3: Quantization Impact
```python
def test_quantization():
    """Can we quantize L while keeping H precise?"""
    h_module_fp16 = load_h_module().half()
    l_module_int4 = quantize_to_int4(load_l_module())
    
    # Test that context quality maintained
    context_fp16 = h_module_fp16(test_puzzle)
    assert context_quality(context_fp16) > 0.9
    
    # Test that execution still works
    output = l_module_int4(test_puzzle, context_fp16)
    assert accuracy(output) > 0.8
```

## Timeline

### Week 1 (Sept 12-18)
- ✅ Document H↔L architecture
- [ ] Implement ContextEncoderV2
- [ ] Build context retrieval system
- [ ] Create H and L module skeletons
- [ ] Design communication protocol

### Week 2 (Sept 19-25)  
- [ ] Train H-module on context extraction
- [ ] Train L-module on context-guided execution
- [ ] Test H↔L communication
- [ ] Measure improvement over V3 baseline

### Week 3 (Sept 26-Oct 2)
- [ ] Implement temporal context persistence
- [ ] Test multi-round H↔L refinement
- [ ] Experiment with L-module quantization
- [ ] Optimize for inference speed

### Week 4 (Oct 3-9)
- [ ] Full system integration test
- [ ] Kaggle submission preparation
- [ ] Documentation and cleanup
- [ ] Performance benchmarking

## Success Metrics

1. **Context Coherence**: H maintains stable context across attempts
2. **Retrieval Quality**: Retrieved examples have >70% relevance
3. **Execution Accuracy**: L achieves >30% on ARC (vs 0% current)
4. **Memory Efficiency**: L quantized to INT4 with <5% accuracy loss
5. **Speed**: <100ms inference per puzzle

## Key Files to Create

```bash
# Core implementation
sage/context/context_encoder_v2.py
sage/context/context_retrieval.py
sage/modules/h_module_context.py
sage/modules/l_module_actor.py
sage/modules/h_l_protocol.py

# Training
sage/training/train_context_action.py
sage/training/context_similarity_loss.py

# Testing
sage/tests/test_context_quality.py
sage/tests/test_h_l_communication.py
sage/tests/test_quantization.py
```

## Remember

**We're not building intelligence - we're formalizing how intelligence already works:**
- H maintains context (like this conversation across days)
- L acts within it (like generating this response)
- Together they create coherent behavior

**The proof is this very discussion making perfect sense.**

---

*Ready to implement. Context is everything.*