# SAGE Attention Mechanism: Dtype Handling Analysis
## Trust-Based Selection vs Baseline Comparison

### Executive Summary

The dtype mismatch in trust-based selection stems from a **critical data type conversion sequence** that diverges between baseline (None) and trust-based paths. Both paths use the same attention weights, but trust-based selection introduces a **numpy→float32→tensor pathway** that can create mismatched precision states downstream.

---

## 1. BASELINE PATH (trust_selector=None)

### SelectiveMoELayer.forward() - Baseline Route (Lines 419-427)

```python
# BASELINE: Standard SNARC-augmented selection
selected_expert_ids, router_weights = self.expert_loader.select_experts_snarc(
    hidden_states,           # [batch, seq, hidden] - original dtype
    self.layer_id,
    num_experts=self.num_experts_per_tok,
    snarc_salience=snarc_salience,
    metabolic_state=metabolic_state
)
```

### SelectiveExpertLoader.select_experts_snarc() - Baseline Tensor Flow (Lines 243-268)

```python
# Step 1: Convert hidden_states to flat representation
hidden_flat = hidden_states.view(-1, hidden_size)  # [batch*seq, hidden]

# Step 2: Compute router logits - CRITICAL DTYPE POINT
router_logits = F.linear(hidden_flat, router)  # [batch*seq, num_experts]
#                                    ^
#                            router is torch.float32 (line 159)
#                 Router output dtype = query_states.dtype

# Step 3: Apply expert availability mask
mask = torch.full((1, 128), float('-inf'), device=router_logits.device)
mask[0, available_experts] = 0

# Step 4: Softmax to get routing weights
routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
#                                                    ^
#                                          Forces float32 internally

# Step 5: Select top-k and reshape back
top_k_values, top_k_indices = torch.topk(routing_weights, k=num_experts, dim=-1)
top_k_values = top_k_values / top_k_values.sum(dim=-1, keepdim=True)

top_k_indices = top_k_indices.view(batch_size, seq_length, num_experts)
top_k_values = top_k_values.view(batch_size, seq_length, num_experts)  # Still float32

return top_k_indices, top_k_values  # Returns tensors
```

**Key: Stays in tensor space, maintains float32 throughout**

---

## 2. TRUST-BASED PATH (trust_selector is not None)

### SelectiveMoELayer.forward() - Trust Route (Lines 387-413)

```python
if self.trust_selector is not None:
    # Get router logits FIRST
    router = self.expert_loader.load_router(self.layer_id)
    hidden_flat = hidden_states.view(-1, hidden_dim)  # [batch*seq, hidden]
    router_logits = F.linear(hidden_flat, router)    # [batch*seq, num_experts]
    
    # CRITICAL DIVERGENCE POINT 1: CPU + numpy conversion
    mean_embedding = hidden_states.mean(dim=(0, 1))     # [hidden]
                    .detach()
                    .cpu()
                    .numpy()                            # Now numpy!
                    .astype(np.float32)                 # Explicit float32
    
    # CRITICAL DIVERGENCE POINT 2: Call trust selector
    result = self.trust_selector.select_experts(
        router_logits=router_logits[0],     # Take first token only!
                                           # [num_experts] tensor
        context=None,
        k=self.num_experts_per_tok,
        input_embedding=mean_embedding      # NUMPY array, not tensor
    )
    
    # CRITICAL DIVERGENCE POINT 3: Convert numpy back to tensor
    selected_ids = torch.tensor(
        result.selected_expert_ids,
        device=hidden_states.device,
        dtype=torch.long
    )
    selected_expert_ids = selected_ids.unsqueeze(0).unsqueeze(0)\
                                     .expand(batch_size, seq_length, -1)
    
    # CRITICAL DIVERGENCE POINT 4: Scores also converted
    selected_weights = torch.tensor(
        result.selection_scores,                # From trust selector result
        device=hidden_states.device,
        dtype=torch.float32                     # Explicitly float32
    )
    router_weights = selected_weights.unsqueeze(0).unsqueeze(0)\
                                    .expand(batch_size, seq_length, -1)
    # NOW we have router_weights as tensors
```

### TrustBasedExpertSelector.select_experts() - Numpy Path (Lines 173-177)

```python
# Convert router logits to NUMPY for "easier manipulation"
if HAS_TORCH and torch is not None and isinstance(router_logits, torch.Tensor):
    router_scores = router_logits.detach().cpu().numpy().astype(np.float32)
    #                                                      ^
    #                                            TENSOR → NUMPY conversion
else:
    router_scores = np.array(router_logits, dtype=np.float32)

# Get contextual trust scores (also numpy, lines 247-259)
trust_scores = self._get_contextual_trust_scores(context)  # Returns np.ndarray
#                                                              dtype=np.float32

# Combine in numpy space (line 185)
α = self.exploration_weight
combined_scores = α * router_scores + (1 - α) * trust_scores
# ^^^ All numpy operations ^^^

# Select top-k in numpy (line 188)
top_k_indices = np.argsort(combined_scores)[-k:][::-1]
selected_scores = combined_scores[top_k_indices]      # numpy float32
selected_router = router_scores[top_k_indices]         # numpy float32
selected_trust = trust_scores[top_k_indices]           # numpy float32

# Return ExpertSelectionResult
return ExpertSelectionResult(
    selected_expert_ids=[int(e) for e in final_experts],
    selection_scores=selected_scores.tolist(),        # Convert to list!
                                                      # Loses dtype info
    router_scores=selected_router.tolist(),
    trust_scores=selected_trust.tolist(),
    ...
)
```

---

## 3. CRITICAL DTYPE DIVERGENCE POINTS

### Divergence Point 1: CPU Movement & Numpy Conversion (Line 397)

**BASELINE:**
- Stays in tensor space throughout
- Router computation: `F.linear()` outputs query_states.dtype

**TRUST-BASED:**
```python
mean_embedding = hidden_states.mean(dim=(0, 1))  # [hidden] tensor
                .detach().cpu()                   # Moved to CPU
                .numpy()                          # tensor → numpy!
                .astype(np.float32)               # Numpy float32
```

**Problem**: Mixed tensor/numpy pathway violates dtype consistency assumptions

### Divergence Point 2: Scalar vs Expanded Routing (Lines 401, 412)

**BASELINE:**
```python
# Returns [batch, seq, num_experts] routing weights
selected_expert_ids: [batch, seq, num_experts]
router_weights: [batch, seq, num_experts] - float32 tensors
```

**TRUST-BASED:**
```python
# Only uses FIRST TOKEN's routing (router_logits[0])
result = self.trust_selector.select_experts(
    router_logits=router_logits[0],  # [num_experts] - single token!
    ...
)

# Then repeats same selection for all tokens
selected_weights = torch.tensor(result.selection_scores, dtype=torch.float32)
router_weights = selected_weights.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)
#                                              ^^^^^^^^^^^^^^^^^
#                           Broadcasting same weights across all tokens
```

**Problem**: Per-token routing becomes uniform routing → Same selections repeated

### Divergence Point 3: List → Tensor Conversion (Line 228)

**BASELINE:**
- Returns numpy arrays directly → `top_k_values` is float32 array
- Gets used as-is in loop (line 443)

**TRUST-BASED:**
```python
selection_scores=selected_scores.tolist(),  # numpy → Python list
#                              ^^^^^
#                         LOSES numpy dtype info

# Later reconstructed (line 412):
selected_weights = torch.tensor(
    result.selection_scores,  # Now a Python list, not numpy
    device=hidden_states.device,
    dtype=torch.float32       # Must be explicit!
)
```

**Problem**: Type information erasure through list conversion

---

## 4. WHERE TRUST-BASED CAUSES DTYPE MISMATCH

### SelectiveMoELayer._expert_forward() (Lines 474-508)

```python
def _expert_forward(self, x: torch.Tensor, expert_weights: Dict[str, torch.Tensor], debug: bool = False) -> torch.Tensor:
    """
    Standard expert MLP: h = down(silu(gate(x)) * up(x))
    """
    gate_output = F.linear(x, gate_proj)
    up_output = F.linear(x, up_proj)
    intermediate = F.silu(gate_output) * up_output
    output = F.linear(intermediate, down_proj)
    return output
```

### SelectiveMoELayer.forward() - Expert Loop (Lines 439-470)

```python
for b in range(batch_size):
    for s in range(seq_length):
        token_hidden = hidden_states[b, s:s+1, :]      # [1, 1, hidden]
        token_expert_ids = selected_expert_ids[b, s]   # [num_experts_per_tok]
        token_weights = router_weights[b, s]           # [num_experts_per_tok]
        
        token_output = torch.zeros_like(token_hidden)
        
        for i, expert_id in enumerate(token_expert_ids):
            expert_out = self._expert_forward(token_hidden, expert_weights, debug=False)
            
            # MISMATCH HAPPENS HERE:
            weight_scalar = token_weights[i].to(dtype=torch.float32)
            #                                   ^^^^^^^^^^^^^^^^^
            #                              Forced conversion suggests mismatch!
            
            token_output += weight_scalar * expert_out
            #               ^^^^^^^^^^^^^ float32
            #               ^^^^^^^^^^^ expert_out may be different dtype
```

**The Mismatch Root Cause:**

1. `token_weights` comes from trust selector (explicitly float32)
2. `expert_out` comes from expert forward pass
3. Expert weights loaded with `.float()` (line 192 in loader)
4. But `x` (token_hidden) dtype depends on what flows through attention

### SelectiveTransformerLayer.forward() - Attention Dtype (Lines 614-629)

```python
# Self-attention with residual
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, debug=debug)
#                               ^^^^^^^^^
#                          Dtype DEPENDS on attention implementation
hidden_states = residual + hidden_states

# MoE with residual
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.moe(
    hidden_states,
    ...
)
```

### GroupedQueryAttention.forward() - CRITICAL ATTENTION DTYPE (Line 319)

```python
# Scaled dot-product attention
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

if attention_mask is not None:
    attn_weights = attn_weights + attention_mask

# CRITICAL: Softmax enforces float32 computation, then converts back!
attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#                                                                    ^^^^^^^^^^^^^^^^^^^
#                            Converts TO query_states.dtype for output!

attn_output = torch.matmul(attn_weights, value_states)

# Reshape back
attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = attn_output.reshape(batch_size, seq_length, self.num_attention_heads * self.head_dim)

# Output projection
attn_output = self.o_proj(attn_output)
return attn_output
```

**Critical insight:** `attn_weights` is cast TO `query_states.dtype` BEFORE matmul with values!

---

## 5. ROOT CAUSE HYPOTHESIS

### The dtype mismatch chain:

```
BASELINE PATH:
┌─────────────────────────────────────────────────────────────┐
│ 1. hidden_states: [batch, seq, hidden] - original dtype    │
│ 2. Attention output: same dtype (attn_weights cast back)   │
│ 3. router_logits: F.linear() → query_states.dtype          │
│ 4. routing_weights: float32 softmax → stays tensors        │
│ 5. Expert input: hidden_states → original dtype             │
│ 6. Expert output: F.linear() on original dtype              │
│ 7. Weighted sum: weight (float32) * expert_out (orig dtype) │
│    → PyTorch broadcasts float32 with original dtype         │
└─────────────────────────────────────────────────────────────┘

TRUST-BASED PATH:
┌─────────────────────────────────────────────────────────────┐
│ 1. hidden_states: [batch, seq, hidden] - original dtype    │
│ 2. Attention output: same dtype (attn_weights cast back)   │
│ 3. router_logits[0]: F.linear() → query_states.dtype       │
│ 4. Router scores → NUMPY (astype(float32))                 │
│ 5. Trust scores → NUMPY (dtype=np.float32)                 │
│ 6. Combined scores → NUMPY (float32 arithmetic)            │
│ 7. List conversion → PYTHON LIST (dtype lost!)             │
│ 8. Back to tensor → torch.tensor(..., dtype=float32)       │
│ 9. Broadcast to all tokens → [batch, seq, num_experts]     │
│ 10. Expert input: hidden_states[b,s:s+1,:] → original dtype │
│ 11. Expert output: F.linear() on original dtype             │
│ 12. weight_scalar: forcibly converted to float32            │
│ 13. Weighted sum: float32 * original_dtype                  │
│     → MAY cause dtype promotion unexpectedly                │
└─────────────────────────────────────────────────────────────┘
```

### Why Trust-Based Causes Mismatch:

1. **Numpy detour loses context**: When router_scores converted to numpy (line 175 of trust selector), the original dtype information is lost

2. **List conversion erases type**: `selection_scores.tolist()` converts numpy array to Python list with no dtype metadata

3. **Explicit float32 cast**: When reconstructing tensor from list (line 412), MUST explicitly use `dtype=torch.float32` because list type is ambiguous

4. **Forced conversion flag**: The explicit `.to(dtype=torch.float32)` on line 462 suggests downstream code found a mismatch and added defensive conversion

5. **Single-token routing repeated**: Using only `router_logits[0]` means all tokens get identical weights, which changes routing semantics compared to per-token baseline routing

---

## 6. SPECIFIC CODE DIFFERENCES

### Difference 1: Router Logit Handling

**BASELINE (select_experts_snarc, line 245):**
```python
router_logits = F.linear(hidden_flat, router)  # [batch*seq, num_experts]
# Stays as tensor, shape and dtype preserved for all tokens
```

**TRUST-BASED (SelectiveMoELayer.forward, line 392):**
```python
router_logits = F.linear(hidden_flat, router)  # [batch*seq, num_experts]
result = self.trust_selector.select_experts(
    router_logits=router_logits[0],  # Only first token [num_experts]!
    ...
)
```

### Difference 2: Type Pathway

**BASELINE:**
```python
Tensor → Tensor → Tensor → Tensor
(F.linear → softmax → topk → reshape)
```

**TRUST-BASED:**
```python
Tensor → Tensor → Numpy → List → Tensor
(F.linear → detach/cpu/numpy → tolist → torch.tensor)
```

### Difference 3: Weight Assignment

**BASELINE (line 263):**
```python
top_k_values = top_k_values / top_k_values.sum(dim=-1, keepdim=True)
# Result: tensor, float32, maintains metadata
```

**TRUST-BASED (line 412):**
```python
selected_weights = torch.tensor(
    result.selection_scores,  # Python list
    device=hidden_states.device,
    dtype=torch.float32       # Must specify explicitly!
)
```

---

## 7. CACHED STATE & PERSISTENCE

### No persistent cache issues found in baseline
- `select_experts_snarc` is stateless (line 211)
- Returns tensors that flow through immediately
- No intermediate storage of mixed types

### Trust selector has class state (lines 131-137)
```python
# Expert cache (expert_id → loaded status)
self.loaded_experts: Dict[int, bool] = {}

# Statistics
self.total_selections = 0
self.total_substitutions = 0
self.cache_hit_rate = 0.0
```

But this shouldn't affect dtype (only tracks cache hits/misses)

---

## 8. RECOMMENDED FIXES

### Option A: Keep Trust Selector in Tensor Space (Recommended)

```python
# In TrustBasedExpertSelector.select_experts():

# DON'T convert to numpy
if HAS_TORCH and torch is not None and isinstance(router_logits, torch.Tensor):
    router_scores = router_logits  # KEEP as tensor!
else:
    router_scores = torch.tensor(router_logits, dtype=torch.float32)

# Get trust scores as tensors too
trust_scores = self._get_contextual_trust_scores_tensor(context)

# Combine in tensor space
α = self.exploration_weight
combined_scores = α * router_scores + (1 - α) * trust_scores

# Select top-k
top_k_indices = torch.topk(combined_scores, k, dim=-1)[1]
selected_scores = torch.topk(combined_scores, k, dim=-1)[0]

# Return tensors directly
return ExpertSelectionResult(
    selected_expert_ids=top_k_indices.tolist(),
    selection_scores=selected_scores,  # Keep as tensor!
    ...
)
```

### Option B: Per-Token Routing in Trust Selector

```python
# In SelectiveMoELayer.forward():

# Use ALL token router logits, not just first!
result = self.trust_selector.select_experts(
    router_logits=router_logits,  # [batch*seq, num_experts]
    ...
)

# Don't expand/broadcast - already per-token
selected_expert_ids: [batch*seq, num_experts]
router_weights: [batch*seq, num_experts]

# Reshape to [batch, seq, num_experts]
# Maintain per-token semantics like baseline
```

### Option C: Explicit Dtype Normalization

```python
# In SelectiveMoELayer.forward():

# After trust selector returns
selected_weights = torch.tensor(
    result.selection_scores,
    device=hidden_states.device,
    dtype=torch.float32
)

# Ensure consistency with attention output dtype
if hasattr(hidden_states, 'dtype'):
    selected_weights = selected_weights.to(dtype=hidden_states.dtype)
```

---

## 9. TESTING VERIFICATION

### Test: Dtype Consistency (Proposed)

```python
def test_dtype_consistency():
    """Verify baseline and trust-based paths preserve dtype"""
    
    # Create input in mixed precision
    hidden_states = torch.randn(2, 10, 2048, dtype=torch.float32)
    
    # Baseline path
    baseline_output = layer(hidden_states, trust_selector=None)
    
    # Trust-based path
    trust_output = layer(hidden_states, trust_selector=trust_selector)
    
    # Both should have same dtype
    assert baseline_output.dtype == trust_output.dtype, \
        f"Dtype mismatch: baseline {baseline_output.dtype}, trust {trust_output.dtype}"
    
    # Expert weights should not cause type promotion
    for b in range(batch_size):
        for s in range(seq_length):
            token_output_dtype = token_output.dtype
            assert token_output_dtype == hidden_states.dtype, \
                f"Token output dtype changed: {hidden_states.dtype} → {token_output_dtype}"
```

---

## 10. CONCLUSION

**Root cause:** Trust-based selection introduces a **tensor→numpy→list→tensor** conversion cycle that erases dtype metadata, forcing explicit `dtype=float32` specifications that may not match the original hidden state dtype. This causes precision mismatches when combining weights with expert outputs.

**Why baseline doesn't have this:** Baseline stays in pure tensor space, preserving dtype information through the entire pipeline.

**Why it matters:** Mixed dtype operations can cause unexpected precision loss or promotion, affecting model accuracy and reproducibility.

**Solution:** Either (1) keep trust selector in tensor space, (2) use per-token routing consistently, or (3) explicitly normalize output dtypes to match input dtypes.
