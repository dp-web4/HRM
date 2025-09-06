# Agent Zero Behavior Confirmation

*Date: September 6, 2025*  
*Confirmation Testing Session*

## Executive Summary

We have confirmed that both the original HRM checkpoint and our new SAGE implementation exhibit "Agent Zero" behavior - achieving seemingly respectable accuracy scores by outputting constants (mostly zeros) that match dataset statistics, not through actual reasoning.

## Test Results

### Original HRM Checkpoint (`hrm_arc_best.pt`)
- **Step**: 7000 (early checkpoint)
- **Parameters**: 6.95M
- **ARC-AGI-2 Score**: 34.64% (on 10-task subset)
- **Behavior**: Strong bias toward class 0 (output bias=0.290 vs -0.25 for others)
- **Evidence**: Tasks with 78.9% zeros achieve 78.9% accuracy; tasks with few zeros get 0%

### Our SAGE Implementation (`sage_best.pt`)
- **Training**: 1 epoch on synthetic data
- **Parameters**: 17M (development config, not full 110M)
- **Test Score**: 67.56% on synthetic sparse grids
- **Behavior**: Outputs ALL zeros regardless of input
- **Evidence**: 100% of test cases produced identical all-zero outputs

## Key Findings

1. **Complete Input Invariance**: Both models produce identical outputs regardless of input variation
2. **Statistical Shortcut**: "Accuracy" comes from matching dataset sparsity (~60-80% zeros in ARC)
3. **No Actual Reasoning**: Zero evidence of pattern recognition or problem-solving

## Why This Matters

This confirmation validates the team's architectural insights:
- **5-7M parameters is below reasoning emergence threshold**
- **Without context awareness, models don't understand what they're solving**
- **Need external LLM for "language to think with"**
- **100M parameters identified as minimum for reasoning emergence**

## The Agent Zero Lesson

Agent Zero achieved perfect consistency by doing nothing, revealing that:
- Benchmarks can be gamed through dataset statistics
- Pixel accuracy ≠ task understanding
- Architecture alone isn't enough - need proper scale and context

This discovery motivated the pivot to SAGE as an attention orchestrator that knows WHEN to employ external resources rather than trying to be everything.

---

*"Agent Zero found the global optimum for a context-free universe. By doing nothing, it revealed everything wrong with how we measure intelligence."*