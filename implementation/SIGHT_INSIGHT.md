# Sight–Insight: Tiled Perception and Cognitive Resonance

This note integrates two parallel principles — **FlashAttention tiling** in Transformers and **biologically inspired vision tiling** — into a single framework of perception and insight.

---

## Core Principle
The system never processes *all information equally*. Instead, it allocates compute based on **tiling + trust weighting**:

- **Peripheral tiles**: fast, lightweight, constantly updated. Capture motion, light, novelty, and conflict signals. Equivalent to cheap attention tiles that keep global situational awareness alive.  
- **Focus tiles**: slower, more detailed, carefully chosen. Capture objects, edges, symbols, or semantic meaning. Equivalent to high-resolution tiles loaded into on-chip memory for exact computation.  

The global state (coherence, strategy, action) emerges not from materializing *all possible comparisons* but from **localized compute plus global accumulation**.

---

## FlashAttention as Artificial Fovea
FlashAttention demonstrates that exact attention results can be achieved without materializing the full `[seq_len × seq_len]` attention matrix. It does this by:

- Streaming queries, keys, and values in **tiles** that fit in GPU SRAM/registers.  
- Computing **incremental softmax** statistics without storing every intermediate.  
- Writing only the final attended values back to global memory.  

This is a **digital parallel** to foveated vision: global awareness through peripheral sweeps, detail from focused windows, coherence from streaming accumulation.

---

## Biological Vision Parallel
Biological vision also divides the perceptual field:

- **Periphery**: low-resolution, high-speed; optimized for motion and light changes. Drives reflexive trust signals ("pay attention here").  
- **Fovea**: high-resolution, slower; optimized for detail and symbolic recognition. Allocates resources to meaning-making and strategy.  

SAGE's proposed vision pipeline mirrors this:  
- Peripheral sensors track global state cheaply.  
- Focus modules perform object/edge/symbol analysis where trust signals point.  
- Integration happens at the H-module, resonating across time scales.

---

## Sight → Insight
The transition from raw **sight** to abstract **insight** follows the same path:

1. **Tiling** partitions input into manageable zones.  
2. **Trust weighting** determines which tiles deserve deep focus.  
3. **Local compute** extracts detail only where needed.  
4. **Global accumulation** integrates results into a coherent temporal field.  

Thus, both vision and attention demonstrate that **wisdom emerges not from seeing everything at once, but from learning what deserves focus**.

---

## Placement
Save this file as `SIGHT_INSIGHT.md` in the shared context directory. It serves as a philosophical and architectural bridge between visual processing and cognitive attention in SAGE.