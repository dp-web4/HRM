
# IMPLEMENTATION_NOTES_TILING.md
Tiling-integrated mailbox layer for SAGE/HRM: peripheral broadcast + focus tensor handoff.

## Mailbox Classes
### 1) PeripheralBroadcastMailbox (many→many, fixed-size records)
- Purpose: motion/light/novelty/conflict; cheap global awareness
- Record: fixed 128–256B (aligned); circular buffer in **GPU global memory**
- Ops: `try_push`, `try_pop_bulk(max_n)`; coalesced loads/stores
- Sync: producer records **CUDA event**; consumer stream `cudaStreamWaitEvent`

### 2) FocusTensorMailbox (few→few, pointer-based, zero-copy)
- Purpose: deep analysis handoffs (objects/edges/symbols, attention tiles)
- Record: device pointer + tensor metadata (`shape/stride/dtype`) + tags/TTL
- Ops: `try_push`, `try_pop`
- No bulk copies; pass device pointers (or DLPack handles)

## Trust-weighted Scheduler (CPU-arbitrated to start)
Inputs: peripheral signals → `trust_map: tile_id → score`  
Policy:
- Select top-K tiles by score; enqueue **FocusTensorMailbox**
- Keep peripheral broadcast flowing under load
Mechanics:
- Per-module CUDA streams; record/wait on events
- Optional stream priority: focus > peripheral

## Data Structures (shared headers)
See `mailbox_peripheral.h` and `mailbox_focus.h` for record layout and APIs.

## Policy Hooks
- **Backpressure**: if focus is full, drop lowest-trust candidate or demote to peripheral
- **TTL**: avoid deep work on stale tiles
- **Fairness**: cap per-producer focus slots
- **Coherence pinning**: grant short time slices to active focus tiles to reduce thrash

## Telemetry (Victory Checks)
- Rates: `peripheral/sec`, `focus/sec`, drops/demotions, queue depth
- Latency: push→pop p50/p95 per mailbox class
- Residency: % device-only vs host hops
- Attention mode: flash/mem/math; effective tile sizes

## Done-When (V1)
- Peripheral broadcast under target load with <1% drops
- Stable focus throughput with backpressure + fairness
- Zero-copy handoff verified (no host copies in profiler)
- SDPA/FlashAttention runs on focus tiles (smoke passes)
- Before/after: insight count increases where trust signals point

## Pitfalls
- Hidden host transfers (pin a profiler trace early)
- Misaligned strides (pad record to 128B for coalescing)
- Spin-wait device kernels (prefer events; persistent router only if needed)
- Pointer lifetime bugs (explicit ownership rules in README)

## Placement
Save alongside `GPU_MAILBOX.md` in the implementation directory.
