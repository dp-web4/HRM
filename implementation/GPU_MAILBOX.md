
# GPU Mailboxes for Module Intercommunication (CUDA Streams + Events)

This document describes a mailbox-based design for zero-copy communication between multiple modules (LLMs, HRM, Sidecar) running on a shared NVIDIA GPU. The goal is to keep data resident on device memory, using the CPU only as an arbiter and persistence layer.

## Goals
- **Zero-copy** GPU-resident exchanges between modules.
- **CPU arbitration only**: CPU allocates buffers, sets/waits events, persists state, but does not shuttle data back and forth.
- **Minimize host trips**: data leaves GPU only for persistence or CPU-optimized tasks.

## Architecture Overview
- **Modules**: HRM, Sidecar Memory, multiple LLMs.
- **Mailboxes**: fixed-size circular buffers in GPU global memory, serving as message queues.
- **Streams & Events**: each module has a dedicated CUDA stream; producers record events when writing, consumers wait on events before reading.
- **Memory Sharing**: large tensors exchanged by pointer metadata (device address + shape/dtype) instead of by value.

## Memory Layout
Each mailbox contains:
```
MailboxHeader {
  magic, version
  capacity_bytes
  record_stride
  write_idx (atomic<uint64>)
  read_idx  (atomic<uint64>)
  producer_id, consumer_id
}
Payload (capacity_bytes)
```
- Payload = fixed-stride slots for records (e.g. 256B each).
- Record format: `{len, msg_type, [payload bytes]}`.

## Device-Side API
- `try_push(Mailbox*, src, len)` → copies payload into next slot, updates write index.
- `try_pop(Mailbox*, dst, len_out)` → copies from slot, updates read index.
- Atomics ensure correctness; `__threadfence()` guarantees memory visibility.

## Synchronization
- **CPU-arbitrated (recommended start)**: producers record CUDA events after push; CPU waits and schedules consumers with `cudaStreamWaitEvent`.
- **Device-arbitrated (advanced)**: persistent kernel loops over mailboxes and signals readiness, reducing CPU involvement but consuming SM time.

## Tensor Handoff
- Allocate shared GPU buffers from a memory pool (`cudaMallocAsync` or PyTorch allocator).
- Mailboxes exchange **pointers + metadata**.
- PyTorch interoperability via **DLPack** (`to_dlpack` / `from_dlpack`) ensures zero host copies.

## Scheduling
- Each module runs in its own CUDA stream.
- Producer records event on stream → consumer stream waits before popping.
- CUDA Graphs can capture steady-state pipelines to reduce launch overhead.

## Persistence
- Jetson lacks GPUDirect Storage.
- Use pinned host memory (`cudaHostAlloc`) + `cudaMemcpyAsync` for staging persistence to disk.

## Red Flags (Premature Victory Checks)
- Host copies creeping in (profile with PyTorch or `nvidia-smi dmon`).
- Oversubscription of VRAM → allocator thrashing.
- Spin-wait device kernels hogging SMs.
- Misaligned or uncoalesced mailbox accesses → poor throughput.
- Lifetime bugs from dangling device pointers.

## Next Steps
1. Single-process prototype: one CUDA context, 3 streams, 2 mailboxes.
2. Use fixed 256B stride records to start.
3. Integrate with PyTorch via DLPack.
4. Instrument: count pushes/pops, measure latency, confirm device-only flow.
