
# PyTorch Extension — Mailbox Bindings (Push/Pop)

This version exposes **push/pop** for both mailboxes so you can exercise them from Python.

## Build
```bash
export TORCH_CUDA_ARCH_LIST="8.7"   # Or Xavier: 7.2
python setup.py build_ext --inplace
```

## API
```python
import torch, mailbox_ext

# Init
hdr_payload = mailbox_ext.pbm_init(record_stride=64, capacity=1024)
pbm_hdr_ptr, pbm_payload_ptr = int(hdr_payload[0].item()), int(hdr_payload[1].item())

hdr_ring = mailbox_ext.ftm_init(capacity=256)
ftm_hdr_ptr, ftm_ring_ptr = int(hdr_ring[0].item()), int(hdr_ring[1].item())

# Peripheral push/pop
record = torch.randint(0, 256, (64,), dtype=torch.uint8, device='cuda')
ok = mailbox_ext.pbm_push_bytes_cuda(pbm_hdr_ptr, pbm_payload_ptr, record)  # returns bool
out = mailbox_ext.pbm_pop_bulk_cuda(pbm_hdr_ptr, pbm_payload_ptr, max_records=8, record_stride=64)
# 'out' is a CUDA uint8 tensor of shape [N*record_stride]

# Focus push/pop (pointer handoff)
x = torch.randn(2,3,4,5, device='cuda', dtype=torch.float16)
ok = mailbox_ext.ftm_push_ptr(ftm_hdr_ptr, ftm_ring_ptr,
                              int(x.data_ptr()), list(x.shape), list(x.stride()),
                              x.dim(), 1, 42, 2)  # dtype=1(F16), tag=42, ttl=2
rec = mailbox_ext.ftm_pop(ftm_hdr_ptr, ftm_ring_ptr)  # returns tuple
```

## Test
```bash
python test_push_pop.py
```

The test:
- Pushes 16 peripheral records and pops some back (on GPU).
- Pushes a focus pointer (tensor) and pops it back; checks that the pointer/shape/stride round-trip.
- Optionally prints a quick profiler summary if available.
