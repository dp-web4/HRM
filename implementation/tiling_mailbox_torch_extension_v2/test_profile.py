
import torch, mailbox_ext
from torch.profiler import profile, record_function, ProfilerActivity

def has_memcpy_events(prof):
    # Look for HtoD / DtoH memcpy entries
    memcpy_keywords = ("memcpy", "Memcpy", "toDevice", "toHost", "MemcpyHtoD", "MemcpyDtoH")
    for evt in prof.key_averages():
        name = evt.key if hasattr(evt, "key") else evt.key
        if any(k in name for k in memcpy_keywords):
            return True, name
    return False, None

def main():
    torch.cuda.synchronize()

    # Init mailboxes
    pbm_hdr, pbm_payload = [int(x.item()) for x in mailbox_ext.pbm_init(64, 1024)]
    ftm_hdr, ftm_ring = [int(x.item()) for x in mailbox_ext.ftm_init(256)]

    # Prepare CUDA records up front (so no implicit HtoD later)
    recs = [torch.full((64,), i % 256, dtype=torch.uint8, device='cuda') for i in range(8)]
    x = torch.randn(2,3,4,5, device='cuda', dtype=torch.float16)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=False) as prof:
        with record_function("mailbox_push_pop_cycle"):
            # Peripheral pushes (CUDA -> CUDA mailbox)
            for r in recs:
                mailbox_ext.pbm_push_bytes_cuda(pbm_hdr, pbm_payload, r)

            # Pop back (CUDA tensor output)
            out = mailbox_ext.pbm_pop_bulk_cuda(pbm_hdr, pbm_payload, max_records=len(recs), record_stride=64)

            # Focus pointer push/pop (no tensor data moved)
            mailbox_ext.ftm_push_ptr(ftm_hdr, ftm_ring, int(x.data_ptr()),
                                     list(x.shape), list(x.stride()),
                                     x.dim(), 1, 7, 2)
            rec = mailbox_ext.ftm_pop(ftm_hdr, ftm_ring)

        torch.cuda.synchronize()

    # Basic functional checks
    assert out.is_cuda and out.dtype == torch.uint8, "PBM pop should be CUDA uint8 tensor"
    dev_ptr = int(rec["dev_ptr"][0].item())
    assert dev_ptr == x.data_ptr(), "Focus pointer should round-trip without change"

    # Memcpy assertion
    memcpy_found, which = has_memcpy_events(prof)
    if memcpy_found:
        print("WARNING: Detected memcpy event in profiler:", which)
    else:
        print("OK: No memcpy H<->D detected during push/pop cycle")

    # Print a short table of the top events for visibility
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    main()
