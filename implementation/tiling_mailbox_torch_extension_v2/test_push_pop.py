
import torch, mailbox_ext

def main():
    # Init mailboxes
    pbm_hdr, pbm_payload = [int(x.item()) for x in mailbox_ext.pbm_init(64, 1024)]
    ftm_hdr, ftm_ring = [int(x.item()) for x in mailbox_ext.ftm_init(256)]

    # Push several peripheral records (CUDA tensors)
    for i in range(16):
        rec = torch.full((64,), i % 256, dtype=torch.uint8, device='cuda')
        mailbox_ext.pbm_push_bytes_cuda(pbm_hdr, pbm_payload, rec)

    # Pop up to 8 records back (CUDA tensor output)
    out = mailbox_ext.pbm_pop_bulk_cuda(pbm_hdr, pbm_payload, max_records=8, record_stride=64)
    print("PBM popped bytes (CUDA):", out.shape)

    # Focus: push a tensor pointer + metadata
    x = torch.randn(2,3,4,5, device='cuda', dtype=torch.float16)
    ok = mailbox_ext.ftm_push_ptr(ftm_hdr, ftm_ring, int(x.data_ptr()),
                                  list(x.shape), list(x.stride()),
                                  x.dim(), 1, 42, 2)
    rec = mailbox_ext.ftm_pop(ftm_hdr, ftm_ring)
    dev_ptr = int(rec["dev_ptr"][0].item())
    shape = [int(v.item()) for v in rec["shape"]]
    stride = [int(v.item()) for v in rec["stride"]]
    print("FTM popped ptr:", hex(dev_ptr), "shape:", shape, "stride:", stride)

    # Simple zero-copy sanity: pointer equality
    print("Pointer equal?", dev_ptr == x.data_ptr())

if __name__ == "__main__":
    main()
