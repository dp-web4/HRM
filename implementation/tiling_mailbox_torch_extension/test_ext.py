
import torch, mailbox_ext

def main():
    # Init PBM
    hdr_payload = mailbox_ext.pbm_init(64, 1024)
    print("PBM header ptr:", int(hdr_payload[0].item()))
    print("PBM payload ptr:", int(hdr_payload[1].item()))

    # Init FTM
    hdr_ring = mailbox_ext.ftm_init(256)
    print("FTM header ptr:", int(hdr_ring[0].item()))
    print("FTM ring ptr:", int(hdr_ring[1].item()))

    # Allocate a CUDA tensor and print its data_ptr to show zero-copy intent
    x = torch.randn(2,3,128,64, device='cuda', dtype=torch.float16)
    print("Example tensor device ptr:", hex(x.data_ptr()))

    print("Extension smoke OK")

if __name__ == "__main__":
    main()
