
import torch
import torchvision.transforms as T
from tinyvae_irp_plugin import TinyVAEIRP
from visual_monitor_impl import get_latest_crop

def run_vae_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyVAEIRP(input_channels=1, latent_dim=16).to(device)
    model.eval()

    transform = T.Compose([
        T.Resize((64, 64)),
        T.Grayscale(),
        T.ToTensor()
    ])

    crop = get_latest_crop()
    crop_tensor = transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        latent = model.get_latents(crop_tensor)
        print("Latent vector:", latent.squeeze().cpu().numpy())

        recon, _, _ = model(crop_tensor)
        recon_image = recon.squeeze().cpu().numpy()
        print("Reconstructed shape:", recon_image.shape)

if __name__ == "__main__":
    run_vae_pipeline()
