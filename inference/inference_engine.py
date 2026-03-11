"""
Inference Engine for the End-to-End Vision Mamba Dehazer.
Handles model loading, image preprocessing, and prediction.
Compatible with both legacy AttentionUNet and the new MambaDehaze.
"""
import torch
import torchvision.transforms as T
from PIL import Image
from models.mamba_arch import MambaDehaze
import os


class DehazeInference:
    def __init__(self, checkpoint_path, device=None, model_type="mamba"):
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if model_type == "mamba":
            if os.path.exists(checkpoint_path):
                # Load checkpoint first to extract the saved training config
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                cfg = checkpoint.get('config', {})
                self.model = MambaDehaze(
                    img_size=cfg.get('image_size', 256),
                    embed_dim=cfg.get('embed_dim', 64),
                    d_state=cfg.get('d_state', 16),
                    n_layers=cfg.get('n_layers', 4),
                    dropout=cfg.get('dropout', 0.1)
                ).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"[Inference] Loaded checkpoint: {checkpoint_path}")
                print(f"[Inference] Model config: embed_dim={cfg.get('embed_dim')}, "
                      f"n_layers={cfg.get('n_layers')}, d_state={cfg.get('d_state')}")
            else:
                # No checkpoint — fall back to defaults and warn
                print(f"[Inference] WARNING: No checkpoint found at {checkpoint_path}")
                self.model = MambaDehaze(img_size=256, embed_dim=64, n_layers=4).to(self.device)
        else:
            # Legacy fallback
            from models.arch import AttentionUNetDehaze
            from models.physics import PhysicsReconstruction
            self.legacy_model = AttentionUNetDehaze().to(self.device)
            self.physics = PhysicsReconstruction().to(self.device)
            self.model = None

        if os.path.exists(checkpoint_path) and model_type != "mamba":
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            target_model = self.model if self.model else self.legacy_model
            target_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[Inference] Loaded checkpoint: {checkpoint_path}")
        elif not os.path.exists(checkpoint_path):
            pass  # Warning already printed above

        if self.model:
            self.model.eval()
        else:
            self.legacy_model.eval()

        self.model_type = model_type
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def predict(self, image_path):
        """
        Dehaze image and return intermediate maps.

        Returns:
            dehazed_img: PIL Image (final result)
            physics_img: PIL Image (AOD coarse result)
            refine_map:  PIL Image (Residual delta visualized)
            metadata: dict
        """
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.model_type == "mamba":
                # Extract all maps from the hybrid model
                dehazed, physics, refine = self.model(img_tensor, return_maps=True)
            else:
                # Legacy fallback (simulated maps)
                t_map, a_light = self.legacy_model(img_tensor)
                dehazed = self.physics(img_tensor, t_map, a_light)
                physics = dehazed
                refine = torch.zeros_like(dehazed)

        # Helper to convert tensor to PIL
        def to_pil(tensor):
            img = T.ToPILImage()(tensor.squeeze(0).cpu().clamp(0, 1))
            return img.resize(original_size, Image.LANCZOS)

        metadata = {
            'model_type': self.model_type,
            'input_size': original_size,
            'processing_size': (256, 256)
        }

        return to_pil(dehazed), to_pil(physics), to_pil(refine), metadata


if __name__ == "__main__":
    print("[Inference Engine] Ready. Use DehazeInference class to predict.")
