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
            self.model = MambaDehaze(img_size=256, embed_dim=64, n_layers=4).to(self.device)
        else:
            # Legacy fallback
            from models.arch import AttentionUNetDehaze
            from models.physics import PhysicsReconstruction
            self.legacy_model = AttentionUNetDehaze().to(self.device)
            self.physics = PhysicsReconstruction().to(self.device)
            self.model = None

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            target_model = self.model if self.model else self.legacy_model
            target_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[Inference] Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"[Inference] WARNING: No checkpoint found at {checkpoint_path}")

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
        Dehaze a single image.

        Args:
            image_path: path to hazy image

        Returns:
            dehazed_img: PIL Image (original resolution)
            metadata: dict with model info
        """
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.model_type == "mamba":
                dehazed = self.model(img_tensor)
            else:
                t_map, a_light = self.legacy_model(img_tensor)
                dehazed = self.physics(img_tensor, t_map, a_light)

        # Convert to PIL and resize back to original dimensions
        dehazed_img = T.ToPILImage()(dehazed.squeeze(0).cpu().clamp(0, 1))
        dehazed_img = dehazed_img.resize(original_size, Image.LANCZOS)

        metadata = {
            'model_type': self.model_type,
            'input_size': original_size,
            'processing_size': (256, 256)
        }

        return dehazed_img, metadata


if __name__ == "__main__":
    print("[Inference Engine] Ready. Use DehazeInference class to predict.")
