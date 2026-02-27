import torch
import torchvision.transforms as T
from PIL import Image
from models import AttentionUNetDehaze, PhysicsReconstruction
import os

class DehazeInference:
    def __init__(self, checkpoint_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttentionUNetDehaze().to(self.device)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.physics = PhysicsReconstruction().to(self.device)
        self.transform = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor()
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size # (width, height)
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            t_map, a_light = self.model(img_tensor)
            dehazed = self.physics(img_tensor, t_map, a_light)
            
        # Convert to PIL and resize back to original dimensions
        dehazed_img = T.ToPILImage()(dehazed.squeeze(0).cpu())
        dehazed_img = dehazed_img.resize(original_size, Image.LANCZOS)
        
        t_map_img = T.ToPILImage()(t_map.squeeze(0).cpu())
        t_map_img = t_map_img.resize(original_size, Image.LANCZOS)
        
        return dehazed_img, t_map_img, a_light.cpu().numpy().tolist()[0]

if __name__ == "__main__":
    # Example usage
    # engine = DehazeInference("outputs/checkpoints/best_model.pth")
    # engine.predict("path/to/hazy.jpg")
    pass
