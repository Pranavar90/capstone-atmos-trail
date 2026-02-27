from .arch import AttentionUNetDehaze
from .physics import PhysicsReconstruction, DehazingLoss
import torch

def get_model_summary(model, input_size=(1, 3, 256, 256)):
    print(f"Model: {model.__class__.__name__}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    
    dummy_input = torch.randn(input_size)
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = dummy_input.cuda()
    
    with torch.no_grad():
        t, a = model(dummy_input)
        print(f"Input: {dummy_input.shape}")
        print(f"Output T: {t.shape}")
        print(f"Output A: {a.shape}")
        
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Max GPU Memory Used: {mem:.2f} MB")

if __name__ == "__main__":
    model = AttentionUNetDehaze()
    get_model_summary(model)
