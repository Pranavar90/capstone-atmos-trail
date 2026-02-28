import torch
from torch.cuda.amp import autocast
from models.mamba_arch import MambaDehaze

model = MambaDehaze(img_size=256, embed_dim=64, n_layers=4).cuda()
dummy = torch.rand(4, 3, 256, 256).cuda()

with autocast():
    out = model(dummy)
    print("Initial out contains NaN?:", torch.isnan(out).any().item())
    
    loss = out.mean()

loss.backward()

hasnans = False
for name, p in model.named_parameters():
    if p.grad is not None and torch.isnan(p.grad).any():
        print(f"NaN grad in {name}")
        hasnans = True

print("Has NaN grads?:", hasnans)
