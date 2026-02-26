import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from training.trainer import DehazeTrainer
import random

# ========================
# Adjustable Hyperparameters
# ========================
EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
IMAGE_SIZE = 256
PHYSICS_LOSS_WEIGHT = 0.2
PERCEPTUAL_LOSS_WEIGHT = 0.1
USE_MIXED_PRECISION = True
GRAD_ACCUM_STEPS = 2
NUM_WORKERS = 4
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

class DehazeDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.hazy_dir = os.path.join(root_dir, split, "hazy")
        self.clear_dir = os.path.join(root_dir, split, "clear")
        self.images = [f for f in os.listdir(self.hazy_dir) if f.endswith(('.png', '.jpg'))]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.hazy_dir, self.images[idx])
        clear_path = os.path.join(self.clear_dir, self.images[idx])
        
        hazy = Image.open(hazy_path).convert("RGB")
        clear = Image.open(clear_path).convert("RGB")
        
        if self.transform:
            hazy = self.transform(hazy)
            clear = self.transform(clear)
            
        return hazy, clear

def run_training():
    config = {
        'lr': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'physics_loss_weight': PHYSICS_LOSS_WEIGHT,
        'perceptual_loss_weight': PERCEPTUAL_LOSS_WEIGHT,
        'use_mixed_precision': USE_MIXED_PRECISION,
        'grad_accum_steps': GRAD_ACCUM_STEPS
    }
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    # Augmentations for training
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])
    
    train_ds = DehazeDataset("data/processed", split="train", transform=train_transform)
    val_ds = DehazeDataset("data/processed", split="val", transform=transform)
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    print(f"Dataset Size: Train={len(train_ds)}, Val={len(val_ds)}")
    
    trainer = DehazeTrainer(config)
    
    best_psnr = 0
    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss = trainer.train_epoch(train_loader)
        val_psnr = trainer.validate(val_loader)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val PSNR: {val_psnr:.2f} | Time: {epoch_time:.2f}s")
        
        trainer.history['train_loss'].append(train_loss)
        trainer.history['val_psnr'].append(val_psnr)
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            trainer.save_checkpoint("outputs/checkpoints/best_model.pth")
            print(f"New best model saved with PSNR: {best_psnr:.2f}")
            
        trainer.save_checkpoint("outputs/checkpoints/last_model.pth")
        trainer.plot_history()
        
        # GPU Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    import time
    if not os.path.exists("data/processed/train/hazy"):
        print("Data not processed yet. Please run process_data.py first.")
    else:
        run_training()
