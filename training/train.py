"""
Mamba Dehazer Training Script
==============================
Entry point for training the End-to-End Vision Mamba dehazer.
Uses domain-randomized augmentations and the new MambaDehazeTrainer.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import time
import random

from training.trainer import MambaDehazeTrainer
from training.augmentations import HazeDomainRandomization
from torchvision import transforms

# ========================
# Hyperparameters
# ========================
EPOCHS = 50
LEARNING_RATE = 1e-5      # Flat schedule for Physics V7 (Zero-Divergence)
BATCH_SIZE = 48           # Optimized for regularization & 48GB VRAM
IMAGE_SIZE = 256
USE_MIXED_PRECISION = True
GRAD_ACCUM_STEPS = 1      # 64 is a solid effective batch; no accumulation needed
NUM_WORKERS = 8           # Increasing workers for high throughput
SEED = 42
# Data root: WSL native ext4 filesystem (fast!) 
DATA_ROOT = os.path.expanduser("~/capstone-data/processed")
WARMUP_EPOCHS = 0         # Disabled warmup to prevent LR spikes
EMBED_DIM = 128           # Restoration capacity
N_LAYERS = 6              # Restoration depth
D_STATE = 16              # 16 states per channel
GRAD_CLIP_NORM = 0.1     # Extreme clipping to prevent any SSM spikes in V8

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MambaDehazeDataset(Dataset):
    """
    Dataset that uses HazeDomainRandomization for training
    and simple resize+ToTensor for validation.
    """
    def __init__(self, root_dir, split="train", image_size=256, augment=True):
        self.hazy_dir = os.path.join(root_dir, split, "hazy")
        self.clear_dir = os.path.join(root_dir, split, "clear")
        self.images = sorted([
            f for f in os.listdir(self.hazy_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.augment = augment

        if augment:
            self.augmentor = HazeDomainRandomization(
                image_size=image_size, apply_noise=True, p=0.5
            )
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        for _ in range(5):  # retry up to 5 times on corrupted files
            try:
                hazy_path = os.path.join(self.hazy_dir, self.images[idx])
                clear_path = os.path.join(self.clear_dir, self.images[idx])

                hazy = Image.open(hazy_path).convert("RGB")
                clear = Image.open(clear_path).convert("RGB")

                if self.augment:
                    hazy_tensor, clear_tensor = self.augmentor(hazy, clear)
                else:
                    hazy_tensor = self.transform(hazy)
                    clear_tensor = self.transform(clear)

                return hazy_tensor, clear_tensor

            except (OSError, Exception) as e:
                print(f"\n  [WARN] Skipping corrupted image '{self.images[idx]}': {e}")
                idx = random.randint(0, len(self.images) - 1)

        raise RuntimeError(f"Could not load a valid image after 5 retries at idx={idx}")


def run_training():
    config = {
        'lr': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'image_size': IMAGE_SIZE,
        'use_mixed_precision': USE_MIXED_PRECISION,
        'grad_accum_steps': GRAD_ACCUM_STEPS,
        'warmup_epochs': WARMUP_EPOCHS,
        'embed_dim': EMBED_DIM,
        'n_layers': N_LAYERS,
        'd_state': D_STATE,
        'grad_clip_norm': GRAD_CLIP_NORM,
        'weight_decay': 1e-4,
        'dropout': 0.1,
        'w_l1': 1.0,
        'w_ssim': 0.8,    # Increased for structural priority
        'w_cr': 0.01,     # Lowered for initial stability
        'w_perc': 0.05,    # Lowered for initial stability
    }

    # --- Datasets ---
    train_ds = MambaDehazeDataset(
        DATA_ROOT, split="train",
        image_size=IMAGE_SIZE, augment=True
    )
    val_ds = MambaDehazeDataset(
        DATA_ROOT, split="val",
        image_size=IMAGE_SIZE, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"=" * 60)
    print(f" End-to-End Vision Mamba Dehazer — Training")
    print(f"=" * 60)
    print(f"  Dataset:   Train={len(train_ds)} | Val={len(val_ds)}")
    print(f"  Model:     embed_dim={EMBED_DIM}, layers={N_LAYERS}, d_state={D_STATE}")
    print(f"  Training:  epochs={EPOCHS}, bs={BATCH_SIZE}, accum={GRAD_ACCUM_STEPS}")
    print(f"  Schedule:  warmup={WARMUP_EPOCHS} epochs, then cosine decay")
    print(f"  Clip:      grad_norm={GRAD_CLIP_NORM}")
    print(f"=" * 60)

    # --- Trainer ---
    trainer = MambaDehazeTrainer(config)

    # Resume from checkpoint if available
    start_epoch = 0
    best_psnr = 0.0
    checkpoint_path = "outputs/checkpoints/mamba_last.pth"

    if os.path.exists(checkpoint_path):
        print(f"\n[Resume] Loading checkpoint: {checkpoint_path}")
        start_epoch, best_psnr = trainer.load_checkpoint(checkpoint_path)
        start_epoch += 1
        print(f"[Resume] Continuing from epoch {start_epoch}, best PSNR={best_psnr:.2f}")

    # --- Training Loop ---
    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()

        # Update learning rate
        trainer.scheduler.step(epoch)
        lr = trainer.optimizer.param_groups[0]['lr']

        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)

        # Validate
        val_psnr, val_ssim = trainer.validate(val_loader, epoch)

        epoch_time = time.time() - start_time

        # Log
        trainer.history['train_loss'].append(train_loss)
        trainer.history['val_psnr'].append(val_psnr)
        trainer.history['val_ssim'].append(val_ssim)
        trainer.history['lr'].append(lr)

        print(f"\n  Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | "
              f"PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | "
              f"LR: {lr:.2e} | Time: {epoch_time:.1f}s")

        # Save best
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            trainer.save_checkpoint("outputs/checkpoints/mamba_best.pth", epoch, best_psnr)
            print(f"  ★ New best model saved — PSNR: {best_psnr:.2f} dB")

        # Save last
        trainer.save_checkpoint(checkpoint_path, epoch, best_psnr)
        trainer.plot_history()

        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f" Training Complete — Best PSNR: {best_psnr:.2f} dB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    if not os.path.exists(os.path.join(DATA_ROOT, "train", "hazy")):
        print(f"ERROR: Data not found at '{DATA_ROOT}/train/hazy'.")
        print("Run: rsync -avh --progress /mnt/c/VU22CSEN0101728/capstone-atmos-trail/data/processed/ ~/capstone-data/processed/")
    else:
        run_training()
