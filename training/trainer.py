"""
Mamba Dehazer Trainer
=====================
Training loop for the End-to-End Vision Mamba dehazer.
Features:
  - Mixed precision (torch.cuda.amp)
  - Aggressive gradient clipping (clip_grad_norm_ = 0.5)
  - CosineAnnealingLR with 5-epoch linear warmup
  - ConvNeXt-based contrastive loss
  - VRAM monitoring
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from math import log10

from models.mamba_arch import MambaDehaze
from training.losses import MambaDehazeLoss


def fast_psnr(img1, img2):
    """Fast PSNR calculation on GPU tensors."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * log10(1.0 / torch.sqrt(mse).item())


def fast_ssim_metric(img1, img2):
    """Quick SSIM metric (not differentiable, just for logging)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = img1.mean(dim=[2, 3], keepdim=True)
    mu2 = img2.mean(dim=[2, 3], keepdim=True)
    sigma1_sq = ((img1 - mu1) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma2_sq = ((img2 - mu2) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=[2, 3], keepdim=True)
    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_val.mean().item()


class WarmupCosineScheduler:
    """
    Linear warmup for first `warmup_epochs` epochs, then cosine annealing.
    Critical for SSM stability — prevents early gradient explosions.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1.0 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * scale)


class MambaDehazeTrainer:
    """
    Full training pipeline for the End-to-End Vision Mamba Dehazer.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            print(f"[GPU] {torch.cuda.get_device_name(0)}")
            print(f"[GPU] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        # --- Model ---
        self.model = MambaDehaze(
            img_size=config.get('image_size', 256),
            embed_dim=config.get('embed_dim', 64),
            d_state=config.get('d_state', 16),
            n_layers=config.get('n_layers', 4),
            dropout=config.get('dropout', 0.1)
        ).to(self.device)

        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Model] MambaDehaze — {params:,} trainable parameters")

        # Optimized weights for stability: prioritize L1/SSIM over noisy perceptual contrastive
        self.criterion = MambaDehazeLoss(
            w_l1=config.get('w_l1', 1.0),
            w_ssim=config.get('w_ssim', 0.8),
            w_cr=config.get('w_cr', 0.01),
            w_perc=config.get('w_perc', 0.05)
        ).to(self.device)

        # --- Optimizer ---
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # --- Scheduler (Warmup + Cosine) ---
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=config.get('warmup_epochs', 5),
            total_epochs=config['epochs']
        )

        # --- Mixed Precision ---
        self.scaler = GradScaler('cuda', enabled=config.get('use_mixed_precision', True))
        self.grad_clip_norm = config.get('grad_clip_norm', 0.5)

        # --- History ---
        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_psnr': [], 'val_ssim': [],
            'lr': [], 'vram_mb': [],
            'perceptual_loss': []
        }

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        loss_components = {'l1': 0, 'ssim': 0, 'contrastive': 0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

        for i, (hazy, clear) in enumerate(pbar):
            hazy = hazy.to(self.device, non_blocking=True)
            clear = clear.to(self.device, non_blocking=True)

            with autocast('cuda', enabled=self.config.get('use_mixed_precision', True)):
                j_pred = self.model(hazy)
                loss, losses = self.criterion(j_pred, clear, hazy)

                # Gradient accumulation
                loss = loss / self.config.get('grad_accum_steps', 1)

            self.scaler.scale(loss).backward()

            if (i + 1) % self.config.get('grad_accum_steps', 1) == 0:
                # CRITICAL: Extreme gradient clipping for SSM stability
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += losses['total']
            for key in loss_components:
                loss_components[key] += losses.get(key, 0)

            pbar.set_postfix({
                'loss': f"{total_loss / (i+1):.4f}",
                'perc': f"{loss_components.get('perceptual', 0) / (i+1):.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

        n = len(train_loader)
        avg_loss = total_loss / n

        # Track perceptual loss average
        self.history['perceptual_loss'].append(
            loss_components.get('perceptual', 0) / n
        )

        # Log VRAM usage
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
            vram_gb = vram_mb / 1024
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.history['vram_mb'].append(vram_mb)
            print(f"  [VRAM] Peak: {vram_gb:.3f} GB / {total_vram_gb:.1f} GB")

        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        total_loss = 0.0
        n = 0

        for hazy, clear in val_loader:
            hazy = hazy.to(self.device, non_blocking=True)
            clear = clear.to(self.device, non_blocking=True)

            with autocast('cuda', enabled=self.config.get('use_mixed_precision', True)):
                j_pred = self.model(hazy)

            j_pred_f32 = j_pred.float()
            clear_f32 = clear.float()

            total_psnr += fast_psnr(j_pred_f32, clear_f32)
            total_ssim += fast_ssim_metric(j_pred_f32, clear_f32)
            n += 1

        avg_psnr = total_psnr / max(n, 1)
        avg_ssim = total_ssim / max(n, 1)
        return avg_psnr, avg_ssim

    def save_checkpoint(self, path, epoch, best_psnr):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_psnr': best_psnr
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint.get('epoch', 0), checkpoint.get('best_psnr', 0)

    def plot_history(self):
        os.makedirs('outputs/plots', exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss curve
        axes[0].plot(self.history['train_loss'], label='Train Loss', color='#FF6B6B')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # PSNR curve
        axes[1].plot(self.history['val_psnr'], label='Val PSNR', color='#4ECDC4')
        axes[1].set_title('Validation PSNR (dB)')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # SSIM curve
        axes[2].plot(self.history['val_ssim'], label='Val SSIM', color='#45B7D1')
        axes[2].set_title('Validation SSIM')
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('outputs/plots/mamba_training_curves.png', dpi=150)
        plt.close()
