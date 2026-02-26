import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import AttentionUNetDehaze, PhysicsReconstruction, DehazingLoss
import numpy as np
from torchvision.models import vgg16
from math import log10

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return self.mse(x_vgg, y_vgg)

def fast_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * log10(1.0 / torch.sqrt(mse).item())

class DehazeTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = AttentionUNetDehaze().to(self.device)
        self.physics = PhysicsReconstruction().to(self.device)
        self.criterion = DehazingLoss(
            w_phys=config.get('physics_loss_weight', 0.2)
        )
        self.perceptual = PerceptualLoss().to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.scaler = GradScaler(enabled=config['use_mixed_precision'])
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_psnr': []}
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc="Training")
        
        for i, (hazy, clear) in enumerate(pbar):
            hazy, clear = hazy.to(self.device), clear.to(self.device)
            
            with autocast(enabled=self.config['use_mixed_precision']):
                pred_t, pred_a = self.model(hazy)
                pred_j = self.physics(hazy, pred_t, pred_a)
                
                loss = self.criterion(pred_j, clear, hazy, pred_t, pred_a)
                loss += 0.1 * self.perceptual(pred_j, clear)
                
                # Gradient Accumulation
                loss = loss / self.config['grad_accum_steps']
            
            self.scaler.scale(loss).backward()
            
            if (i + 1) % self.config['grad_accum_steps'] == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
            total_loss += loss.item() * self.config['grad_accum_steps']
            pbar.set_postfix({'loss': total_loss / (i + 1)})
            
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_psnr = 0
        with torch.no_grad():
            for hazy, clear in val_loader:
                hazy, clear = hazy.to(self.device), clear.to(self.device)
                pred_t, pred_a = self.model(hazy)
                pred_j = self.physics(hazy, pred_t, pred_a)
                total_psnr += fast_psnr(pred_j, clear)
                
        return total_psnr / len(val_loader)

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)

    def plot_history(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.title('Loss Curve')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['val_psnr'], label='Val PSNR')
        plt.title('PSNR Curve')
        plt.legend()
        
        plt.savefig('outputs/plots/training_curves.png')
        plt.close()
