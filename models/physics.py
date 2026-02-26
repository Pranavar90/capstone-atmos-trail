import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsReconstruction(nn.Module):
    """
    Implements J(x) = (I(x) - A) / max(t(x), epsilon) + A
    """
    def __init__(self, epsilon=0.1):
        super(PhysicsReconstruction, self).__init__()
        self.epsilon = epsilon

    def forward(self, hazy, transmission, atmospheric_light):
        # hazy: (B, 3, H, W)
        # transmission: (B, 1, H, W)
        # atmospheric_light: (B, 3, 1, 1) or (B, 3)
        
        if len(atmospheric_light.shape) == 2:
            atmospheric_light = atmospheric_light.unsqueeze(-1).unsqueeze(-1)
            
        t = torch.clamp(transmission, min=self.epsilon)
        dehazed = (hazy - atmospheric_light) / t + atmospheric_light
        return torch.clamp(dehazed, 0, 1)

class DehazingLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=1.0, w_perceptual=0.1, w_phys=0.2):
        super(DehazingLoss, self).__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_perceptual = w_perceptual
        self.w_phys = w_phys
        self.l1 = nn.L1Loss()
        
    def forward(self, pred_j, gt_j, hazy=None, pred_t=None, pred_a=None):
        loss_l1 = self.l1(pred_j, gt_j)
        
        # SSIM and Perceptual will be added in trainer with specialized modules
        total_loss = self.w_l1 * loss_l1
        
        # Physics consistency: I - (J*t + A(1-t))
        if hazy is not None and pred_t is not None and pred_a is not None:
            if len(pred_a.shape) == 2:
                pred_a = pred_a.unsqueeze(-1).unsqueeze(-1)
            reconstructed_i = pred_j * pred_t + pred_a * (1 - pred_t)
            loss_phys = self.l1(reconstructed_i, hazy)
            total_loss += self.w_phys * loss_phys
            
        return total_loss
