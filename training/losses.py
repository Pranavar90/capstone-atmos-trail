"""
Phase D: Loss Functions for End-to-End Vision Mamba Dehazer
===========================================================
Implements:
  1. L1 Loss (pixel-level)
  2. SSIM Loss (structural integrity)
  3. ConvNeXt-based Contrastive Regularization (AECR-style, replaces VGG-19)

ConvNeXt-Tiny is used as a frozen semantic feature extractor to compute
contrastive distances: minimise distance(dehazed, clear) and maximise
distance(dehazed, hazy).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


# =============================================================================
# ConvNeXt Perceptual Loss (modern 2022 architecture — replaces legacy VGG)
# =============================================================================

class ConvNeXtPerceptualLoss(nn.Module):
    """
    Perceptual loss using frozen ConvNeXt-Tiny feature stages.
    Uses the same backbone as ContrastiveRegularizationLoss to share compute.
    Computes multi-scale L1 feature distance (stage0, stage1, stage2).

    Why ConvNeXt over VGG:
      - ConvNeXt (2022) captures richer semantic + structural features
      - Already loaded in the pipeline (no extra VRAM for a second network)
      - ImageNet-21K pretraining gives far better perceptual alignment
    """
    def __init__(self, extractor: 'ConvNeXtFeatureExtractor' = None):
        super().__init__()
        # Accept a shared extractor to avoid loading ConvNeXt twice
        self._shared = extractor  # will be set after ContrastiveRegularizationLoss is created

    def set_extractor(self, extractor):
        self._shared = extractor

    def forward(self, pred, target):
        assert self._shared is not None, "Call set_extractor() before using ConvNeXtPerceptualLoss"
        with torch.no_grad():
            feats_target = self._shared(target)
        feats_pred = self._shared(pred)
        loss = sum(F.l1_loss(fp, ft.detach())
                   for fp, ft in zip(feats_pred, feats_target))
        return loss / len(feats_pred)


# =============================================================================
# SSIM Loss
# =============================================================================

def _gaussian_kernel(size=11, sigma=1.5, channels=3):
    """Create a Gaussian kernel for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    g = g / g.sum()
    kernel = g.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Measure loss.
    Returns 1 - SSIM(pred, target) so that minimising the loss maximises SSIM.
    """
    def __init__(self, window_size=11, sigma=1.5, channels=3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer(
            'kernel', _gaussian_kernel(window_size, sigma, channels)
        )
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred, target):
        # Ensure kernel is on the right device
        kernel = self.kernel.to(pred.device, dtype=pred.dtype)
        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, kernel, padding=pad, groups=self.channels)
        mu_target = F.conv2d(target, kernel, padding=pad, groups=self.channels)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=self.channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, kernel, padding=pad, groups=self.channels) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, kernel, padding=pad, groups=self.channels) - mu_pred_target

        ssim_map = ((2 * mu_pred_target + self.C1) * (2 * sigma_pred_target + self.C2)) / \
                   ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))

        return 1.0 - ssim_map.mean()


# =============================================================================
# ConvNeXt-based Contrastive Regularization Loss (replaces VGG-19)
# =============================================================================

class ConvNeXtFeatureExtractor(nn.Module):
    """
    Frozen ConvNeXt-Tiny feature extractor for semantic contrastive loss.
    Replaces VGG-19 with a modern 2022+ architecture that captures far
    richer semantic and structural features.
    """
    def __init__(self):
        super().__init__()
        # Load ConvNeXt-Tiny with pretrained ImageNet weights
        convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # Extract multi-scale features from the hierarchical stages
        # Stage 0: low-level edges/textures, Stage 1-2: mid/high-level semantics
        self.stage0 = convnext.features[0:2]   # stem + stage 1
        self.stage1 = convnext.features[2:4]   # stage 2
        self.stage2 = convnext.features[4:6]   # stage 3

        # Freeze ALL weights — we only use ConvNeXt for feature extraction
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, x):
        """
        Extract multi-scale features.
        Args:
            x: (B, 3, H, W) image tensor normalised to [0, 1]
        Returns:
            list of feature tensors at 3 scales
        """
        # ImageNet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        f0 = self.stage0(x)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)
        return [f0, f1, f2]


class ContrastiveRegularizationLoss(nn.Module):
    """
    AECR-style Contrastive Regularization using ConvNeXt-Tiny features.

    Minimise: distance(features(J_pred), features(J_target))   [positive pair]
    Maximise: distance(features(J_pred), features(I_hazy))     [negative pair]

    This prevents the model from hallucinating: if the dehazed output
    looks similar to the hazy input (bad), the loss penalises it;
    if it looks similar to the clear ground truth (good), the loss rewards it.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.feature_extractor = ConvNeXtFeatureExtractor()
        self.margin = margin

    def _cosine_distance(self, f1, f2):
        """Compute mean cosine distance across spatial dimensions."""
        f1_flat = f1.flatten(2)   # (B, C, H*W)
        f2_flat = f2.flatten(2)
        cos_sim = F.cosine_similarity(f1_flat, f2_flat, dim=1)  # (B, H*W)
        return 1.0 - cos_sim.mean()

    def forward(self, j_pred, j_target, i_hazy):
        """
        Args:
            j_pred:   (B, 3, H, W) dehazed output
            j_target: (B, 3, H, W) ground truth clear
            i_hazy:   (B, 3, H, W) hazy input
        Returns:
            contrastive loss scalar
        """
        with torch.no_grad():
            feats_target = self.feature_extractor(j_target)
            feats_hazy = self.feature_extractor(i_hazy)

        feats_pred = self.feature_extractor(j_pred)

        loss = 0.0
        for fp, ft, fh in zip(feats_pred, feats_target, feats_hazy):
            # Pull dehazed towards clear
            d_positive = self._cosine_distance(fp, ft.detach())
            # Push dehazed away from hazy
            d_negative = self._cosine_distance(fp, fh.detach())
            # Contrastive: minimise positive, maximise negative
            loss += d_positive + F.relu(self.margin - d_negative)

        return loss / len(feats_pred)


# =============================================================================
# Unified MambaDehazing Loss
# =============================================================================

class MambaDehazeLoss(nn.Module):
    """
    Combined loss for the End-to-End Vision Mamba Dehazer.

    Total = w_l1 * L1 + w_ssim * SSIM + w_cr * ContrastiveReg + w_perc * ConvNeXtPerceptual
    """
    def __init__(self, w_l1=1.0, w_ssim=0.5, w_cr=0.1, w_perc=0.1):
        super().__init__()
        self.w_l1   = w_l1
        self.w_ssim = w_ssim
        self.w_cr   = w_cr
        self.w_perc = w_perc

        self.l1_loss   = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.cr_loss   = ContrastiveRegularizationLoss()

        # Share ConvNeXt extractor to avoid loading the backbone twice
        self.perc_loss = ConvNeXtPerceptualLoss()
        self.perc_loss.set_extractor(self.cr_loss.feature_extractor)

    def forward(self, j_pred, j_target, i_hazy):
        """
        Args:
            j_pred:   (B, 3, H, W) model output
            j_target: (B, 3, H, W) ground truth
            i_hazy:   (B, 3, H, W) input hazy image
        Returns:
            total_loss, dict of individual losses
        """
        # Safety clamp to prevent NaN in SSIM/ConvNeXt during mixed precision
        j_pred   = torch.clamp(j_pred,   0.0, 1.0)
        j_target = torch.clamp(j_target, 0.0, 1.0)
        i_hazy   = torch.clamp(i_hazy,   0.0, 1.0)

        loss_l1   = self.l1_loss(j_pred, j_target)
        loss_ssim = self.ssim_loss(j_pred, j_target)
        loss_cr   = self.cr_loss(j_pred, j_target, i_hazy)
        loss_perc = self.perc_loss(j_pred, j_target)

        total = (self.w_l1   * loss_l1
               + self.w_ssim * loss_ssim
               + self.w_cr   * loss_cr
               + self.w_perc * loss_perc)

        losses = {
            'l1':          loss_l1.item(),
            'ssim':        loss_ssim.item(),
            'contrastive': loss_cr.item(),
            'perceptual':  loss_perc.item(),
            'total':       total.item()
        }

        return total, losses
