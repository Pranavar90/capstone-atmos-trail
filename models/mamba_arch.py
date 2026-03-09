"""
Phase B + C: End-to-End Vision Mamba (Vim) Dehazer
==================================================
World-first: Physics-Guided State Space Model for Single Image Dehazing.

Architecture:
  Input Image  -->  PatchEmbedding  -->  Bi-directional SSM Blocks
       -->  Spatial Unflatten  -->  CNN Refinement  -->  K(x) prediction
       -->  AOD Physics:  J(x) = K(x) * I(x) - K(x) + 1

Key design choices:
  - Official Triton `mamba_ssm` kernel for maximum hardware acceleration
  - Shifted Window Scanning (Swin-style) to preserve high-frequency local textures
  - Multi-Scale Physics Fusion (5 scales) for robust depth estimation
  - End-to-End AOD formulation: NO division, NO separate A/t(x) prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try importing Triton-accelerated Mamba. If missing, fail fast.
try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("Workstation version requires 'mamba_ssm'. Please install it on Linux/WSL2.")


# =============================================================================
# Shifted Window Utilities (Swin-inspired)
# =============================================================================

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# =============================================================================
# Shifted Window Bi-Directional Vision Mamba Block (Triton Accelerated)
# =============================================================================

class WindowedBiMambaBlock(nn.Module):
    """
    Processes the grid in localized windows with optional shifts.
    Inside each window, it flattens the pixels to 1D and runs a 
    Triton-accelerated Bi-Directional Mamba scan.
    
    This solves the issue of global Mamba losing extremely fine 
    local texture details, functioning similarly to Swin Transformers
    but with O(N) linear time per window.
    """
    def __init__(self, d_model, d_state=16, shift_size=0, window_size=8, dropout=0.1):
        super().__init__()
        self.shift_size = shift_size
        self.window_size = window_size
        self.d_model = d_model

        # Official Triton Mamba Kernels
        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )
        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2,
        )
        
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, H, W):
        """
        Args:
            x: (B, L, D) - 1D sequence
            H, W: Spatial dimensions
        Returns:
            (B, L, D) - updated sequence
        """
        B, L, C = x.shape
        assert L == H * W, "Sequence length must equal H * W"
        
        # 1. Reshape sequence to 2D
        x_2d = x.view(B, H, W, C)

        # 2. Cyclic Shift (if shift_size > 0)
        if self.shift_size > 0:
            shifted_x = torch.roll(x_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_2d

        # 3. Partition into Windows
        # shape: (B * num_windows, window_size, window_size, C)
        x_windows = window_partition(shifted_x, self.window_size)
        
        # Flatten windows into 1D sequences
        # shape: (B * num_windows, window_size*window_size, C)
        num_windows_b = x_windows.shape[0]
        x_seq = x_windows.view(num_windows_b, self.window_size * self.window_size, C)

        # 4. Bi-Directional Mamba Scan inside Windows
        # Forward Sweep
        y_fwd = self.forward_mamba(x_seq)
        
        # Backward Sweep
        x_rev = torch.flip(x_seq, dims=[1])
        y_bwd = self.backward_mamba(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])

        # Fuse
        y_fused = torch.cat([y_fwd, y_bwd], dim=-1)
        y_fused = self.fusion(y_fused)
        y_fused = self.drop(y_fused)

        # 5. Unflatten Windows
        y_windows = y_fused.view(num_windows_b, self.window_size, self.window_size, C)
        shifted_y = window_reverse(y_windows, self.window_size, H, W)

        # 6. Reverse Cyclic Shift
        if self.shift_size > 0:
            y_2d = torch.roll(shifted_y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            y_2d = shifted_y
            
        # 7. Reshape back to 1D and Apply Norm
        y_1d = y_2d.view(B, L, C)
        
        # Residual connection + norm
        out = self.norm(x + y_1d)
        
        return out


# =============================================================================
# Patch Embedding (Image → 1D Sequence)
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    Chops the input image into non-overlapping patches and projects each
    patch into the embedding dimension D.
    """
    def __init__(self, img_size=256, patch_size=8, in_channels=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Conv2d acts as the patch projection
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.proj(x)                          # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)           # (B, num_patches, D)
        x = x + self.pos_embed
        return x


# =============================================================================
# Multi-Scale Physics Fusion (Workstation Upgrade)
# =============================================================================

class MultiScalePhysicsFusion(nn.Module):
    """
    Decodes the Mamba feature sequence into 5 distinct spatial scales 
    (1/16, 1/8, 1/4, 1/2, 1) to capture both global atmospheric haze gradients
    and extremely fine local edge transmission details.
    
    All scales are upsampled, concatenated, and fused via a 1x1 conv 
    to form the final unified K(x) map.
    """
    def __init__(self, embed_dim=64, img_size=256, patch_size=8, out_channels=3):
        super().__init__()
        self.grid_size = img_size // patch_size   # usually 256//8 = 32 (1/8 scale)
        self.embed_dim = embed_dim

        # Project embedding back to spatial channels
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Scale Extractors
        # Origin scale out of Mamba is 1/8 if patch_size=8.
        # Let's derive 5 scales relative to full resolution.
        
        # Scale 1/16 (Downsample from 1/8)
        self.scale_16 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )
        
        # Scale 1/8 (Direct from Mamba)
        self.scale_8_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Scale 1/4 (Upsample from 1/8)
        self.scale_4 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.GELU()
        )
        
        # Scale 1/2 (Upsample from 1/4)
        self.scale_2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU()
        )
        
        # Scale 1 (Full Res) (Upsample from 1/2)
        self.scale_1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU()
        )
        
        # Final Fusion: concatenates all 5 scales (upsampled to full res)
        # channels: embed_dim + embed_dim + embed_dim//2 + embed_dim//4 + embed_dim//4
        # For embed_dim=64: 64 + 64 + 32 + 16 + 16 = 192 channels
        total_channels = embed_dim + embed_dim + (embed_dim // 2) + (embed_dim // 4) + (embed_dim // 4)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x, input_size):
        """
        Args:
            x: (B, num_patches, embed_dim)
            input_size: (H, W) of the original image
        Returns:
            K: (B, 3, H, W) — the unified physics parameter fused from 5 scales
        """
        B = x.shape[0]
        x = self.proj(x)

        # Unflatten: (B, L, D) → (B, D, grid, grid)
        # Assuming grid_size dynamically matches patch_size
        H, W = input_size
        patch_h, patch_w = H // 8, W // 8
        x = x.transpose(1, 2).reshape(B, self.embed_dim, patch_h, patch_w)

        # Extract features at 5 scales
        feat_16 = self.scale_16(x)
        feat_8 = self.scale_8_conv(x)
        feat_4 = self.scale_4(x)
        feat_2 = self.scale_2(feat_4)
        feat_1 = self.scale_1(feat_2)

        # Upsample all features to full resolution (H, W)
        feat_16_up = F.interpolate(feat_16, size=(H, W), mode='bilinear', align_corners=False)
        feat_8_up = F.interpolate(feat_8, size=(H, W), mode='bilinear', align_corners=False)
        feat_4_up = F.interpolate(feat_4, size=(H, W), mode='bilinear', align_corners=False)
        feat_2_up = F.interpolate(feat_2, size=(H, W), mode='bilinear', align_corners=False)
        # feat_1 is already at H,W if patch_size=8 and calculations align, 
        # but we interpolate to be mathematically safe.
        feat_1_up = F.interpolate(feat_1, size=(H, W), mode='bilinear', align_corners=False)

        # Concatenate and fuse
        concat_feats = torch.cat([feat_16_up, feat_8_up, feat_4_up, feat_2_up, feat_1_up], dim=1)
        K = self.fusion(concat_feats)
        
        return K


# =============================================================================
# Main Model: MambaDehaze (End-to-End AOD Physics)
# =============================================================================

class MambaDehaze(nn.Module):
    """
    End-to-End Vision Mamba Dehazer (Workstation Upgrade).

    Architecture:
        PatchEmbedding → N × ShiftedWindowMambaBlocks → MultiScalePhysicsFusion → K(x)
        J(x) = K(x) * I(x) - K(x) + 1
    """
    def __init__(self, img_size=256, patch_size=8, embed_dim=64,
                 d_state=16, window_size=8, n_layers=4, dropout=0.1):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        # Stage 1: Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )

        # Stage 2: Stacked Triton Mamba Blocks with alternating Shifted Windows
        self.vim_blocks = nn.ModuleList()
        for i in range(n_layers):
            # Alternate shift sizes (0 then window_size//2)
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            self.vim_blocks.append(
                WindowedBiMambaBlock(
                    d_model=embed_dim, 
                    d_state=d_state, 
                    shift_size=shift_size,
                    window_size=window_size,
                    dropout=dropout
                )
            )

        # Stage 3: Multi-Scale Physics Fusion → K(x)
        self.spatial_head = MultiScalePhysicsFusion(
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            out_channels=3
        )

    def forward(self, hazy_input):
        """
        Args:
            hazy_input: (B, 3, H, W) — the hazy image tensor

        Returns:
            j_pred: (B, 3, H, W) — the dehazed output (End-to-End AOD physics)
        """
        B, C, H, W = hazy_input.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        # --- Mamba Backbone ---
        x = self.patch_embed(hazy_input)               # (B, L, D)

        for block in self.vim_blocks:
            x = block(x, patch_h, patch_w)             # Pass spatial dimensions for windowing

        # --- Predict K(x) ---
        K = self.spatial_head(x, (H, W))               # (B, 3, H, W)

        # --- End-to-End AOD Physics ---
        # J(x) = K(x) * I(x) - K(x) + 1
        j_pred = K * hazy_input - K + 1.0

        # Clamp output to valid image range
        j_pred = torch.clamp(j_pred, 0.0, 1.0)

        return j_pred


# =============================================================================
# Quick test & parameter count
# =============================================================================

if __name__ == "__main__":
    # Note: Will crash if mamba_ssm is not installed.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaDehaze(img_size=256, embed_dim=64, n_layers=4).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Workstation MambaDehaze Trainable Parameters: {params:,}")
    
    # Static verification logic handles mathematical dimension checks.
