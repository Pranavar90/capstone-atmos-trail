"""
Phase B + C: End-to-End Vision Mamba (Vim) Dehazer
==================================================
World-first: Physics-Guided State Space Model for Single Image Dehazing.

Architecture:
  Input Image  -->  PatchEmbedding  -->  Bi-directional SSM Blocks
       -->  Spatial Unflatten  -->  CNN Refinement  -->  K(x) prediction
       -->  AOD Physics:  J(x) = K(x) * I(x) - K(x) + 1

Key design choices:
  - Pure PyTorch SSM (no mamba-ssm CUDA kernels) for Windows compatibility
  - Bi-directional scanning to preserve 2D spatial awareness
  - End-to-End AOD formulation: NO division, NO separate A/t(x) prediction
  - Target VRAM: <3.5 GB on RTX 3050
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# =============================================================================
# Pure-PyTorch State Space Model Block (S4-inspired, no CUDA kernels)
# =============================================================================

class S4Block(nn.Module):
    """
    Simplified State Space Sequence Model (S4-inspired).
    Implements a discretised continuous-time SSM:
        x_{k+1} = A_bar * x_k + B_bar * u_k
        y_k     = C * x_k + D * u_k

    Uses diagonal state matrix for efficiency (like S4D / DSS).
    Purely PyTorch — no custom CUDA kernels required.
    """
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Learnable SSM parameters (diagonal state matrix per channel)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.B_param = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C_param = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.D = nn.Parameter(torch.ones(d_model))  # skip connection

        # Learnable discretisation step
        self.log_dt = nn.Parameter(torch.randn(d_model) * 0.01 - 4.0)

        # Input projection + gating (SiLU like Mamba)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """HiPPO-inspired initialisation for the state matrix."""
        with torch.no_grad():
            # Initialise A as negative values (stable dynamics)
            # A_log represents the log of the ABSOLUTE VALUE of A
            n = self.d_state
            A_mag = torch.arange(1, n + 1, dtype=torch.float32)
            self.A_log.copy_(
                A_mag.unsqueeze(0).expand(self.d_model, -1).log().clamp(min=-7)
            )

    def _discretise(self):
        """Zero-Order Hold (ZOH) discretisation."""
        dt = torch.exp(self.log_dt).clamp(min=1e-4, max=1.0)   # (d_model,)
        A = -torch.exp(self.A_log)                               # (d_model, d_state)

        # A_bar = exp(A * dt)
        A_bar = torch.exp(A * dt.unsqueeze(-1))                  # (d_model, d_state)
        return A_bar, dt

    def _ssm_scan(self, u):
        """
        Full-Rank Per-Channel SSM Scan (V6).
        Calculates the kernel h(t) = sum_n C_n * B_n * A_n^t
        
        Args:
            u: (B, L, D)  input sequence
        Returns:
            y: (B, L, D)  output sequence
        """
        B_batch, L, D = u.shape
        A_bar, dt = self._discretise()  # (D, N), (D,)

        # Stability: Ensure A_bar is strictly decaying
        A_bar = A_bar.float().clamp(min=1e-8, max=1.0 - 1e-8)  # (D, N)
        B_weight = self.B_param.float()                         # (D, N)
        C_weight = self.C_param.float()                         # (D, N)
        
        # --- Build causal convolution kernel (Full Math) ---
        # h[d, t] = sum_n (C[d,n] * B[d,n] * A_bar[d,n]^t) * dt[d]
        t_idx = torch.arange(L, device=u.device, dtype=torch.float32)  # (L,)
        log_A = torch.log(A_bar) # (D, N)
        
        # A_powers shape: (L, D, N)
        A_powers = torch.exp(t_idx[:, None, None] * log_A[None, :, :]) 

        # Combined weight: C * B * dt
        # Shape: (D, N)
        cb_dt = C_weight * B_weight * dt.unsqueeze(-1)
        
        # Multiply across state dimension N and sum
        # Kernel shape: (L, D)
        kernel_t_d = (A_powers * cb_dt[None, :, :]).sum(dim=-1)
        
        # Reshape to (D, 1, L) for depthwise conv
        kernel = kernel_t_d.T.unsqueeze(1).flip(-1) 

        # --- Depthwise causal conv over channels ---
        u_t = u.transpose(1, 2).contiguous().float()   # (B, D, L)
        u_padded = F.pad(u_t, (L - 1, 0))              # (B, D, 2L-1)
        
        y = F.conv1d(u_padded, kernel, groups=D)  # (B, D, L)
        y = y.transpose(1, 2).to(u.dtype)          # (B, L, D)

        return y

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        # Force EVERYTHING inside this block to FP32 to prevent Autocast NaNs
        x = x.to(torch.float32)
        residual = x
        x = self.norm(x)

        # Input projection + gating
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        z = F.silu(z)  # gate

        # SSM scan
        y = self._ssm_scan(x_proj)
        y = y * z  # gated output

        y = self.out_proj(y)
        y = self.dropout(y)

        # Stability: Final residual connection clamping
        return torch.clamp(y + residual, -10, 10)


# =============================================================================
# Bi-Directional Vision Mamba Block
# =============================================================================

class BiDirectionalVimBlock(nn.Module):
    """
    Processes the 1D patch sequence in both Forward and Backward directions,
    then fuses the results. This preserves spatial context that would be
    lost with a single-direction sweep.
    """
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.forward_ssm = S4Block(d_model, d_state, dropout)
        self.backward_ssm = S4Block(d_model, d_state, dropout)
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        # Forward sweep
        y_fwd = self.forward_ssm(x)

        # Backward sweep  (reverse → scan → reverse back)
        x_rev = torch.flip(x, dims=[1])
        y_bwd = self.backward_ssm(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])

        # Fuse forward + backward
        y = torch.cat([y_fwd, y_bwd], dim=-1)
        y = self.fusion(y)
        y = self.norm(y)

        return y


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
# Adaptive Physics Heads: Global Atmosphere + Spatial Mask
# =============================================================================

class GlobalAtmosphereHead(nn.Module):
    """
    Light neural network to "understand" the image atmosphere and assign channels.
    Predicts a 3-channel global color vector (Atmospheric Light hint).
    """
    def __init__(self, embed_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 3),
            nn.Sigmoid() # Constrain color assignment to (0, 1)
        )
        self._init_bias()

    def _init_bias(self):
        """Strict Identity Init: Forces C_atm to start at exactly ~1.0."""
        with torch.no_grad():
            # Identity: weights = 0, bias = high (for Sigmoid -> 1.0)
            # Find the linear layers in the sequential block
            for layer in self.mlp:
                if hasattr(layer, 'weight'):
                    nn.init.constant_(layer.weight, 0)
                if hasattr(layer, 'bias'):
                    nn.init.constant_(layer.bias, 4.0)

    def forward(self, x):
        """
        x: (B, L, D)
        returns: (B, 3, 1, 1)
        """
        # Global Average Pooling over tokens
        x_gap = x.mean(dim=1) # (B, D)
        c_atm = self.mlp(x_gap) # (B, 3)
        return c_atm.unsqueeze(-1).unsqueeze(-1)


class SpatialReconstruction(nn.Module):
    """
    Unflattens the 1D Mamba sequence back to 2D spatial grid.
    Outputs 4-channel tensor for:
      - 1 channel for K_mask(x) (Physics geometry)
      - 3 channels for Delta(x) (Neural refinement)
    """
    def __init__(self, embed_dim=64, img_size=256, patch_size=8, out_channels=4):
        super().__init__()
        self.grid_size = img_size // patch_size   # e.g., 32
        self.embed_dim = embed_dim

        # Project embedding back to spatial channels
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Multi-Path Reconstruction (Double-Barreled Upsampling)
        # Branch A: Sharp (Sub-Pixel Shuffle)
        # Using 256 channels as 4 * (8^2) = 256 for 4-channel effective output
        self.path_sharp = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=8),
            nn.GELU()
        )
        
        # Branch B: Smooth (Bicubic Baseline)
        self.path_smooth = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bicubic', align_corners=False),
            nn.Conv2d(embed_dim, 4, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Final Fusion: Merge Smooth + Sharp paths
        # out_channels = 4 (1 for K_mask, 3 for ΔJ_residual)
        self.fusion = nn.Conv2d(8, out_channels, kernel_size=3, padding=1)

        self._init_upsampler()
        self._init_refinement_to_zero()

    def _init_refinement_to_zero(self):
        """
        1. Zero-init weights so sharp/smooth paths don't jitter on Epoch 0.
        2. Set K_mask bias so Sigmoid(bias)*10 starts near 1.0 (identity).
        3. Set Delta bias to 0.
        """
        with torch.no_grad():
            # Zero ALL weights in fusion to ignore random sharp/smooth noise at start
            self.fusion.weight.fill_(0)
            
            # Delta branch (channels 1, 2, 3) -> Bias = 0
            if self.fusion.bias is not None:
                self.fusion.bias[1:].fill_(0)
            
            # K_mask branch (channel 0) -> Sigmoid(x)*10 ~= 1.0
            # logit(0.1) = ln(0.1/0.9) ~= -2.2
            if self.fusion.bias is not None:
                self.fusion.bias[0].fill_(-2.2)

    def _init_upsampler(self):
        """
        ICNR initialization for the PixelShuffle layer's convolution.
        Prevents checkerboard artifacts by ensuring the sub-pixels are 
        initialized to a smooth upscale.
        """
        conv = self.path_sharp[0]
        upscale_factor = 8
        out_channels = 4  # Target channels after shuffle (Mask + 3-ch Delta)
        
        with torch.no_grad():
            # Initial weight: (out_channels * r^2, in_channels, k, k)
            weight = conv.weight
            ni, ci, h, w = weight.shape
            
            # Sub-kernel: (out_channels, in_channels, k, k)
            new_shape = (out_channels, ci, h, w)
            sub_kernel = torch.zeros(new_shape)
            nn.init.kaiming_normal_(sub_kernel, mode='fan_in', nonlinearity='relu')
            
            # Repeat sub-kernel across the checkerboard grid
            kernel = sub_kernel.repeat(upscale_factor**2, 1, 1, 1)
            weight.copy_(kernel)
            
            # Reset bias
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, num_patches, embed_dim)
        Returns:
            map: (B, 4, H, W) -> [K_mask, ΔR, ΔG, ΔB]
        """
        B = x.shape[0]
        x = self.proj(x)

        # Unflatten: (B, L, D) → (B, D, grid, grid)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, self.grid_size, self.grid_size)

        # Multi-Path Reconstruction
        sharp = self.path_sharp(x)
        smooth = self.path_smooth(x)
        
        # Fuse branches (result is out_channels=4)
        out = self.fusion(torch.cat([sharp, smooth], dim=1))
        return out


# =============================================================================
# Main Model: MambaDehaze (End-to-End AOD Physics)
# =============================================================================

class MambaDehaze(nn.Module):
    """
    End-to-End Vision Mamba Dehazer.

    Architecture:
        PatchEmbedding → N × BiDirectionalVimBlock → SpatialReconstruction → K(x)
        J(x) = K(x) * I(x) - K(x) + 1

    Args:
        img_size: input resolution (default 256)
        patch_size: patch size for embedding (default 8)
        embed_dim: SSM embedding dimension (default 64)
        d_state: SSM hidden state dimension (default 16)
        n_layers: number of stacked Vim blocks (default 4)
        dropout: dropout rate
    """
    def __init__(self, img_size=256, patch_size=8, embed_dim=128,
                 d_state=16, n_layers=6, dropout=0.1):
        super().__init__()

        self.img_size = img_size

        # Stage 1: Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )

        # Stage 2: Stacked Bi-Directional Vim Blocks
        self.vim_blocks = nn.ModuleList([
            BiDirectionalVimBlock(embed_dim, d_state, dropout)
            for _ in range(n_layers)
        ])

        # Stage 3: Spatial Reconstruction → K_mask(x) + ΔJ(x)
        self.spatial_head = SpatialReconstruction(
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            out_channels=4  # 1 for K_mask, 3 for Delta
        )

        # Stage 4: Global Atmosphere Head → C_atm
        self.atm_head = GlobalAtmosphereHead(embed_dim=embed_dim)

    def forward(self, hazy_input, return_maps=False):
        """
        Adaptive Physics-Neural Hybrid Forward Pass (V3)
        ================================================
        1. Process input via Vision Mamba Backbone.
        2. Predict Global Atmosphere Color (C_atm).
        3. Predict Spatial Transmission Mask (K_mask) and Refinement (ΔJ).
        4. Hybrid Physics: K(x) = K_mask(x) * C_atm.
        5. Physics Coarse: J_aod = K*I - K + 1.
        6. Fuse: J_final = torch.clamp(J_aod + ΔJ, 0, 1).
        """
        # --- 1. Mamba Backbone ---
        x = self.patch_embed(hazy_input)               # (B, L, D)

        for block in self.vim_blocks:
            x = block(x)                                # (B, L, D)

        # --- 2. Adaptive Physics Heads ---
        # A. Global Color Vector (Assignable channels)
        c_atm = self.atm_head(x)                        # (B, 3, 1, 1)

        # B. Spatial Heads (Mask + Residuals)
        out = self.spatial_head(x)                      # (B, 4, H, W)

        # Rescale if needed
        if out.shape[2:] != hazy_input.shape[2:]:
            out = F.interpolate(out, size=hazy_input.shape[2:],
                                mode='bilinear', align_corners=False)

        # --- 3. Composite Physics Formulation (Force FP32) ---
        # Moving to FP32 for sensitive physics math to prevent mixed-precision divergence
        hazy_f32 = hazy_input.float()
        out_f32 = out.float()
        c_atm_f32 = c_atm.float()

        # K_mask (geometry of haze)
        # Expansion: Scale by 10.0 to allow K > 1.0 (required for dehazing)
        k_mask = torch.sigmoid(out_f32[:, 0:1, :, :]) * 10.0  # (B, 1, H, W)
        
        # Adaptive K(x): Image-wide color consistency + Spatial geometry
        K = k_mask * c_atm_f32                              # (B, 3, H, W)

        # Refinement residuals (scaled for stability)
        delta = torch.tanh(out_f32[:, 1:4, :, :]) * 0.1     # (B, 3, H, W)

        # --- 4. Physics Reconstruction ---
        j_aod = K * hazy_f32 - K + 1.0                # (B, 3, H, W)
        j_pred = torch.clamp(j_aod + delta, 0.0, 1.0)

        # Cast back to input dtype for trainer consistency
        j_pred = j_pred.to(hazy_input.dtype)

        if return_maps:
            return j_pred, torch.clamp(j_aod, 0, 1).to(hazy_input.dtype), delta.to(hazy_input.dtype)

        return j_pred


# =============================================================================
# Quick test & parameter count
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaDehaze(img_size=256, embed_dim=64, n_layers=4).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MambaDehaze Trainable Parameters: {params:,}")

    dummy = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")

    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak VRAM: {mem:.3f} GB")
