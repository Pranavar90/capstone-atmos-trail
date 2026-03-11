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

        # Learnable SSM parameters (diagonal state matrix for efficiency)
        # Log-space parameterisation for numerical stability
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)
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
        Vectorized SSM scan via depthwise causal convolution.
        Replaces the slow Python for-loop (O(L) Python overhead) with a
        single GPU-resident F.conv1d call. Mathematically equivalent to
        the sequential scan because A_bar is time-invariant (diagonal, fixed).

        For each output position t:
            x_t = sum_{k=0}^{t} A_bar^{t-k} * B_proj[k]   (causal)
            y_t = C(mean_D(x_t)) + D * u_t

        Since A_bar is constant, this is a depthwise causal convolution
        with impulse response [1, A, A^2, ..., A^{L-1}] per state channel.

        Args:
            u: (B, L, D)  input sequence
        Returns:
            y: (B, L, D)  output sequence
        """
        B_batch, L, D = u.shape
        A_bar, dt = self._discretise()  # (D, N), (D,)

        # Project input to state space
        B_proj = self.B(u)   # (B, L, N)
        B_proj = B_proj * dt.unsqueeze(0).unsqueeze(0).mean(dim=-1, keepdim=True)

        # Force FP32 for numerical stability (same as original)
        A_bar = A_bar.float().clamp(min=1e-8, max=1.0 - 1e-8)  # (D, N)
        B_proj = B_proj.float()                                   # (B, L, N)

        # --- Build causal convolution kernel ---
        # A_powers[t, d, n] = A_bar[d, n]^t  for t = 0 .. L-1
        t_idx = torch.arange(L, device=u.device, dtype=torch.float32)  # (L,)
        log_A  = torch.log(A_bar)                                        # (D, N)
        A_powers = torch.exp(t_idx[:, None, None] * log_A[None])         # (L, D, N)

        # Average over D to match the original x.mean(dim=1) reduction
        # A_mean_powers[t, n] = mean_d( A_bar[d, n]^t )
        A_mean_powers = A_powers.mean(dim=1)                              # (L, N)

        # Kernel shape (N, 1, L): flipped so position 0 = A^{L-1} (oldest)
        kernel = A_mean_powers.T.unsqueeze(1).flip(-1)                    # (N, 1, L)

        # --- Depthwise causal conv over sequence ---
        # Pad L-1 zeros on the left to make the conv causal
        B_t      = B_proj.transpose(1, 2).contiguous()   # (B, N, L)
        B_padded = F.pad(B_t, (L - 1, 0))                # (B, N, 2L-1)
        x_mean   = F.conv1d(B_padded, kernel, groups=self.d_state)  # (B, N, L)
        x_mean   = x_mean.transpose(1, 2)                # (B, L, N)

        # Output: C maps (B, L, N) -> (B, L, D), plus skip connection
        y = self.C(x_mean.to(u.dtype)) + self.D * u
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

        return y + residual


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
# Spatial Reconstruction + CNN Refinement
# =============================================================================

class SpatialReconstruction(nn.Module):
    """
    Unflattens the 1D Mamba sequence back to 2D spatial grid and applies
    lightweight depthwise-separable CNN refinement for local edge recovery.
    Outputs 6-channel tensor for:
      - 3 channels for K(x) (Physics Map)
      - 3 channels for Delta(x) (Refinement Map)
    """
    def __init__(self, embed_dim=64, img_size=256, patch_size=8, out_channels=6):
        super().__init__()
        self.grid_size = img_size // patch_size   # e.g., 32
        self.embed_dim = embed_dim

        # Project embedding back to spatial channels
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Lightweight CNN refinement (depthwise separable for VRAM efficiency)
        self.refine = nn.Sequential(
            # Upsample from grid_size to img_size
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            # Depthwise separable conv for local edge refinement
            nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=3, padding=1,
                      groups=embed_dim // 4, bias=False),
            nn.Conv2d(embed_dim // 4, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x):
        """
        Args:
            x: (B, num_patches, embed_dim)
        Returns:
            K: (B, 3, H, W) — the unified physics parameter
        """
        B = x.shape[0]
        x = self.proj(x)

        # Unflatten: (B, L, D) → (B, D, grid, grid)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, self.grid_size, self.grid_size)

        # CNN refinement → K(x)
        K = self.refine(x)
        return K


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
    def __init__(self, img_size=256, patch_size=8, embed_dim=64,
                 d_state=16, n_layers=4, dropout=0.1):
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

        # Stage 3: Spatial Reconstruction → K(x)
        self.spatial_head = SpatialReconstruction(
            embed_dim=embed_dim,
            img_size=img_size,
            patch_size=patch_size,
            out_channels=6  # 3 for K, 3 for Delta
        )

    def forward(self, hazy_input, return_maps=False):
        """
        Physics-Neural Hybrid Forward Pass
        ================================================
        1. Process input via Vision Mamba Backbone.
        2. Predict Physics Head (K) and Refinement Head (ΔJ).
        3. Compute Physics Coarse: J_aod = K*I - K + 1.
        4. Fuse: J_final = clamp(J_aod + ΔJ, 0, 1).

        Args:
            hazy_input: (B, 3, H, W)
            return_maps: If True, returns (final, physics, refinement)
        Returns:
            j_pred: (B, 3, H, W) restored output
        """
        # --- 1. Mamba Backbone ---
        x = self.patch_embed(hazy_input)               # (B, L, D)

        for block in self.vim_blocks:
            x = block(x)                                # (B, L, D)

        # --- 2. Dual-Head Prediction ---
        out = self.spatial_head(x)                      # (B, 6, H, W)

        # Ensure spatial size matches input
        if out.shape[2:] != hazy_input.shape[2:]:
            out = F.interpolate(out, size=hazy_input.shape[2:],
                                mode='bilinear', align_corners=False)

        # Split into K (Physics) and Delta (Refining)
        # K uses sigmoid to constrain to (0, 1) for AOD stability
        # Delta uses tanh to constrain to (-1, 1) for additive/subtractive correction
        K = torch.sigmoid(out[:, :3, :, :])
        delta = torch.tanh(out[:, 3:, :, :])

        # --- 3. Physics Guided Fusion ---
        # Coarse AOD output
        j_aod = K * hazy_input - K + 1.0

        # Final refined output
        j_pred = torch.clamp(j_aod + delta, 0.0, 1.0)

        if return_maps:
            return j_pred, torch.clamp(j_aod, 0, 1), delta

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
