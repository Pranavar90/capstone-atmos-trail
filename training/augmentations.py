"""
Phase A: Domain-Randomized Haze Augmentation Pipeline
=====================================================
Forces the model to learn atmospheric scattering physics rather than
memorizing the specific grey-tinted haze of the RESIDE/thesis dataset.

Key augmentations:
  - HazeColorShift: random warm/cool/brown tint on the hazy input only
  - DensityJitter: random brightness/contrast to simulate varying scatter
  - GaussianNoise: sensor noise robustness
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import math


class HazeColorShift:
    """
    Randomly shifts the colour temperature of the hazy input to simulate
    warm golden smog, blue dusk fog, or brown pollution haze.
    Only applied to the HAZY image, never to the ground truth.
    """
    def __init__(self, hue_range=0.08, sat_range=0.3, val_range=0.15):
        self.hue_range = hue_range
        self.sat_range = sat_range
        self.val_range = val_range

    def __call__(self, img):
        """img: PIL Image or Tensor"""
        hue_factor = random.uniform(-self.hue_range, self.hue_range)
        sat_factor = random.uniform(max(0, 1 - self.sat_range), 1 + self.sat_range)
        val_factor = random.uniform(max(0, 1 - self.val_range), 1 + self.val_range)

        img = TF.adjust_hue(img, hue_factor)
        img = TF.adjust_saturation(img, sat_factor)
        img = TF.adjust_brightness(img, val_factor)
        return img


class DensityJitter:
    """
    Simulates varying haze density by randomly adjusting contrast and
    brightness — mimicking different beta scattering coefficients.
    """
    def __init__(self, contrast_range=(0.7, 1.3), brightness_range=(-0.1, 0.1)):
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range

    def __call__(self, img):
        contrast = random.uniform(*self.contrast_range)
        brightness = random.uniform(*self.brightness_range)
        img = TF.adjust_contrast(img, contrast)
        img = TF.adjust_brightness(img, 1.0 + brightness)
        return img


class AddGaussianNoise:
    """
    Adds random Gaussian noise to the tensor to improve robustness.
    Applied AFTER ToTensor (works on tensors, not PIL images).
    """
    def __init__(self, mean=0.0, std_range=(0.0, 0.03)):
        self.mean = mean
        self.std_range = std_range

    def __call__(self, tensor):
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(tensor) * std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


class HazeDomainRandomization:
    """
    Complete domain-randomization pipeline for hazy images.
    Applies colour-shift + density-jitter + optional noise.

    Usage in DataLoader:
        augmentor = HazeDomainRandomization(image_size=256)
        hazy_tensor, clear_tensor = augmentor(hazy_pil, clear_pil)
    """
    def __init__(self, image_size=256, apply_noise=True, p=0.5):
        self.image_size = image_size
        self.apply_noise = apply_noise
        self.p = p  # probability of applying domain randomization

        # Shared geometric transforms (applied to BOTH hazy and clear)
        self.resize = T.Resize((image_size, image_size))
        self.to_tensor = T.ToTensor()

        # Hazy-only colour domain randomization
        self.color_shift = HazeColorShift()
        self.density_jitter = DensityJitter()
        self.noise = AddGaussianNoise()

    def __call__(self, hazy_pil, clear_pil):
        """
        Args:
            hazy_pil: PIL Image of hazy scene
            clear_pil: PIL Image of ground-truth clear scene

        Returns:
            hazy_tensor, clear_tensor: both (3, H, W) tensors
        """
        # --- Shared geometric augmentations ---
        # Random horizontal flip
        if random.random() > 0.5:
            hazy_pil = TF.hflip(hazy_pil)
            clear_pil = TF.hflip(clear_pil)

        # Random rotation (-15 to +15 degrees)
        angle = random.uniform(-15, 15)
        hazy_pil = TF.rotate(hazy_pil, angle)
        clear_pil = TF.rotate(clear_pil, angle)

        # Resize
        hazy_pil = self.resize(hazy_pil)
        clear_pil = self.resize(clear_pil)

        # --- Hazy-only domain randomization ---
        if random.random() < self.p:
            hazy_pil = self.color_shift(hazy_pil)
            hazy_pil = self.density_jitter(hazy_pil)

        # Convert to tensor
        hazy_tensor = self.to_tensor(hazy_pil)
        clear_tensor = self.to_tensor(clear_pil)

        # Add sensor noise to hazy input only
        if self.apply_noise and random.random() < self.p:
            hazy_tensor = self.noise(hazy_tensor)

        return hazy_tensor, clear_tensor
