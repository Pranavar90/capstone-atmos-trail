# AtmosDehaze AI: Research-Grade Single Image Dehazing

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev/)

AtmosDehaze AI is a powerful, research-grade prototype designed for single image restoration using a hybrid approach of deep learning and atmospheric physics. Optimized for consumer-grade hardware (specifically the **RTX 3050 - 4GB VRAM**), it implements a full end-to-end machine learning pipeline from automated dataset acquisition to a premium web interface.

---

## 🔬 Scientific Foundation

The system is built upon the **Koschmieder Atmospheric Scattering Model**:

$$I(x) = J(x)t(x) + A(1 - t(x))$$

Where:
- **$I(x)$**: Observed hazy image (Input)
- **$J(x)$**: Scene radiance (Dehazed output)
- **$t(x)$**: Transmission map
- **$A$**: Global atmospheric light

### Model Strategy (Option A)
Unlike naive image-to-image translation, this model predicts the physical parameters $t(x)$ and $A$ using an **Attention U-Net** architecture. The final image $J(x)$ is analytically reconstructed:

$$J(x) = \frac{I(x) - A}{\max(t(x), \epsilon)} + A$$

This ensures the restoration adheres to the physical laws of light propagation.

---

## 🛠 Tech Stack & Optimizations

### Deep Learning Engine
- **Framework**: PyTorch
- **Architecture**: Lightweight Attention U-Net with **Depthwise Separable Convolutions** to minimize VRAM footprint.
- **Mixed Precision**: Uses `torch.cuda.amp` to accelerate training and reduce memory usage.
- **Memory Management**: Supports gradient accumulation and adaptive batching for 4GB VRAM constraints.

### Automation & Tuning
- **Kaggle API**: Fully automated multi-dataset download and extraction.
- **Optuna**: Integrated Bayesian optimization for hyperparameter tuning.
- **Metrics**: Standard evaluation using PSNR, SSIM, and MSE.

### Web Infrastructure
- **Backend**: Flask API with GPU/CPU auto-fallback.
- **Frontend**: Vite + React + Lucide Icons for a premium, responsive side-by-side comparison UI.

---

## 📁 Project Structure

```bash
project/
├── data/
│   ├── raw/             # Original datasets (Kaggle/Local Zip)
│   └── processed/       # Unified Train/Val/Test splits (256x256)
├── models/
│   ├── arch.py          # Attention U-Net Definition
│   └── physics.py       # Koschmieder Reconstruction Logic
├── training/
│   ├── trainer.py       # Core training & validation loops
│   ├── train.py         # Main execution script with config block
│   └── tune.py          # Optuna hyperparameter tuning
├── backend/
│   └── app.py           # Flask Inference API
├── frontend/            # Vite/React Application
├── outputs/
│   ├── checkpoints/     # Best and last model weights
│   ├── plots/           # Loss and Metric curves
│   └── inference/       # Temp storage for prediction results
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+
- Node.js & npm (for frontend)
- NVIDIA GPU with 4GB+ VRAM (Recommended)
- Kaggle API Credentials (`kaggle.json`)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/atmosdehaze-ai.git
cd atmosdehaze-ai

# Install Python dependencies
pip install -r requirements.txt

# Install Frontend dependencies
cd frontend
npm install
cd ..
```

### 3. Data Preparation
The system handles multiple datasets: RESIDE, NH-Haze, I-Haze, O-Haze, Dense-Haze, etc.
```bash
# Download and Extract (Standardize Kaggle + Local Zip)
python download_datasets.py

# Process & Standardize (Resize to 256x256, create splits)
python process_data.py
```

### 4. Training & Tuning
Adjust hyperparameters in the `Config Block` inside `training/train.py`.
```bash
# Start standard training
python training/train.py

# Run Bayesian Optimization (20 trials)
python training/tune.py
```

### 5. Launch the Web Application
```bash
# Terminal 1: Backend
python backend/app.py

# Terminal 2: Frontend
cd frontend
npm run dev
```

---

---

## ⚠️ Current Limitations & Hallucination Analysis

While the current model effectively removes haze from synthetic in-distribution datasets (RESIDE, Dense-Haze), it exhibits significant limitations when applied to real-world, out-of-distribution (OOD) hazy images.

### 1. The Hallucination Problem
When presented with real-world hazy scenes (e.g., grey/white concrete buildings in dense fog), the model frequently hallucinates vibrant, incorrect colors (like vivid yellows, blues, and reds).
**Root Cause:** The model learns statistical associations from its training data (e.g., "rectangular shapes under haze are usually colorful buildings"). At inference time, it guesses these colors rather than recovering true pixel values. It pattern-matches to its training distribution instead of solving the true inverse problem.

### 2. The Atmospheric Light ($A$) Estimation Gap
The training datasets primarily feature neutral grey synthetic haze. Real-world scenes often feature color-shifted haze (e.g., warm golden sunlight or smog).
**Root Cause:** The model's tiny fully-connected network estimates $A$ from deep bottleneck features. When it encounters warm haze, it misinterprets the atmospheric light, causing the Koschmieder physics formula to subtract incorrect values, amplifying warmth instead of neutralizing it.

### 3. L1 Loss Induced Blurriness
Outputs often lack high-frequency detail and appear washed out.
**Root Cause:** The model is trained primarily with L1 loss, which minimizes median pixel error. When uncertain, it generates a "safe" blurry average. Furthermore, the SSIM loss term is structurally missing from the training loop, meaning structural integrity is not enforced.

---

## 🔮 Future R&D Solution Tracks

To elevate the model from a basic physics-prior CNN to a robust real-world dehazer, the following research tracks have been identified based on state-of-the-art literature:

### Track 1: Fix the Training Signal (High Impact, Low Effort)
- **Contrastive Regularization (AECR-Net):** Add a contrastive loss term to penalize the model for generating structural content that doesn't exist in the input, forcing it to recover rather than invent colors.
- **Implement SSIM & Perceptual Loss:** Wire the missing SSIM loss and extract perceptual features from shallower VGG layers (`relu1_2`, `relu2_2`) to preserve texture and color constancy.

### Track 2: Ground the Physics (Medium Impact, Medium Effort)
- **Dark Channel Prior (DCP) Guidance:** Do not let the network freely predict the transmission map $t(x)$ from scratch. Use DCP (He et al., 2009) to establish a physics-grounded mathematical boundary that the network only refines.
- **Analytic $A$ Estimation:** Replace the neural network prediction of $A$ with a quad-tree bright-pixel search algorithm to dynamically capture true scene illumination.

### Track 3: Architectural Overhaul (SOTA Target)
- **End-to-End Regression (AOD-Net style):** Drop explicit $A$ and $t(x)$ prediction entirely. Reformulate the network to predict a single unified transform parameter $K(x)$, avoiding the error amplification inherent in division by small $t(x)$ values.
- **Feature Fusion Attention (FFA-Net):** Replace the standard U-Net with Pixel/Channel attention blocks to handle non-uniform real-world haze.
- **GAN / Cyclic Training:** Introduce a Discriminator or use CycleGAN training paradigms to enforce that dehazed outputs definitively belong to the domain of "real, clear photos".

---

## 📊 Evaluation & Metrics

The training script automatically logs per-epoch results and generates curves. Metrics tracked:
- **PSNR**: Peak Signal-to-Noise Ratio (Higher is better). Current baseline: ~18.3 dB.
- **SSIM**: Structural Similarity Index (Approaching 1.0 is better)
- **L1 Loss**: Pixel-wise intensity consistency.
- **Physics Loss**: Consistency between prediction and the scattering model.

---

## 📜 License
This project is for research and educational purposes. Dataset licenses vary by source (NTIRE, RESIDE, etc.).

## 🤝 Acknowledgments
- **NTIRE Challenges** for providing high-quality realistic datasets.
- **Kaggle** for hosting large-scale benchmarks.
- **The PyTorch Team** for the flexible deep learning framework.
