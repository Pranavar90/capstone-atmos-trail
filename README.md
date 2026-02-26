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

## 📊 Evaluation & Metrics

The training script automatically logs per-epoch results and generates curves. Metrics tracked:
- **PSNR**: Peak Signal-to-Noise Ratio (Higher is better)
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
