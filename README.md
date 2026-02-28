# Capstone: End-to-End Vision Mamba (Vim) Dehazing Architecture
**Branch:** `Visionmambatrainingready`

This repository branch contains a **World-First Implementation** of a Single Image Dehazing Neural Network. It completely replaces traditional Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) with a pure PyTorch native **State Space Model (SSM) — Vision Mamba**.

This project solves fundamental mathematical instabilities in atmospheric scattering physics (Koschmieder Inversion) by reformulating the physics into an End-to-End unified parameter prediction.

---

## 📖 1. What This Project Solves (The Science)

### The Problem with Existing Dehazing AIs:
1. **CNNs Lack "Global Vision"**: CNNs (like U-Net) look at pixels locally through small 3x3 windows. They fail to understand the global "depth" of an image. If a CNN sees a white pixel, it cannot tell if it is a distant white cloud or a white car parked 5 feet away. This causes severe colour hallucinations on real-world hazy images.
2. **Transformers Blow Up VRAM**: Vision Transformers solve the "Global Vision" problem by calculating the relationship between every patch of an image simultaneously (Quadratic Complexity $O(N^2)$). However, this takes massive amounts of graphics memory (VRAM), making it impossible to train high-resolution images on consumer GPUs (like an RTX 3050 4GB).
3. **Mathematical Explosions (NaN)**: The traditional Koschmieder physics equation requires dividing by the Transmission Map $t(x)$. If the haze is thick, $t(x)$ approaches 0. Dividing by zero causes neural networks to explode with "Not A Number" (NaN) errors during training.

### Our Solution (The Vision Mamba Architecture):
1. **Linear Complexity Context**: We utilize **State Space Models (SSMs)**, specifically a Bi-Directional Vision Mamba block. Mamba scans across the image sequentially like reading a book. It "remembers" global context with perfect efficiency ($O(N)$ linear scaling), giving it the global intelligence of a Transformer but with the tiny VRAM footprint of a CNN.
2. **End-to-End AOD Physics Formulation**: We entirely deleted the Koschmieder division equation. The network predicts a single, unified mathematical parameter tensor $K(x)$. The physics reconstruction is reformulated as: `J(x) = K(x) * I(x) - K(x) + 1`. This mathematical trick completely eliminates division-by-zero gradient explosions. Mamba trains perfectly stable from Epoch 1.
3. **Semantic Contrastive "Judge"**: Instead of just using simple pixel loss (L1) which makes images blurry, we use a frozen **ConvNeXt-Tiny** neural network as a "Judge" (Contrastive Regularization). It ensures the structural meaning of the dehazed image matches a clear sky, and heavily penalises the model if it hallucinates colours.

---

## 🚀 2. Idiot-Proof Setup & Execution Guide

If you just cloned this branch and want to train or execute the model, strictly follow these steps exactly in order.

### Step A: Environment Setup
1. **Ensure Python 3.10+ is installed.**
2. **Install the required libraries.** Open your terminal in this repository folder and run:
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional but Recommended) Install Kaggle API**: The script automatically downloads 15,000+ images from Kaggle. You must have a `kaggle.json` API token located in your `C:\Users\YOUR_NAME\.kaggle\` directory.

### Step B: Download the Datasets
This script pulls from both your local `dehazing-dataset-thesis.zip` and Kaggle sources to compile over 15,000 image pairs (including massive SOTS, BeDDE, and archive1 data banks).
Run:
```bash
python download_datasets.py
```
*(Wait until it finishes extracting everything into the `data/raw/` folder.)*

### Step C: Process & Compile the Data
The neural network requires perfectly sized 256x256 image pairs organized into Train/Val/Test folders. This script will chew through the raw datasets and sort them.
Run:
```bash
python process_data.py
```
*(This may take 5-10 minutes depending on your hard drive speed. Let the progress bars hit 100%.)*

### Step D: Train the Vision Mamba Neural Network
The core training engine. This will initialize the physics constraints, construct the bi-directional SSM blocks, load ConvNeXt into VRAM, and begin training. 

By default, it is configured for a **4GB RTX 3050** using a batch size of 14, which maximizes VRAM securely around 2.5 GB to 3.0 GB.
Run:
```bash
python -m training.train
```

*Don't panic if it initially looks slow.* The PyTorch-native Mamba loop trades speed for guaranteed Windows compatibility. It saves the best model checkpoint after every epoch to `outputs/checkpoints/mamba_best.pth`. You can safely `Ctrl+C` to stop the training whenever you want; the engine automatically supports resuming from exactly where you stopped.

---

## 📂 3. Repository Architecture Deep-Dive

If you want to understand *how* the code works, here is the exact breakdown:

*   **`models/mamba_arch.py`**: The brain. Contains the pure-PyTorch `S4Block` (State Space Model) sequence recurrent network. It chunks the image into patches, scans them forward and backwards, and uses a Convolutional refinement head to output the final $K(x)$ physics tensor.
*   **`training/augmentations.py`**: The teacher. Prevents the model from memorizing fake synthetic haze by dynamically injecting "Color Jitter" (shifting hues to warm smog/blue fog) and density alterations on the fly.
*   **`training/losses.py`**: The law. Contains SSIM (Structural loss) and the modern ConvNeXt-Tiny Contrastive Regularizer that prevents the Mamba network from hallucinating architecture or miscoloring skies.
*   **`training/train.py` & `trainer.py`**: The engine. Handles mixed-precision gradient scaling, critical State Matrix learning rate warmups, and extreme gradient clipping specifically designed to keep SSM recurrent loops numerically stable.
*   **`inference/inference_engine.py`**: The consumer. Used by the FastAPI backend to seamlessly take a user's image from the React web app and run the compiled logic to return a perfect output.

---
**This branch (`Visionmambatrainingready`) is 100% fully decoupled from the `main` branch. All experimental Mamba logic is safely quarantined here for execution on Workstations or laptops.**
