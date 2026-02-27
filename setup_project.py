import os

def create_dirs():
    dirs = [
        "data/raw",
        "data/processed/train/hazy",
        "data/processed/train/clear",
        "data/processed/val/hazy",
        "data/processed/val/clear",
        "data/processed/test/hazy",
        "data/processed/test/clear",
        "models",
        "training",
        "inference",
        "backend",
        "outputs/checkpoints",
        "outputs/plots",
        "outputs/inference"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

if __name__ == "__main__":
    create_dirs()
