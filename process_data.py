import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import random

def get_image_pairs(root_dir, ds_name):
    pairs = []
    if ds_name == "NH-HAZE":
        base = os.path.join(root_dir, "NH-HAZE", "NH-HAZE")
        if not os.path.exists(base): return []
        files = [f for f in os.listdir(base) if f.endswith("_hazy.png")]
        for f in files:
            hazy = os.path.join(base, f)
            gt = os.path.join(base, f.replace("_hazy.png", "_GT.png"))
            if os.path.exists(gt):
                pairs.append((hazy, gt))
    
    elif ds_name in ["I-HAZE", "O-HAZE"]:
        subfolder = "I-HAZY NTIRE 2018" if ds_name == "I-HAZE" else "O-HAZY NTIRE 2018"
        hazy_dir = os.path.join(root_dir, ds_name, subfolder, "hazy")
        gt_dir = os.path.join(root_dir, ds_name, subfolder, "GT")
        if not os.path.exists(hazy_dir): return []
        for f in os.listdir(hazy_dir):
            hazy = os.path.join(hazy_dir, f)
            # Some datasets have slightly different naming for GT
            gt = os.path.join(gt_dir, f)
            if not os.path.exists(gt):
                # Try common suffixes
                gt = os.path.join(gt_dir, f.replace(".png", "_GT.png").replace(".jpg", "_GT.jpg"))
            if os.path.exists(gt):
                pairs.append((hazy, gt))

    elif ds_name == "Dense_Haze":
        hazy_dir = os.path.join(root_dir, "thesis", "Dense_Haze_NTIRE19", "hazy")
        gt_dir = os.path.join(root_dir, "thesis", "Dense_Haze_NTIRE19", "GT")
        if os.path.exists(hazy_dir):
            for f in os.listdir(hazy_dir):
                hazy_path = os.path.join(hazy_dir, f)
                gt_path = os.path.join(gt_dir, f.replace("_hazy.png", "_GT.png"))
                if os.path.exists(gt_path):
                    pairs.append((hazy_path, gt_path))

    elif ds_name == "SOTS":
        for mode in ["indoor", "outdoor"]:
            hazy_dir = os.path.join(root_dir, "thesis", "SOTS", "SOTS", mode, "hazy")
            gt_dir = os.path.join(root_dir, "thesis", "SOTS", "SOTS", mode, "gt")
            if os.path.exists(hazy_dir):
                for f in os.listdir(hazy_dir):
                    # SOTS indoor: GT naming might vary. e.g., 1.png in hazy maps to 1.png in gt?
                    # Actually SOTS indoor has 500 hazy and 50 gt? RESIDE is complex.
                    # We'll take what we can.
                    gt_name = f.split('_')[0] + ".png" # common RESIDE mapping
                    gt_path = os.path.join(gt_dir, gt_name)
                    if os.path.exists(gt_path):
                        pairs.append((os.path.join(hazy_dir, f), gt_path))

    return pairs

def process_dataset(image_size=256):
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    
    all_pairs = []
    
    # 1. Thesis datasets
    thesis_dir = os.path.join(raw_dir, "thesis")
    for ds in ["NH-HAZE", "I-HAZE", "O-HAZE", "Dense_Haze", "SOTS"]:
        print(f"Parsing {ds}...")
        pairs = get_image_pairs(raw_dir if ds in ["NH-HAZE", "I-HAZE", "O-HAZE"] else raw_dir, ds)
        # Note: adjust paths as needed
        all_pairs.extend(pairs)

    # 2. Haze1k (if downloaded)
    haze1k_dir = os.path.join(raw_dir, "haze1k")
    if os.path.exists(haze1k_dir):
        # Haze1k structure: train/test_{thin,moderate,thick}/{hazy,clear}
        for root, dirs, files in os.walk(haze1k_dir):
            if "hazy" in root:
                clear_root = root.replace("hazy", "clear")
                for f in files:
                    if f.endswith(('.png', '.jpg')):
                        all_pairs.append((os.path.join(root, f), os.path.join(clear_root, f)))

    random.shuffle(all_pairs)
    
    # Split: 80% train, 10% val, 10% test
    n = len(all_pairs)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    splits = {
        "train": all_pairs[:train_end],
        "val": all_pairs[train_end:val_end],
        "test": all_pairs[val_end:]
    }
    
    for split_name, pairs in splits.items():
        print(f"Processing {split_name} split ({len(pairs)} pairs)...")
        hazy_out = os.path.join(processed_dir, split_name, "hazy")
        clear_out = os.path.join(processed_dir, split_name, "clear")
        
        for i, (hazy_path, clear_path) in enumerate(tqdm(pairs)):
            try:
                h_img = cv2.imread(hazy_path)
                c_img = cv2.imread(clear_path)
                
                if h_img is None or c_img is None: continue
                
                h_img = cv2.resize(h_img, (image_size, image_size))
                c_img = cv2.resize(c_img, (image_size, image_size))
                
                cv2.imwrite(os.path.join(hazy_out, f"{i}.png"), h_img)
                cv2.imwrite(os.path.join(clear_out, f"{i}.png"), c_img)
            except Exception as e:
                print(f"Error processing {hazy_path}: {e}")

if __name__ == "__main__":
    process_dataset()
