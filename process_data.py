import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import random
from concurrent.futures import ProcessPoolExecutor

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
                    gt_name = f.split('_')[0] + ".png" # common RESIDE mapping
                    gt_path = os.path.join(gt_dir, gt_name)
                    if os.path.exists(gt_path):
                        pairs.append((os.path.join(hazy_dir, f), gt_path))

    elif ds_name == "archive(1)":
        # Over 14,000 images. Format: hazy/ID_variant_beta.png matches clear/ID.png
        hazy_dir = os.path.join(root_dir, "thesis", "archive(1)", "hazy")
        gt_dir = os.path.join(root_dir, "thesis", "archive(1)", "clear")
        if os.path.exists(hazy_dir) and os.path.exists(gt_dir):
            for f in os.listdir(hazy_dir):
                if not f.endswith(('.png', '.jpg')): continue
                gt_name = f.split('_')[0] + ".png"
                gt_path = os.path.join(gt_dir, gt_name)
                # Some clear images might be .jpg
                if not os.path.exists(gt_path):
                    gt_path = os.path.join(gt_dir, f.split('_')[0] + ".jpg")
                if os.path.exists(gt_path):
                    pairs.append((os.path.join(hazy_dir, f), gt_path))

    elif ds_name == "BeDDE":
        base_dir = os.path.join(root_dir, "thesis", "BeDDE", "BeDDE")
        if os.path.exists(base_dir):
            for city in os.listdir(base_dir):
                city_dir = os.path.join(base_dir, city)
                if not os.path.isdir(city_dir): continue
                hazy_dir = os.path.join(city_dir, "fog")
                gt_dir = os.path.join(city_dir, "gt")
                gt_path = os.path.join(gt_dir, f"{city}_clear.png")
                if not os.path.exists(gt_path):
                    gt_path = os.path.join(gt_dir, f"{city}_clear.jpg")
                
                if os.path.exists(hazy_dir) and os.path.exists(gt_path):
                    for f in os.listdir(hazy_dir):
                        if f.endswith(('.png', '.jpg')):
                            pairs.append((os.path.join(hazy_dir, f), gt_path))

    return pairs

def process_image(args):
    hazy_path, clear_path, out_idx, hazy_out, clear_out, image_size = args
    try:
        h_img = cv2.imread(hazy_path)
        c_img = cv2.imread(clear_path)
        
        if h_img is None or c_img is None: return False
        
        h_img = cv2.resize(h_img, (image_size, image_size))
        c_img = cv2.resize(c_img, (image_size, image_size))
        
        cv2.imwrite(os.path.join(hazy_out, f"{out_idx}.png"), h_img)
        cv2.imwrite(os.path.join(clear_out, f"{out_idx}.png"), c_img)
        return True
    except Exception as e:
        print(f"Error processing {hazy_path}: {e}")
        return False

def process_dataset(image_size=256):
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    
    all_pairs = []
    
    # 1. Thesis datasets
    thesis_dir = os.path.join(raw_dir, "thesis")
    for ds in ["NH-HAZE", "I-HAZE", "O-HAZE", "Dense_Haze", "SOTS", "archive(1)", "BeDDE"]:
        print(f"Parsing {ds}...")
        pairs = get_image_pairs(raw_dir if ds in ["NH-HAZE", "I-HAZE", "O-HAZE"] else raw_dir, ds)
        all_pairs.extend(pairs)

    # 2. Haze1k (if downloaded)
    haze1k_dir = os.path.join(raw_dir, "haze1k")
    if os.path.exists(haze1k_dir):
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
        
        os.makedirs(hazy_out, exist_ok=True)
        os.makedirs(clear_out, exist_ok=True)
        
        # Prepare arguments for multiprocessing
        tasks = []
        for i, (hazy_path, clear_path) in enumerate(pairs):
            tasks.append((hazy_path, clear_path, i, hazy_out, clear_out, image_size))
            
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(process_image, tasks), total=len(tasks)))

if __name__ == "__main__":
    process_dataset()
