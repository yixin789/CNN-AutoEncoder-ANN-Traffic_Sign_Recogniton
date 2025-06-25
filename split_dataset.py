import json
import os
import shutil
from collections import defaultdict
import random
from tqdm import tqdm # For progress bar

# --- Configuration ---
SOURCE_BASE_DIR = 'test.v1i.coco' # 您从Roboflow下载的原始数据集的根目录
TARGET_BASE_DIR = 'data_stratified' # 新生成的、分层抽样数据集的根目录

TRAIN_SPLIT_RATIO = 0.86
VAL_SPLIT_RATIO = 0.13 
TEST_SPLIT_RATIO = 0.01

# 确保比例总和为 1
assert TRAIN_SPLIT_RATIO + VAL_SPLIT_RATIO + TEST_SPLIT_RATIO == 1.0, \
    "Split ratios must sum to 1.0!"

# --- Helper Functions ---
def load_coco_json(json_path):
    """Loads a COCO JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_coco_json(data, json_path):
    """Saves data to a COCO JSON file."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def merge_datasets(source_dirs, source_base_dir):
    """
    Merges multiple COCO datasets (from different splits within the same Roboflow export)
    into a single consolidated dictionary.
    Handles unique IDs and collects all images and annotations.
    """
    merged_data = {
        "info": {"description": "Merged COCO dataset from Roboflow export"},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    seen_category_names = set()
    current_image_id = 0
    current_annotation_id = 0

    for split_dir in source_dirs:
        json_path = os.path.join(source_base_dir, split_dir, '_annotations.coco.json')
        print(f"Loading {json_path}...")
        
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found. Skipping.")
            continue

        data = load_coco_json(json_path)

        for cat in data["categories"]:
            if cat['name'] not in seen_category_names:
                merged_data["categories"].append(cat)
                seen_category_names.add(cat['name'])
        
        merged_data["categories"].sort(key=lambda x: x['id'])

        image_id_map_for_current_split = {}

        for img in data["images"]:
            old_image_id = img['id']
            new_image_id = current_image_id
            image_id_map_for_current_split[old_image_id] = new_image_id
            
            new_img = img.copy()
            new_img['id'] = new_image_id
            new_img['original_relative_path'] = os.path.join(split_dir, img['file_name'])

            merged_data["images"].append(new_img)
            current_image_id += 1

        for ann in data["annotations"]:
            old_image_id = ann['image_id']
            new_image_id = image_id_map_for_current_split.get(old_image_id) 

            if new_image_id is None:
                print(f"Warning: Annotation {ann['id']} references missing image {old_image_id} from current split {split_dir}. Skipping.")
                continue

            new_ann = ann.copy()
            new_ann['id'] = current_annotation_id
            new_ann['image_id'] = new_image_id
            
            merged_data["annotations"].append(new_ann)
            current_annotation_id += 1
            
    print(f"Merged {len(merged_data['images'])} images and {len(merged_data['annotations'])} annotations.")
    return merged_data

def perform_stratified_split(merged_data):
    """
    Performs a stratified split on the merged COCO dataset.
    Ensures all categories are present in train, valid, and test splits.
    """
    images_by_category = defaultdict(list)
    image_to_categories = defaultdict(set)

    for ann in merged_data['annotations']:
        images_by_category[ann['category_id']].append(ann['image_id'])
        image_to_categories[ann['image_id']].add(ann['category_id'])

    all_image_ids = list(set([img['id'] for img in merged_data['images']]))
    random.shuffle(all_image_ids)

    train_ids = set()
    valid_ids = set() # Changed from val_ids to valid_ids for consistency
    test_ids = set()
    
    remaining_ids_pool = set(all_image_ids)

    min_samples_per_category_in_valid_test = 2 # At least 2 images for each category in valid/test

    valid_category_counts = defaultdict(int) # Changed from val_category_counts
    test_category_counts = defaultdict(int)

    for cat_id_obj in tqdm(merged_data['categories'], desc="Ensuring category presence in valid/test"):
        cat_id = cat_id_obj['id']
        candidate_images = [img_id for img_id in images_by_category[cat_id] if img_id in remaining_ids_pool]
        random.shuffle(candidate_images)

        # Allocate to valid
        for img_id in candidate_images:
            if valid_category_counts[cat_id] < min_samples_per_category_in_valid_test:
                if img_id not in valid_ids and img_id not in test_ids:
                    valid_ids.add(img_id)
                    remaining_ids_pool.remove(img_id)
                    for c_id in image_to_categories[img_id]:
                        valid_category_counts[c_id] += 1
            else:
                break

        # Allocate to test
        for img_id in candidate_images:
            if test_category_counts[cat_id] < min_samples_per_category_in_valid_test:
                if img_id not in valid_ids and img_id not in test_ids:
                    test_ids.add(img_id)
                    remaining_ids_pool.remove(img_id)
                    for c_id in image_to_categories[img_id]:
                        test_category_counts[c_id] += 1
            else:
                break
    
    remaining_list = list(remaining_ids_pool)
    random.shuffle(remaining_list)

    num_total_images = len(all_image_ids)
    num_train_target = int(num_total_images * TRAIN_SPLIT_RATIO)
    num_valid_target = int(num_total_images * VAL_SPLIT_RATIO) # Changed from num_val_target
    num_test_target = int(num_total_images * TEST_SPLIT_RATIO)

    for img_id in remaining_list:
        if len(train_ids) < num_train_target:
            train_ids.add(img_id)
        elif len(valid_ids) < num_valid_target: # Changed from val_ids
            valid_ids.add(img_id)
        elif len(test_ids) < num_test_target:
            test_ids.add(img_id)
        else:
            train_ids.add(img_id)

    assigned_ids = train_ids.union(valid_ids).union(test_ids) # Changed from val_ids
    assert len(assigned_ids) == num_total_images, "Error: Not all images were assigned to a split!"
    assert len(train_ids.intersection(valid_ids)) == 0, "Error: Image assigned to both train and valid!" # Changed
    assert len(train_ids.intersection(test_ids)) == 0, "Error: Image assigned to both train and test!"
    assert len(valid_ids.intersection(test_ids)) == 0, "Error: Image assigned to both valid and test!" # Changed


    # Build the new COCO JSON structures
    new_splits_data = {
        "train": {
            "info": merged_data["info"], "licenses": merged_data["licenses"],
            "categories": merged_data["categories"], "images": [], "annotations": []
        },
        "valid": { # Key is 'valid' for consistency
            "info": merged_data["info"], "licenses": merged_data["licenses"],
            "categories": merged_data["categories"], "images": [], "annotations": []
        },
        "test": {
            "info": merged_data["info"], "licenses": merged_data["licenses"],
            "categories": merged_data["categories"], "images": [], "annotations": []
        }
    }
    
    image_id_map_in_split = defaultdict(dict)
    new_image_id_counters = { "train": 0, "valid": 0, "test": 0 } # Changed 'val' to 'valid'

    for img in merged_data['images']:
        original_global_image_id = img['id']
        split_name = None
        if original_global_image_id in train_ids:
            split_name = "train"
        elif original_global_image_id in valid_ids: # Changed from val_ids
            split_name = "valid" # Changed from 'val'
        elif original_global_image_id in test_ids:
            split_name = "test"
        
        if split_name:
            new_img = img.copy()
            new_img['id'] = new_image_id_counters[split_name]
            image_id_map_in_split[split_name][original_global_image_id] = new_img['id']
            
            new_img['file_name'] = os.path.basename(new_img['original_relative_path'])
            
            new_splits_data[split_name]["images"].append(new_img)
            new_image_id_counters[split_name] += 1


    for ann in merged_data['annotations']:
        original_global_image_id = ann['image_id']
        split_name = None
        if original_global_image_id in train_ids:
            split_name = "train"
        elif original_global_image_id in valid_ids: # Changed from val_ids
            split_name = "valid" # Changed from 'val'
        elif original_global_image_id in test_ids:
            split_name = "test"
        
        if split_name:
            new_ann = ann.copy()
            new_ann['image_id'] = image_id_map_in_split[split_name][original_global_image_id]
            new_splits_data[split_name]["annotations"].append(new_ann)

    print("\n--- Split Summary ---")
    print(f"Train images: {len(new_splits_data['train']['images'])}")
    print(f"Validation images: {len(new_splits_data['valid']['images'])}") # Key is 'valid'
    print(f"Test images: {len(new_splits_data['test']['images'])}")
    print("---------------------\n")

    return new_splits_data

def copy_images_for_splits(merged_data, new_splits_data):
    """
    Copies images from their original locations to the new split directories.
    Uses the 'original_relative_path' stored during merging.
    """
    print("Copying images to new split directories...")
    # Create target directories
    for split_name_key in ["train", "valid", "test"]: # Loop over valid split name keys
        os.makedirs(os.path.join(TARGET_BASE_DIR, split_name_key), exist_ok=True) # Creates 'valid' folder
    
    # Copy files
    for split_name_key, split_data in new_splits_data.items(): # Iterates "train", "valid", "test"
        print(f"Copying images for {split_name_key} split...")
        target_image_dir = os.path.join(TARGET_BASE_DIR, split_name_key) # Uses 'valid' here
        
        for img_info_in_new_split in tqdm(split_data['images'], desc=f"Copying {split_name_key} images"):
            src_path = os.path.join(SOURCE_BASE_DIR, img_info_in_new_split['original_relative_path'])
            dest_path = os.path.join(target_image_dir, img_info_in_new_split['file_name'])

            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
            else:
                print(f"Error: Original source image not found at {src_path}. Skipping copy for {img_info_in_new_split['file_name']}.")
            

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting dataset merging and splitting...")

    # 1. Merge all original datasets
    source_splits = ['train', 'valid', 'test'] # 假设这是您Roboflow下载的原始文件夹
    all_merged_data = merge_datasets(source_splits, SOURCE_BASE_DIR)

    # 2. Perform stratified split
    new_splits_data = perform_stratified_split(all_merged_data)

    # 3. Save new COCO JSON files
    print("Saving new COCO JSON files...")
    save_coco_json(new_splits_data["train"], os.path.join(TARGET_BASE_DIR, 'train', '_annotations.coco.json'))
    save_coco_json(new_splits_data["valid"], os.path.join(TARGET_BASE_DIR, 'valid', '_annotations.coco.json')) # 统一使用 'valid'
    save_coco_json(new_splits_data["test"], os.path.join(TARGET_BASE_DIR, 'test', '_annotations.coco.json'))

    # 4. Copy images to new directories
    copy_images_for_splits(all_merged_data, new_splits_data)

    print("Dataset splitting complete. Update your main.py to use the new paths:")
    print(f"train_json_path = '{TARGET_BASE_DIR}/train/_annotations.coco.json'")
    print(f"train_image_dir = '{TARGET_BASE_DIR}/train'")
    print(f"valid_json_path = '{TARGET_BASE_DIR}/valid/_annotations.coco.json'") # 统一使用 'valid'
    print(f"valid_image_dir = '{TARGET_BASE_DIR}/valid'") # 统一使用 'valid'
    print(f"test_json_path = '{TARGET_BASE_DIR}/test/_annotations.coco.json'")
    print(f"test_image_dir = '{TARGET_BASE_DIR}/test'")