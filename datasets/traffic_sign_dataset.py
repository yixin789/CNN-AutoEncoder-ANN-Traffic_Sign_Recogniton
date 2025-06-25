# datasets/traffic_sign_dataset.py
import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision import transforms
from PIL import Image

NUM_CLASSES = 63 # Make sure this is correct for your dataset

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

class TrafficSignDataset(Dataset):
    """
    Custom PyTorch Dataset for traffic sign images with COCO annotations,
    designed for MULTI-CLASS (SINGLE-LABEL) classification.
    """
    def __init__(self, json_path, image_dir, device, transform=None):
        """
        Initializes the dataset.
        Args:
            json_path (str): Path to the COCO annotation JSON file.
            image_dir (str): Path to the directory containing images.
            device (torch.device): The device (CPU or GPU) to load tensors onto.
            transform (torchvision.transforms.Compose, optional): Optional transformations to be applied.
                                                                 If None, a default transform will be used.
        """
        self.coco = COCO(json_path)
        self.image_dir = image_dir
        self.image_ids = self.coco.getImgIds()
        self.device = device

        categories = self.coco.loadCats(self.coco.getCatIds())
        # Create a mapping from COCO category_id to a contiguous 0-indexed label
        # This is essential for CrossEntropyLoss which expects 0-indexed labels
        self.label_mapping = {cat['id']: i for i, cat in enumerate(categories)}

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy data, ensure label is a single index, e.g., 0
            # For multi-class, labels should be LongTensor
            return torch.zeros(3, 224, 224, device=self.device), torch.tensor(0, dtype=torch.long, device=self.device)

        if self.transform:
            image = self.transform(image)

        image = image.to(self.device, non_blocking=True)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # --- CRITICAL CHANGE FOR MULTI-CLASS (SINGLE-LABEL) ---
        # Initialize with a default/dummy label in case no annotation is found (should not happen with good data)
        # Or, more robustly, skip this sample later in DataLoader if it's not valid
        label_idx = -1 # Indicate no valid label found initially

        if anns:
            # For multi-class (single-label), we assume one dominant label per image.
            # Take the first annotation's category ID and map it.
            # If an image has multiple annotations, this will just pick the first one.
            # In a real traffic sign dataset for single-label, each image should ideally have only one relevant sign.
            first_ann = anns[0] 
            category_id = first_ann['category_id']
            if category_id in self.label_mapping:
                label_idx = self.label_mapping[category_id]
            else:
                print(f"Warning: Category ID {category_id} not found in mapping for image {image_path}. Using dummy label 0.")
                label_idx = 0 # Fallback to a default label (e.g., background or class 0)
        else:
            # Handle case where an image has no annotations (e.g., background class or skip)
            # For simplicity, returning class 0 as a dummy, but ideally, these samples should be filtered.
            print(f"Warning: No annotations found for image {image_path}. Assigning dummy label 0.")
            label_idx = 0 
            
        # Ensure the label is a LongTensor for CrossEntropyLoss
        label_tensor = torch.tensor(label_idx, dtype=torch.long, device=self.device)
                
        return image, label_tensor
