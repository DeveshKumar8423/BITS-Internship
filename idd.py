import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.label_utils import idd_label_map
import json
import cv2

def json_to_mask(json_path, height, width):
    """Parses a JSON file with polygons and creates a mask."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # IDD label to ID mapping
    label_to_id = {
        'road': 0, 'sidewalk': 1, 'building': 2, 'wall': 3, 'fence': 4,
        'pole': 5, 'traffic light': 6, 'traffic sign': 7, 'vegetation': 8,
        'terrain': 9, 'sky': 10, 'person': 11, 'rider': 12, 'car': 13,
        'truck': 14, 'bus': 15, 'train': 16, 'motorcycle': 17, 'bicycle': 18,
        'autorickshaw': 19, 'animal': 20, 'traffic sign back': 21, 'curb': 22,
        'obstacle': 23, 'parking': 24, 'caravan': 25, 'trailer': 26
    }
    
    for obj in data.get('objects', []):
        label = obj.get('label')
        polygon = obj.get('polygon')
        if label and polygon and label in label_to_id:
            label_id = label_to_id[label]
            cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], label_id)
            
    return mask

class IDDDataset(Dataset):
    def __init__(self, config, split='val', transform=None):
        self.config = config
        self.split = split
        self.transform = transform
        self.label_mapper = idd_label_map()
        
        self.image_dir = os.path.join(config.data_root, "leftImg8bit", split)
        self.mask_dir = os.path.join(config.data_root, "gtFine", split)

        # Find all image files
        img_pattern_png = os.path.join(self.image_dir, '**', '*_leftImg8bit.png')
        img_pattern_jpg = os.path.join(self.image_dir, '**', '*_leftImg8bit.jpg')
        self.image_paths = sorted(list(set(glob.glob(img_pattern_png, recursive=True) + glob.glob(img_pattern_jpg, recursive=True))))

        if hasattr(config, 'dataset_percentage') and config.dataset_percentage < 1.0:
            num_samples = int(len(self.image_paths) * config.dataset_percentage)
            self.image_paths = self.image_paths[:num_samples]

        # FIX: Replaced the fragile string replacement with a robust path construction.
        self.mask_paths = []
        for p in self.image_paths:
            # e.g., 'frame0821_leftImg8bit.jpg'
            base_filename = os.path.basename(p)
            # e.g., '205'
            sub_folder = os.path.basename(os.path.dirname(p))
            # e.g., 'frame0821_gtFine_polygons.json'
            mask_filename = base_filename.replace('_leftImg8bit.png', '_gtFine_polygons.json').replace('_leftImg8bit.jpg', '_gtFine_polygons.json')
            # e.g., './data/idd/gtFine/val/205/frame0821_gtFine_polygons.json'
            self.mask_paths.append(os.path.join(self.mask_dir, sub_folder, mask_filename))

        print(f"Total images found for split '{split}': {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        json_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        
        if os.path.exists(json_path):
            mask = json_to_mask(json_path, image.height, image.width)
        else:
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
        
        mask = self.label_mapper(mask)
        image_np = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image_np, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        return image, torch.from_numpy(np.array(mask)).long()