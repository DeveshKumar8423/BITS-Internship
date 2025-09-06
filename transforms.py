import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_train_transforms(config):
    """Returns an improved set of training augmentations for better generalization."""
    image_size = config.image_size 

    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1], always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.Blur(blur_limit=3, p=1.0),
        ], p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms(config):
    """Returns a standard set of validation transforms."""
    image_size = config.image_size

    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1], always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def decode_segmap(mask, class_colors, ignore_index=255):
    """
    Decodes a segmentation mask into a color image.
    Handles an ignore_index by coloring it grey.
    """
    # Create an empty RGB image
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Color each class
    for class_id, color in enumerate(class_colors):
        if class_id < len(class_colors):
            rgb_mask[mask == class_id] = color
        
    # FIX: Explicitly color the ignored pixels grey for visibility.
    # This will fix the solid color ground truth issue.
    if ignore_index is not None:
        rgb_mask[mask == ignore_index] = (128, 128, 128) # Grey color for ignored areas
        
    return rgb_mask