import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F

from src.data_loading.cityscapes import CityscapesDataset
from src.data_loading.idd import IDDDataset
from src.data_loading.transforms import get_val_transforms

class Evaluator:
    def __init__(self, config, model):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.model.eval()

    def evaluate(self, dataset_name, split='val', output_dir='eval_results'):
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine the correct evaluation config based on the model's output classes
        model_num_classes = self.config.num_classes
        print(f"Model has {model_num_classes} output classes")
        
        if dataset_name == 'cityscapes':
            from configs.cityscapes_config import CityscapesConfig
            eval_config = CityscapesConfig()
            eval_dataset = CityscapesDataset(eval_config, split=split, transform=get_val_transforms(eval_config))
        elif dataset_name == 'idd':
            if model_num_classes == 27:
                # Model outputs 27 classes (native IDD), use IDD config for evaluation
                from configs.idd_config import IDDConfig
                eval_config = IDDConfig()
                eval_dataset = IDDDataset(eval_config, split=split, transform=get_val_transforms(eval_config))
                print("Using IDD native evaluation (27 classes)")
            else:
                # Model outputs 19 classes (mapped), use Cityscapes config for evaluation
                from configs.cityscapes_config import CityscapesConfig
                eval_config = CityscapesConfig()
                from configs.idd_config import IDDConfig
                idd_config = IDDConfig() 
                eval_dataset = IDDDataset(idd_config, split=split, transform=get_val_transforms(eval_config))
                print("Using Cityscapes mapped evaluation (19 classes)")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        loader = DataLoader(eval_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        num_classes = eval_config.num_classes  # Use the correct number of classes for evaluation
        print(f"Evaluation using {num_classes} classes")
        total_cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=self.device)
        
        total_pixels = 0
        correct_pixels = 0

        with torch.no_grad():
            for images, masks in tqdm(loader, desc=f"Evaluating on {dataset_name} {split}"):
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, tuple): 
                    outputs = outputs[0]
                
                # Resize predictions to match mask size
                if outputs.shape[2:] != masks.shape[1:]:
                    outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                preds = torch.argmax(outputs, dim=1)

                # Calculate confusion matrix for valid pixels only
                valid_mask = (masks >= 0) & (masks < num_classes)
                valid_preds = preds[valid_mask]
                valid_masks = masks[valid_mask]
                
                if len(valid_preds) > 0:
                    # Calculate pixel accuracy
                    correct_pixels += (valid_preds == valid_masks).sum().item()
                    total_pixels += len(valid_preds)
                    
                    # Calculate confusion matrix
                    indices = valid_masks * num_classes + valid_preds
                    batch_cm = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
                    total_cm += batch_cm
        
        # Calculate metrics from confusion matrix
        intersection = torch.diag(total_cm).float()
        union = total_cm.sum(dim=1).float() + total_cm.sum(dim=0).float() - intersection
        
        # Calculate IoU per class
        iou_per_class = intersection / (union + 1e-8)
        
        # Only consider classes that appear in the dataset
        valid_classes = union > 0
        valid_ious = iou_per_class[valid_classes]
        
        mean_iou = valid_ious.mean().item() if len(valid_ious) > 0 else 0.0
        pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0

        # Print detailed results
        print("\n" + "="*30)
        print("Class-wise IoU Scores")
        for i, class_name in enumerate(eval_config.class_names):
            if valid_classes[i]:
                print(f"  {class_name: <15}: {iou_per_class[i].item():.4f}")
            else:
                print(f"  {class_name: <15}: {0.0:.4f}")
        print("="*30)
        
        return {'mIoU': mean_iou, 'pixel_accuracy': pixel_accuracy}