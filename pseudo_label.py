# src/training/pseudo_label.py
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from src.data_loading.idd import IDDDataset
from src.data_loading.transforms import get_val_transforms
from src.utils.metrics import compute_iou

class PseudoLabelGenerator:
    def __init__(self, config, model):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.model.eval()
        
        # Create dataset and loader for target domain
        transform = get_val_transforms(config)
        self.target_dataset = IDDDataset(
            config,
            split=config.train_split,  
            transform=transform
        )
        self.target_loader = DataLoader(
            self.target_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # Directory to save pseudo-labels
        self.pseudo_label_dir = os.path.join(config.data_root, 'pseudo_labels')
        os.makedirs(self.pseudo_label_dir, exist_ok=True)
    
    def generate_pseudo_labels(self, confidence_threshold=0.8):
        """
        Generate pseudo-labels for the target dataset
        Args:
            confidence_threshold: Only keep predictions with confidence above this threshold
        Returns:
            Dictionary containing paths to generated pseudo-labels
        """
        pseudo_labels = {}
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(tqdm(self.target_loader, desc="Generating pseudo-labels")):
                images = images.to(self.device)
                
                # Get model predictions
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probs, dim=1)
                
                # Apply confidence threshold
                mask = confidences >= confidence_threshold
                
                # Convert to numpy
                predictions = predictions.cpu().numpy()
                confidences = confidences.cpu().numpy()
                mask = mask.cpu().numpy()
                
                # Save pseudo-labels
                for i in range(len(images)):
                    # Get original image path
                    img_path = self.target_dataset.images[batch_idx * self.config.batch_size + i]
                    base_name = os.path.basename(img_path).replace('_leftImg8bit.png', '')
                    
                    # Create pseudo-label
                    pseudo_label = predictions[i]
                    pseudo_label[~mask[i]] = self.config.ignore_index  # Mask low-confidence pixels
                    
                    # Save pseudo-label
                    save_path = os.path.join(self.pseudo_label_dir, f"{base_name}_pseudo_label.npy")
                    np.save(save_path, pseudo_label)
                    pseudo_labels[img_path] = save_path
        
        return pseudo_labels

class PseudoLabelDataset(Dataset):
    def __init__(self, config, pseudo_labels, transform=None):
        self.config = config
        self.transform = transform
        self.images = list(pseudo_labels.keys())
        self.pseudo_labels = list(pseudo_labels.values())
        
        # Apply dataset percentage
        if config.dataset_percentage < 1.0:
            num_samples = int(len(self.images) * config.dataset_percentage)
            self.images = self.images[:num_samples]
            self.pseudo_labels = self.pseudo_labels[:num_samples]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert('RGB'))
        pseudo_label = np.load(self.pseudo_labels[idx])
        
        if self.transform:
            transformed = self.transform(image=image, mask=pseudo_label)
            image = transformed['image']
            pseudo_label = transformed['mask']
        
        return image, pseudo_label.long()

transform = Compose([
    Resize(512, 1024),
    Normalize(mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])