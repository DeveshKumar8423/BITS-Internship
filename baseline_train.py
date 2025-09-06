import os
import glob
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
import argparse

# Local imports that DON'T create a circle
from src.models.deeplabv3plus import DeepLabV3Plus
from src.utils.metrics import compute_iou
from src.utils.utils import load_config
from src.data_loading.transforms import get_train_transforms, get_val_transforms
from src.utils.label_utils import cityscapes_label_map
from torch.nn.utils.clip_grad import clip_grad_norm_

# ====================================================================================
#  DEFINITIVE CityscapesDataset CLASS (Included directly in this file)
# ====================================================================================
class CityscapesDataset(Dataset):
    def __init__(self, config, split='train', transform=None):
        self.config = config
        self.split = split
        
        self.image_dir = os.path.join(config.data_root, "leftImg8bit", split)
        self.mask_dir = os.path.join(config.data_root, "gtFine", split)

        # Robustly find all image files
        img_pattern = os.path.join(self.image_dir, '**', '*_leftImg8bit.png')
        self.image_paths = sorted(glob.glob(img_pattern, recursive=True))

        if hasattr(config, 'dataset_percentage') and config.dataset_percentage < 1.0:
            num_samples = int(len(self.image_paths) * config.dataset_percentage)
            self.image_paths = self.image_paths[:num_samples]

        # This line now creates the correct mask filename, e.g., "..._gtFine_labelIds.png"
        self.mask_paths = [
            p.replace("_leftImg8bit.png", "_gtFine_labelIds.png").replace("leftImg8bit", "gtFine")
            for p in self.image_paths
        ]
        
        print(f"[INFO] Found {len(self.image_paths)} images in {self.image_dir} (Split: {split})")

        if transform:
            self.transform = transform
        else:
            self.transform = get_train_transforms(config) if split == 'train' else get_val_transforms(config)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        if not os.path.exists(mask_path):
            print(f"[ERROR] Mask file does not exist: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image, mask = np.array(image), np.array(mask)
        
        # Apply Cityscapes label mapping to convert raw labelIds to trainIds
        label_mapper = cityscapes_label_map()
        mask = label_mapper(mask)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        return image, torch.from_numpy(np.array(mask)).long()

# ====================================================================================
#  BaselineTrainer CLASS
# ====================================================================================
class BaselineTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(getattr(config, 'device', 'cpu'))
        self.model = DeepLabV3Plus(config).to(self.device)
        
        # Class weights for better training - give more weight to important classes
        class_weights = torch.tensor([
            3.0,   # road
            2.0,   # sidewalk  
            1.5,   # building
            1.0,   # wall
            1.0,   # fence
            2.0,   # pole
            2.5,   # traffic light
            2.5,   # traffic sign
            2.0,   # vegetation
            1.5,   # terrain
            2.0,   # sky
            3.0,   # person
            3.0,   # rider
            3.0,   # car
            2.5,   # truck
            2.5,   # bus
            2.0,   # train
            2.5,   # motorcycle
            2.5    # bicycle
        ], device=self.device)
        
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=getattr(config, 'ignore_index', 255),
            weight=class_weights
        )
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        self.checkpoint_dir = Path(config.checkpoint_dir) / "baseline"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize DataLoaders using the class defined above
        self.train_loader = DataLoader(
            CityscapesDataset(config, split='train'),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=getattr(config, 'num_workers', 0),
            drop_last=True
        )
        self.val_loader = DataLoader(
            CityscapesDataset(config, split='val'),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=getattr(config, 'num_workers', 0)
        )
        self.best_iou = 0.0

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Add gradient clipping for stability
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
        return avg_loss

    def _validate(self, epoch):
        self.model.eval()
        total_iou = 0.0
        num_valid_batches = 0
        
        if not self.val_loader or len(self.val_loader) == 0:
            print("Validation loader is empty. Skipping validation.")
            return 0.0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                predictions = outputs.argmax(dim=1)
                iou = compute_iou(predictions, labels, self.config.num_classes, getattr(self.config, 'ignore_index', 255))
                if iou is not None and iou > 0: 
                    total_iou += iou
                    num_valid_batches += 1

        if num_valid_batches > 0:
            mean_iou = total_iou / num_valid_batches
        else:
            mean_iou = 0.0
            
        print(f"Validation IoU: {mean_iou:.4f}")
        return mean_iou

    def train(self):
        print("Starting improved baseline training...")
        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch(epoch)
            val_iou = self._validate(epoch)
            
            # Update learning rate scheduler
            self.scheduler.step(val_iou)
            
            if val_iou > self.best_iou:
                self.best_iou = val_iou
                print(f"New best model with IoU: {self.best_iou:.4f}. Saving checkpoint.")
                torch.save({'state_dict': self.model.state_dict()}, self.checkpoint_dir / "best_model.pth")
        
        # Save final model if no validation data was available
        if self.best_iou == 0.0:
            print("No validation data available. Saving final model checkpoint.")
            torch.save({'state_dict': self.model.state_dict()}, self.checkpoint_dir / "best_model.pth")
        
        print(f"Finished training. Best IoU: {self.best_iou:.4f}")

# ====================================================================================
#  MAIN EXECUTION BLOCK
# ====================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Model Training")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file.")
    parser.add_argument('--batch_size', type=int, help="Override batch size in config.")
    parser.add_argument('--num_epochs', type=int, help="Override number of epochs in config.")
    parser.add_argument('--dataset_percentage', type=float, help="Override dataset percentage in config.")
    args = parser.parse_args()

    config = load_config(args.config)
    
    if args.batch_size: config.batch_size = args.batch_size
    if args.num_epochs: config.num_epochs = args.num_epochs
    if args.dataset_percentage: config.dataset_percentage = args.dataset_percentage

    print("Starting Baseline Training...")
    trainer = BaselineTrainer(config)
    trainer.train()