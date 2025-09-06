import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from src.data_loading.cityscapes import CityscapesDataset
from src.data_loading.idd import IDDDataset
from src.models.daformer import DaFormer
from src.utils.metrics import compute_iou
from src.utils.logger import Logger
from src.data_loading.transforms import get_train_transforms, get_val_transforms

class SpatialConsistencyLoss(nn.Module):
    """Loss to encourage spatial smoothness in predictions"""
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred):
        # Calculate gradient magnitudes in x and y directions
        grad_x = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        grad_y = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        
        # Return mean gradient magnitude as smoothness loss
        return self.weight * (grad_x.mean() + grad_y.mean())

class ConfidenceBasedLoss(nn.Module):
    """Loss that weights pixels based on prediction confidence"""
    def __init__(self, ignore_index=255, confidence_threshold=0.8):
        super().__init__()
        self.ignore_index = ignore_index
        self.confidence_threshold = confidence_threshold
        
    def forward(self, logits, targets):
        # Get prediction probabilities
        probs = F.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        
        # Create confidence mask
        confident_mask = max_probs > self.confidence_threshold
        valid_mask = (targets != self.ignore_index)
        final_mask = confident_mask & valid_mask
        
        if final_mask.sum() == 0:
            return F.cross_entropy(logits, targets, ignore_index=self.ignore_index)
        
        # Apply loss only to confident predictions
        loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction='none')
        weighted_loss = (loss * final_mask.float()).sum() / final_mask.sum()
        
        return weighted_loss

class DomainAdaptationTrainer:
    def __init__(self, config, source_config, target_config):
        self.config = config
        self.source_config = source_config
        self.target_config = target_config
        self.device = torch.device(config.device)
        self.logger = Logger(config.log_dir)

        # Initialize model
        self.model = DaFormer(config).to(self.device)

        # Enhanced loss functions
        class_weights = torch.ones(config.num_classes).to(self.device)
        class_weights[0] = 3.0  # Road class - most important
        class_weights[8] = 2.0  # Vegetation
        class_weights[10] = 2.0  # Sky
        
        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index, weight=class_weights)
        self.confidence_loss = ConfidenceBasedLoss(ignore_index=config.ignore_index, confidence_threshold=0.9)
        self.spatial_loss = SpatialConsistencyLoss(weight=0.05)
        self.domain_criterion = nn.BCEWithLogitsLoss()

        # Optimizer with careful learning rates
        self.optimizer = Adam([
            {'params': self.model.layer0.parameters(), 'lr': config.learning_rate * 0.1},
            {'params': self.model.layer1.parameters(), 'lr': config.learning_rate * 0.1},
            {'params': self.model.layer2.parameters(), 'lr': config.learning_rate * 0.1},
            {'params': self.model.layer3.parameters(), 'lr': config.learning_rate * 0.1}, 
            {'params': self.model.layer4.parameters(), 'lr': config.learning_rate * 0.1},
            {'params': self.model.decoder.parameters(), 'lr': config.learning_rate},
            {'params': self.model.domain_discriminator.parameters(), 'lr': config.learning_rate * 0.5},
        ], weight_decay=config.weight_decay)

        # Scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.8, patience=2, verbose=True)

        # Initialize datasets
        self._init_datasets()

        # Training state
        self.best_iou = 0.0
        self.epochs_no_improve = 0

    def _init_datasets(self):
        # Source dataset (Cityscapes)
        source_transform = get_train_transforms(self.config)
        self.source_dataset = CityscapesDataset(self.source_config, split='train', transform=source_transform)

        # Target dataset (IDD)
        target_transform = get_train_transforms(self.config)
        self.target_dataset = IDDDataset(self.target_config, split='train', transform=target_transform)

        if len(self.target_dataset) == 0:
            raise ValueError("Target dataset is empty! Check your data path and split.")

        print("Target dataset size:", len(self.target_dataset))

        # Create dataloaders
        self.source_loader = DataLoader(
            self.source_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.target_loader = DataLoader(
            self.target_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )

        # Validation dataset (IDD)
        val_transform = get_val_transforms(self.config)
        self.val_dataset = IDDDataset(self.target_config, split='val', transform=val_transform)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def train(self):
        for epoch in range(self.config.num_epochs):
            self._train_epoch(epoch)
            val_iou = self._validate(epoch)
            
            self.scheduler.step(val_iou)

            is_best = val_iou > self.best_iou
            if is_best:
                self.best_iou = val_iou
                self.epochs_no_improve = 0
                print(f"New best IoU: {self.best_iou:.4f}. Saving model.")
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_no_improve += 1
                self._save_checkpoint(epoch, is_best=False)

            if self.epochs_no_improve >= self.config.patience:
                print(f"Early stopping triggered after {self.config.patience} epochs with no improvement.")
                break
        
        print(f"Training completed. Best validation IoU: {self.best_iou:.4f}")

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, total_seg_loss, total_spatial_loss, total_domain_loss = 0.0, 0.0, 0.0, 0.0

        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader)

        num_batches = min(len(self.source_loader), len(self.target_loader))
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

        for batch_idx in progress_bar:
            try:
                source_images, source_labels = next(source_iter)
                target_images, _ = next(target_iter)
            except StopIteration:
                break

            source_images, source_labels = source_images.to(self.device), source_labels.to(self.device)
            target_images = target_images.to(self.device)

            self.optimizer.zero_grad()

            # Very careful domain adaptation - start slow and build up
            alpha = min(0.5, (epoch + batch_idx / num_batches) / (self.config.num_epochs * 2))

            # Source forward pass with enhanced losses
            source_outputs, source_domain_pred = self.model(source_images, adapt=True, alpha=alpha)
            
            # Resize source outputs to match labels
            source_outputs = F.interpolate(source_outputs, size=source_labels.shape[-2:], 
                                         mode='bilinear', align_corners=False)
            
            # Multiple loss components for source
            seg_loss = self.confidence_loss(source_outputs, source_labels)
            spatial_loss = self.spatial_loss(F.softmax(source_outputs, dim=1))
            
            # Target forward pass - focus on consistency
            target_outputs, target_domain_pred = self.model(target_images, adapt=True, alpha=alpha)
            target_spatial_loss = self.spatial_loss(F.softmax(target_outputs, dim=1))
            
            # Domain loss with careful weighting
            batch_size = source_images.size(0)
            source_domain_labels = torch.zeros(batch_size, 1, device=self.device)
            target_domain_labels = torch.ones(batch_size, 1, device=self.device)
            
            source_domain_loss = self.domain_criterion(source_domain_pred, source_domain_labels)
            target_domain_loss = self.domain_criterion(target_domain_pred, target_domain_labels)
            domain_loss = (source_domain_loss + target_domain_loss) * 0.5
            
            # Combined loss with careful weighting
            combined_spatial_loss = spatial_loss + target_spatial_loss
            total_loss_batch = (seg_loss + 
                              0.1 * combined_spatial_loss + 
                              alpha * 0.3 * domain_loss)  # Reduced domain weight
            
            total_loss_batch.backward()
            
            # Gradient clipping for stability  
            from torch.nn.utils.clip_grad import clip_grad_norm_
            clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            self.optimizer.step()

            total_loss += total_loss_batch.item()
            total_seg_loss += seg_loss.item()
            total_spatial_loss += combined_spatial_loss.item()
            total_domain_loss += domain_loss.item()

            progress_bar.set_postfix({
                'loss': f'{total_loss_batch.item():.3f}',
                'seg': f'{seg_loss.item():.3f}', 
                'spatial': f'{combined_spatial_loss.item():.3f}',
                'domain': f'{domain_loss.item():.3f}',
                'α': f'{alpha:.3f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_seg_loss = total_seg_loss / num_batches
        avg_spatial_loss = total_spatial_loss / num_batches
        avg_domain_loss = total_domain_loss / num_batches
        
        self.logger.log({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'seg_loss': avg_seg_loss,
            'spatial_loss': avg_spatial_loss,
            'domain_loss': avg_domain_loss
        })
        
        print(f"Training - Loss: {avg_loss:.4f}, Seg: {avg_seg_loss:.4f}, Spatial: {avg_spatial_loss:.4f}, Domain: {avg_domain_loss:.4f}")

    def _validate(self, epoch):
        self.model.eval()
        total_iou = 0.0
        valid_batches = 0
        
        progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch + 1}")

        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs, _ = self.model(images, adapt=False)
                outputs = F.interpolate(outputs, size=labels.shape[-2:], 
                                      mode='bilinear', align_corners=False)
                
                # Apply post-processing for smoother predictions
                outputs = self._apply_postprocessing(outputs)
                preds = torch.argmax(outputs, dim=1)

                iou = compute_iou(preds, labels, self.config.num_classes, self.config.ignore_index)
                if iou is not None and not torch.isnan(torch.tensor(iou)):
                    total_iou += iou
                    valid_batches += 1
                
                progress_bar.set_postfix({'batch_iou': f'{iou:.4f}' if iou is not None else 'nan'})

        mean_iou = total_iou / valid_batches if valid_batches > 0 else 0.0
        
        self.logger.log({'epoch': epoch + 1, 'val_iou': mean_iou})
        print(f"Validation IoU: {mean_iou:.4f}")
        return mean_iou

    def _apply_postprocessing(self, outputs):
        """Apply smoothing and confidence filtering to predictions"""
        # Apply Gaussian smoothing to reduce noise
        batch_size, num_classes, height, width = outputs.shape
        
        # Create Gaussian kernel for smoothing
        kernel_size = 5
        sigma = 1.0
        kernel = self._gaussian_kernel(kernel_size, sigma).to(outputs.device)
        
        # Apply smoothing to each class channel
        smoothed_outputs = torch.zeros_like(outputs)
        for b in range(batch_size):
            for c in range(num_classes):
                channel = outputs[b:b+1, c:c+1, :, :]
                smoothed = F.conv2d(channel, kernel, padding=kernel_size//2)
                smoothed_outputs[b, c, :, :] = smoothed.squeeze()
        
        return smoothed_outputs
    
    def _gaussian_kernel(self, size, sigma):
        """Create 2D Gaussian kernel for smoothing"""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = coords**2
        g = (-g / (2 * sigma**2)).exp()
        g /= g.sum()
        kernel = g[:, None] * g[None, :]
        return kernel.unsqueeze(0).unsqueeze(0)

    def _save_checkpoint(self, epoch, is_best):
        save_path = os.path.join("checkpoints", "daformer")
        os.makedirs(save_path, exist_ok=True)
        
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_iou': self.best_iou,
        }

        if is_best:
            torch.save(state, os.path.join(save_path, "best_model.pth"))
        else:
            pass