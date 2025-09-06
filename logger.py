# src/utils/logger.py
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'training_log.json')
        self.metrics = {
            'train': [],
            'val': []
        }
        
        # Load existing log if it exists
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                self.metrics = json.load(f)
    
    def log(self, metrics_dict):
        """Log metrics to file"""
        self.metrics['train'].append(metrics_dict)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def plot_metrics(self):
        """Plot training and validation metrics"""
        if not self.metrics['train']:
            return
        
        epochs = [m['epoch'] for m in self.metrics['train']]
        
        # Plot loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, [m['train_loss'] for m in self.metrics['train']], label='Train')
        if 'val_loss' in self.metrics['train'][0]:
            plt.plot(epochs, [m['val_loss'] for m in self.metrics['train']], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss')
        
        # Plot IoU
        plt.subplot(1, 2, 2)
        plt.plot(epochs, [m['train_iou'] for m in self.metrics['train']], label='Train')
        if 'val_iou' in self.metrics['train'][0]:
            plt.plot(epochs, [m['val_iou'] for m in self.metrics['train']], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.title('Training IoU')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_metrics.png'))
        plt.close()