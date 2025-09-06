import torch
from torch.utils.tensorboard.writer import SummaryWriter
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

class TensorBoardLogger:
    def __init__(self, log_dir='runs/training_metrics'):
        self.writer = SummaryWriter(log_dir)
        self.log_dir = Path(log_dir)
        
    def log_metrics(self, metrics, step):
        """Log training metrics to TensorBoard"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Metrics/{key}', value, step)
            elif isinstance(value, torch.Tensor):
                self.writer.add_scalar(f'Metrics/{key}', value.item(), step)
    
    def log_images(self, images, predictions, targets, step, max_images=3):
        """Log images with their predictions and ground truth"""
        # Convert tensors to numpy arrays if needed
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
            
        # Log only first few images
        for idx in range(min(max_images, images.shape[0])):
            self.writer.add_image(f'Batch_{step}/Image_{idx}/Input', 
                                images[idx], step)
            self.writer.add_image(f'Batch_{step}/Image_{idx}/Prediction', 
                                predictions[idx], step)
            self.writer.add_image(f'Batch_{step}/Image_{idx}/Ground_Truth', 
                                targets[idx], step)
    
    def log_histogram(self, name, values, step):
        """Log histogram of model parameters or other values"""
        self.writer.add_histogram(name, values, step)
    
    def convert_json_logs(self, json_path):
        """Convert JSON logs to TensorBoard format"""
        with open(json_path) as f:
            logs = json.load(f)
        
        for entry in logs['train']:
            if 'epoch' in entry:
                epoch = entry['epoch']
                if 'train_loss' in entry:
                    self.writer.add_scalar('Loss/train', entry['train_loss'], epoch)
                if 'train_iou' in entry:
                    self.writer.add_scalar('IoU/train', entry['train_iou'], epoch)
                if 'learning_rate' in entry:
                    self.writer.add_scalar('Learning_Rate', entry['learning_rate'], epoch)
            elif 'val_iou' in entry:
                self.writer.add_scalar('IoU/validation', entry['val_iou'], epoch)
    
    def log_model_graph(self, model, input_shape=(1, 3, 512, 1024)):
        """Log model architecture graph"""
        device = next(model.parameters()).device
        dummy_input = torch.rand(input_shape).to(device)
        self.writer.add_graph(model, dummy_input)
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()

def main():
    logger = TensorBoardLogger()
    json_path = Path("logs/training_log.json")
    logger.convert_json_logs(json_path)
    logger.close()

if __name__ == "__main__":
    main()