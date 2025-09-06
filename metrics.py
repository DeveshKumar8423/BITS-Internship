# src/utils/metrics.py
import torch
import numpy as np

def compute_iou(pred, target, num_classes, ignore_index=255):
    """
    Computes the Intersection over Union (IoU) metric.
    """
    pred = pred.view(-1)
    target = target.view(-1)

    mask = (target != ignore_index)
    pred = pred[mask]
    target = target[mask]
    
    if pred.shape[0] == 0:
        return 0.0

    confusion_matrix = torch.bincount(
        target.long() * num_classes + pred.long(),
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes).to(pred.device)

    intersection = torch.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(dim=1)
    predicted_set = confusion_matrix.sum(dim=0)
    union = ground_truth_set + predicted_set - intersection

    iou = intersection / (union.float() + 1e-6)
    
    present_classes = ground_truth_set > 0
    
    if present_classes.sum() == 0:
        return 0.0

    return iou[present_classes].mean().item()


def compute_pixel_accuracy(pred, target, ignore_index=255):
    """Computes pixel accuracy."""
    pred = pred.view(-1)
    target = target.view(-1)
    
    mask = (target != ignore_index)
    pred = pred[mask]
    target = target[mask]
    
    if target.shape[0] == 0:
        return 0.0
        
    correct = torch.sum(pred == target)
    total = target.shape[0] # This is a plain integer
    
    # FIX: Removed the incorrect .float() call on the integer variable 'total'.
    # PyTorch can correctly divide a tensor by a standard number.
    return (correct.float() / total).item()