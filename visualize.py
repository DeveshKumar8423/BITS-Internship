import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from src.data_loading.transforms import decode_segmap, get_val_transforms
#from src.data_loading.datasets import get_dataset
from src.models.model_factory import get_model
from src.data_loading.cityscapes import CityscapesDataset
from src.data_loading.idd import IDDDataset  # if it exists
from src.data_loading.transforms import decode_segmap

def get_dataset(name, split, transform, config):  # <- added config here
    if name == "cityscapes":
        return CityscapesDataset(config, split=split, transform=transform)
    elif name == "idd":
        return IDDDataset(split=split, transform=transform, config=config)  # <- added config
    else:
        raise ValueError(f"Unknown dataset: {name}")


class Visualizer:
    def __init__(self, config):
        self.config = config
        self.class_colors = config.class_colors
        self.class_names = config.class_names

    def plot_predictions(self, images, gt_masks, pred_masks, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for idx, (img, gt, pred) in enumerate(zip(images, gt_masks, pred_masks)):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img)
            axs[0].set_title("Input Image")
            axs[1].imshow(decode_segmap(gt, self.class_colors))
            axs[1].set_title("Ground Truth")
            axs[2].imshow(decode_segmap(pred, self.class_colors))
            axs[2].set_title("Prediction")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sample_{idx}.png"))

    def plot_class_distribution(self, masks, output_path=None):
        """Plot class distribution from segmentation masks"""
        class_counts = np.zeros(len(self.class_names))
        
        for mask in masks:
            unique, counts = np.unique(mask, return_counts=True)
            for u, c in zip(unique, counts):
                if u < len(class_counts):
                    class_counts[u] += c
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(self.class_names)), class_counts)
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=90)
        plt.ylabel("Pixel Count")
        plt.title("Class Distribution")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
        plt.close()

    def generate_visualizations(self, model, dataset_name, split, num_samples=5, output_dir=None):
        """
        Load a few samples, run inference, and visualize.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)

        val_dataset = get_dataset(dataset_name, split=split, transform=get_val_transforms(self.config), config=self.config)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

        images = []
        gt_masks = []
        pred_masks = []

        with torch.no_grad():
            for idx, (img, mask) in enumerate(tqdm(val_loader, desc="Visualizing")):
                img = img.to(device)
                output = model(img)
                if isinstance(output, tuple):  # If model returns tuple, take the first item
                    output = output[0]
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

                images.append(to_pil_image(img.squeeze(0).cpu()))
                gt_masks.append(mask.squeeze(0).cpu().numpy())
                pred_masks.append(pred)

                if len(images) >= num_samples:
                    break

        self.plot_predictions(images, gt_masks, pred_masks, output_dir)

        return images, gt_masks, pred_masks