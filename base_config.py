# configs/base_config.py
import os
from dataclasses import dataclass

@dataclass 
class BaseConfig:
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"
    ignore_index: int = 255

    batch_size: int = 4  # Keep manageable
    num_epochs: int = 15  # Much more training for baseline
    dataset_percentage: float = 0.1  # Use 10% of dataset

    augment_prob: float = 0.8

    image_size = (256, 512)  # Good balance
    num_workers: int = 0

    learning_rate: float = 0.0005  # Lower learning rate for stability
    weight_decay: float = 0.001  # More regularization
    patience: int = 8  # More patience

    model_name: str = "deeplabv3plus"
    backbone: str = "resnet101"
    num_classes: int = 19

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

Config = BaseConfig