import importlib.util
import torch
from torch.utils.data import DataLoader
from src.data_loading.transforms import get_train_transforms, get_val_transforms

# NOTE: The imports for the datasets are now INSIDE the get_loader function to prevent circular imports.

def load_config(config_path):
    """Loads a configuration class from a Python file path."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config from {config_path}")
    
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    if hasattr(config_module, 'Config'):
        return config_module.Config()
    raise AttributeError(f"Config file at {config_path} does not have a 'Config' class.")

def get_loader(dataset_name, config, split, batch_size, shuffle=True):
    """Creates and returns a DataLoader for the specified dataset."""
    # Local imports to break the circular dependency
    from src.data_loading.cityscapes import CityscapesDataset
    from src.data_loading.idd import IDDDataset

    transform = get_train_transforms(config) if split == 'train' else get_val_transforms(config)

    if dataset_name == 'cityscapes':
        dataset = CityscapesDataset(config, split=split, transform=transform)
    elif dataset_name == 'idd':
        dataset = IDDDataset(config, split=split, transform=transform)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True
    )

def load_model(config, checkpoint_path=None):
    # This function remains the same as before.
    # It's included here to ensure the file is complete.
    model_name = getattr(config, 'model', 'deeplabv3plus')
    if checkpoint_path and 'daformer' in str(checkpoint_path).lower():
        model_name = 'daformer'

    if model_name == 'daformer':
        from src.models.daformer import DaFormer
        model = DaFormer(config)
    else:
        from src.models.deeplabv3plus import DeepLabV3Plus
        model = DeepLabV3Plus(config)
    
    device = getattr(config, 'device', 'cpu')
    model = model.to(device)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
    
    return model

def save_predictions(predictions, output_dir, filename):
    """
    Saves prediction masks as pseudo-labels.
    
    Args:
        predictions: Tensor of shape (H, W) with class predictions
        output_dir: Directory to save the pseudo-labels
        filename: Filename for the saved pseudo-label
    """
    import numpy as np
    from PIL import Image
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert predictions to numpy array
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    # Convert to uint8 for saving as PNG
    predictions = predictions.astype(np.uint8)
    
    # Create PIL Image and save
    image = Image.fromarray(predictions)
    save_path = os.path.join(output_dir, filename)
    image.save(save_path)
    
    return save_path