import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Import necessary configs and utilities
from configs.idd_config import IDDConfig
from configs.cityscapes_config import CityscapesConfig
from src.utils.utils import load_model
from src.data_loading.idd import IDDDataset
from src.data_loading.transforms import get_val_transforms, decode_segmap

def visualize_comparison(args):
    idd_config = IDDConfig()
    cityscapes_config = CityscapesConfig()
    
    # Load models with correct configs
    print("Loading models...")
    
    # Load baseline model (trained with Cityscapes config)
    baseline_model = load_model(cityscapes_config, args.baseline_ckpt)
    
    # Load adapted model with correct config based on checkpoint
    import torch
    checkpoint = torch.load(args.adapted_ckpt, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Debug: Print all keys to understand structure
    print("Checkpoint keys:")
    for key in state_dict.keys():
        if 'weight' in key and ('decoder' in key or 'classifier' in key or 'conv3' in key):
            print(f"  {key}: {state_dict[key].shape}")
    
    # Check the number of classes by looking at the decoder weight shape
    decoder_weight_key = None
    num_classes_from_checkpoint = None
    
    # Look for classifier weights in different possible locations
    for key in state_dict.keys():
        if 'decoder.classifier.weight' in key:
            # This is the final classifier layer we want
            decoder_weight_key = key
            num_classes_from_checkpoint = state_dict[key].shape[0]
            print(f"Found decoder classifier: {key} with {num_classes_from_checkpoint} classes")
            break
        elif 'conv3.weight' in key and 'decoder' in key and len(state_dict[key].shape) == 4:
            # For DeepLabV3+ models final layer
            decoder_weight_key = key
            num_classes_from_checkpoint = state_dict[key].shape[0]
            print(f"Found decoder conv3: {key} with {num_classes_from_checkpoint} classes")
            break
    
    # If not found, look for any final layer with class outputs
    if num_classes_from_checkpoint is None:
        for key in state_dict.keys():
            if key.endswith('.weight') and len(state_dict[key].shape) == 4:
                # Check if this looks like a final classification layer
                if state_dict[key].shape[2] == 1 and state_dict[key].shape[3] == 1:
                    decoder_weight_key = key
                    num_classes_from_checkpoint = state_dict[key].shape[0]
                    print(f"Found classification layer: {key} with {num_classes_from_checkpoint} classes")
                    break
    
    print(f"Detected {num_classes_from_checkpoint} classes from checkpoint")
    
    # Determine model config based on number of classes
    if num_classes_from_checkpoint == 27:
        # Model was trained with IDD config (27 classes)
        adapted_model = load_model(idd_config, args.adapted_ckpt)
        print(f"Loading adapted model with IDD config (27 classes)")
    else:
        # Model was trained with Cityscapes config (19 classes) or fallback
        adapted_model = load_model(cityscapes_config, args.adapted_ckpt)
        print(f"Loading adapted model with Cityscapes config (19 classes)")
    
    # Load dataset
    dataset = IDDDataset(idd_config, split='val', transform=get_val_transforms(cityscapes_config))
    sample_idx = np.random.randint(0, len(dataset))
    image_tensor, mask_tensor = dataset[sample_idx]
    image_tensor = image_tensor.unsqueeze(0).to(cityscapes_config.device)
    
    # Generate predictions
    print("Generating predictions...")
    baseline_model.eval()
    adapted_model.eval()
    
    with torch.no_grad():
        baseline_output = baseline_model(image_tensor)
        if isinstance(baseline_output, tuple):
            baseline_output = baseline_output[0]
        baseline_pred = torch.argmax(baseline_output, dim=1).squeeze(0).cpu().numpy()
        
        adapted_output = adapted_model(image_tensor)
        if isinstance(adapted_output, tuple):
            adapted_output = adapted_output[0]
        adapted_pred = torch.argmax(adapted_output, dim=1).squeeze(0).cpu().numpy()
    
    # Convert tensors to numpy arrays for visualization
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)
    
    mask_np = mask_tensor.cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Input image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis('off')
    
    # Ground truth
    gt_colored = decode_segmap(mask_np, cityscapes_config.class_colors)
    axes[0, 1].imshow(gt_colored)
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis('off')
    
    # Baseline prediction
    baseline_colored = decode_segmap(baseline_pred, cityscapes_config.class_colors)
    axes[1, 0].imshow(baseline_colored)
    axes[1, 0].set_title("Baseline Prediction")
    axes[1, 0].axis('off')
    
    # Adapted prediction
    adapted_colored = decode_segmap(adapted_pred, cityscapes_config.class_colors)
    axes[1, 1].imshow(adapted_colored)
    axes[1, 1].set_title("Adapted Prediction")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("comparison_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Visualization saved as comparison_visualization.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", required=True, help="Path to baseline model checkpoint")
    parser.add_argument("--adapted_ckpt", required=True, help="Path to adapted model checkpoint")
    args = parser.parse_args()
    
    visualize_comparison(args)