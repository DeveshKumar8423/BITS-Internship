import argparse
import logging
from src.utils.utils import load_config, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_evaluation(args):
    from src.evaluation.evaluate import Evaluator
    import torch
    
    # Determine which config to use for model loading by checking the checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Debug: print relevant keys
    print("Relevant checkpoint keys:")
    for key in list(state_dict.keys())[:10]:  # Print first 10 keys
        print(f"  {key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'no shape'}")
    
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
        elif 'decoder.conv3.weight' in key and len(state_dict[key].shape) == 4:
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
    
    # Determine model config based on number of classes AND checkpoint path
    if 'daformer' in args.checkpoint.lower():
        # DAFormer models should use the same config as they were trained with
        if num_classes_from_checkpoint == 19:
            model_config = load_config('configs/cityscapes_config.py')
            print(f"Loading DAFormer with Cityscapes config (19 classes)")
        else:
            model_config = load_config('configs/idd_config.py')
            print(f"Loading DAFormer with IDD config (27 classes)")
    elif num_classes_from_checkpoint == 27:
        # Model was trained with IDD config (27 classes)
        model_config = load_config('configs/idd_config.py')
        print(f"Loading model with IDD config (27 classes)")
    elif num_classes_from_checkpoint == 19:
        # Model was trained with Cityscapes config (19 classes)
        model_config = load_config('configs/cityscapes_config.py')
        print(f"Loading model with Cityscapes config (19 classes)")
    else:
        # Fallback: use checkpoint name to determine config
        if 'daformer' in args.checkpoint.lower():
            model_config = load_config('configs/idd_config.py')  # DAFormer typically uses IDD config
        else:
            model_config = load_config(args.config)
        print(f"Using fallback config detection")
    
    # Load the model with the correct config
    model = load_model(model_config, args.checkpoint)
    
    # Load the dataset config for evaluation (always use the requested config for dataset)
    config = load_config(args.config)
    evaluator = Evaluator(config, model)
    metrics = evaluator.evaluate(args.dataset, args.split)
    print(f"Evaluation Results - mIoU: {metrics['mIoU']:.4f}, Pixel Acc: {metrics['pixel_accuracy']:.4f}")

def run_adapt_train(args):
    from src.training.adapt_train import DomainAdaptationTrainer
    config, source_config, target_config = map(load_config, [args.config, args.source_config, args.target_config])
    
    if args.batch_size: config.batch_size = args.batch_size
    if args.num_epochs: config.num_epochs = args.num_epochs
    if args.dataset_percentage:
        source_config.dataset_percentage = args.dataset_percentage
        target_config.dataset_percentage = args.dataset_percentage
    
    trainer = DomainAdaptationTrainer(config, source_config, target_config)
    trainer.train()

def run_generate_pseudo_labels(args):
    from src.pseudo_labeling import generate_pseudo_labels
    import torch
    
    config = load_config(args.config)
    
    # FIX: Pass the dataset_percentage to the config for the pseudo-label command
    if args.dataset_percentage:
        config.dataset_percentage = args.dataset_percentage
        
    # Use the same model detection logic as evaluation
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Check the number of classes by looking at the decoder weight shape
    num_classes_from_checkpoint = None
    
    for key in state_dict.keys():
        if 'decoder.classifier.weight' in key:
            num_classes_from_checkpoint = state_dict[key].shape[0]
            print(f"Found decoder classifier: {key} with {num_classes_from_checkpoint} classes")
            break
    
    # Determine model config based on number of classes AND checkpoint path
    if 'daformer' in args.checkpoint.lower():
        # DAFormer models should use the same config as they were trained with
        if num_classes_from_checkpoint == 19:
            model_config = load_config('configs/cityscapes_config.py')
            print(f"Loading DAFormer with Cityscapes config (19 classes)")
        else:
            model_config = load_config('configs/idd_config.py')
            print(f"Loading DAFormer with IDD config (27 classes)")
    else:
        # Baseline model uses the same config as the dataset
        model_config = load_config(args.config)
    
    # Load the model with the correct config
    model = load_model(model_config, args.checkpoint)
    generate_pseudo_labels(config, model, args.dataset, args.output_dir)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Evaluation parser
    eval_parser = subparsers.add_parser('evaluate')
    eval_parser.add_argument('--config', required=True)
    eval_parser.add_argument('--checkpoint', required=True)
    eval_parser.add_argument('--dataset', choices=['cityscapes', 'idd'], required=True)
    eval_parser.add_argument('--split', default='val')
    eval_parser.set_defaults(func=run_evaluation)
    
    # Adapt train parser
    adapt_parser = subparsers.add_parser('adapt_train')
    adapt_parser.add_argument('--config', required=True)
    adapt_parser.add_argument('--source_config', required=True)
    adapt_parser.add_argument('--target_config', required=True)
    adapt_parser.add_argument('--batch_size', type=int)
    adapt_parser.add_argument('--dataset_percentage', type=float)
    adapt_parser.add_argument('--num_epochs', type=int)
    adapt_parser.add_argument('--pseudo_labels', default=None)
    adapt_parser.set_defaults(func=run_adapt_train)
    
    # Pseudo label parser
    gen_parser = subparsers.add_parser('generate_pseudo_labels')
    gen_parser.add_argument("--config", required=True)
    gen_parser.add_argument("--checkpoint", required=True)
    gen_parser.add_argument("--output_dir", required=True)
    gen_parser.add_argument("--dataset", required=True)
    # FIX: Add the dataset_percentage argument here
    gen_parser.add_argument('--dataset_percentage', type=float, help='Percentage of dataset to use')
    gen_parser.set_defaults(func=run_generate_pseudo_labels)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()