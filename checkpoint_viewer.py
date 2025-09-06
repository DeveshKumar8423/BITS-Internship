import torch
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter

def view_checkpoints():
    checkpoint_dir = Path("checkpoints/baseline")
    writer = SummaryWriter('runs/checkpoint_comparison')
    
    # Load best model metrics
    best_model = torch.load(checkpoint_dir / "best_model.pth", map_location='cpu')
    writer.add_scalar('Best Model/Loss', best_model.get('loss', 0), 0)
    writer.add_scalar('Best Model/IoU', best_model.get('iou', 0), 0)
    
    # Load epoch checkpoints
    for checkpoint_file in checkpoint_dir.glob("checkpoint_epoch_*.pth"):
        epoch = int(checkpoint_file.stem.split('_')[-1])
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        
        writer.add_scalar('Training/Loss', checkpoint.get('loss', 0), epoch)
        writer.add_scalar('Training/IoU', checkpoint.get('iou', 0), epoch)
        writer.add_scalar('Training/Learning Rate', 
                         checkpoint.get('optimizer', {}).get('param_groups', [{}])[0].get('lr', 0), 
                         epoch)
    
    writer.close()
    print("Checkpoint data converted to TensorBoard format")

if __name__ == "__main__":
    view_checkpoints()