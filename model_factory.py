from src.models.daformer import DaFormer
from src.models.deeplabv3plus import DeepLabV3Plus

def get_model(config):
    if config.model == 'daformer':
        return DaFormer(config)
    elif config.model == 'deeplabv3plus':
        return DeepLabV3Plus(config)
    else:
        raise ValueError(f"Unsupported model: {config.model}")
