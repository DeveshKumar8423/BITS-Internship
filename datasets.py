from src.data_loading.idd import IDDDataset
from src.data_loading.cityscapes import CityscapesDataset

def get_dataset(name, config, split, transform=None):
    if name == 'idd':
        return IDDDataset(config=config, split=split, transform=transform)
    elif name == 'cityscapes':
        return CityscapesDataset(config=config, split=split, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
