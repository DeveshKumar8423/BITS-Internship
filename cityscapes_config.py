from configs.base_config import BaseConfig

class CityscapesConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        # Update this path to match your actual folder name and structure
        self.data_root = "./data/CityScape"  # <-- Change to your folder name
        self.num_classes = 19  # or the correct number for your setup
        self.class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
            'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]
        self.class_colors = [
            (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
            (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
            ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
            (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32)
        ]
        self.num_workers = 0
        self.image_size = (256, 512)  # Consistent with base config
        # Add any other Cityscapes-specific settings here
        self.run_validation = True  # Enable validation
        self.dataset_percentage = 0.1  # Use 10% of dataset

Config = CityscapesConfig