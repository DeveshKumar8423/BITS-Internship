# configs/idd_config.py
from configs.base_config import BaseConfig

class IDDConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.data_root = "./data/idd"
        self.num_classes = 27  # IDD has 27 classes (will be mapped to 19 for evaluation)
        self.class_names = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
            "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
            "motorcycle", "bicycle", "autorickshaw", "animal", "traffic sign back", "curb", "obstacle",
            "parking", "caravan", "trailer"
        ]
        
        # FIX: The class_colors list MUST have the same number of entries as num_classes.
        # It was 19, now it is 27.
        self.class_colors = [
            # Original 19 Cityscapes Colors
            (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
            (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
            (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
            (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
            # Added placeholder colors for the remaining 8 IDD classes
            (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128)
        ]
        
        self.image_size = (256, 512)  # Consistent with base config
        self.model = 'daformer'
        self.dataset_percentage = 0.1  # Use 10% of dataset
        self.run_validation = True  # Enable validation

Config = IDDConfig