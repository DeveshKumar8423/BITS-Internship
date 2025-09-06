# src/utils/label_utils.py

import numpy as np

def cityscapes_label_map():
    """
    Returns a function that maps Cityscapes labelIds to trainIds.
    Maps the 34 original Cityscapes classes to 19 training classes.
    All unlabeled or ignored classes get mapped to 255.
    """
    # Cityscapes label mapping: labelId -> trainId
    # Based on the official Cityscapes mapping
    mapping = {
        0: 255,    # unlabeled
        1: 255,    # ego vehicle
        2: 255,    # rectification border
        3: 255,    # out of roi
        4: 255,    # static
        5: 255,    # dynamic
        6: 255,    # ground
        7: 0,      # road
        8: 1,      # sidewalk
        9: 255,    # parking
        10: 255,   # rail track
        11: 2,     # building
        12: 3,     # wall
        13: 4,     # fence
        14: 255,   # guard rail
        15: 255,   # bridge
        16: 255,   # tunnel
        17: 5,     # pole
        18: 255,   # polegroup
        19: 6,     # traffic light
        20: 7,     # traffic sign
        21: 8,     # vegetation
        22: 9,     # terrain
        23: 10,    # sky
        24: 11,    # person
        25: 12,    # rider
        26: 13,    # car
        27: 14,    # truck
        28: 15,    # bus
        29: 255,   # caravan
        30: 255,   # trailer
        31: 16,    # train
        32: 17,    # motorcycle
        33: 18,    # bicycle
        34: 255,   # license plate
    }

    def map_fn(mask):
        remapped = np.full_like(mask, fill_value=255)
        for k, v in mapping.items():
            remapped[mask == k] = v
        return remapped

    return map_fn

def idd_to_cityscapes_mapping():
    """
    Returns a function that maps IDD labels (27 classes) to Cityscapes labels (19 classes).
    This enables cross-domain evaluation by mapping similar classes together.
    """
    # IDD class index -> Cityscapes class index mapping
    # IDD classes: road, sidewalk, building, wall, fence, pole, traffic light, traffic sign,
    #              vegetation, terrain, sky, person, rider, car, truck, bus, train,
    #              motorcycle, bicycle, autorickshaw, animal, traffic sign back, curb, obstacle,
    #              parking, caravan, trailer
    mapping = {
        0: 0,    # road -> road
        1: 1,    # sidewalk -> sidewalk  
        2: 2,    # building -> building
        3: 3,    # wall -> wall
        4: 4,    # fence -> fence
        5: 5,    # pole -> pole
        6: 6,    # traffic light -> traffic light
        7: 7,    # traffic sign -> traffic sign
        8: 8,    # vegetation -> vegetation
        9: 9,    # terrain -> terrain
        10: 10,  # sky -> sky
        11: 11,  # person -> person
        12: 12,  # rider -> rider
        13: 13,  # car -> car
        14: 14,  # truck -> truck
        15: 15,  # bus -> bus
        16: 16,  # train -> train
        17: 17,  # motorcycle -> motorcycle
        18: 18,  # bicycle -> bicycle
        19: 13,  # autorickshaw -> car (closest match)
        20: 255, # animal -> ignore (no equivalent)
        21: 7,   # traffic sign back -> traffic sign
        22: 1,   # curb -> sidewalk (closest match)
        23: 255, # obstacle -> ignore (no equivalent)
        24: 255, # parking -> ignore (mapped to ignore in cityscapes)
        25: 255, # caravan -> ignore (mapped to ignore in cityscapes)
        26: 255  # trailer -> ignore (mapped to ignore in cityscapes)
    }

    def map_fn(mask):
        remapped = np.full_like(mask, fill_value=255, dtype=mask.dtype)
        for idd_id, cityscapes_id in mapping.items():
            remapped[mask == idd_id] = cityscapes_id
        return remapped

    return map_fn

def idd_label_map():
    """
    Returns the IDD to Cityscapes mapping function for cross-domain evaluation.
    This replaces the old direct IDD mapping.
    """
    return idd_to_cityscapes_mapping()
