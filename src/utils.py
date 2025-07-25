import torch
from collections import OrderedDict
from typing import List

def kepler_collate_fn(batch:List):
    """
    Collate function for the Kepler dataset.
    """
    # Separate the elements of each sample tuple (x, y, mask, info) into separate lists
    x1, x2, y, x1_target, x2_target, infos = zip(*batch)

    # Convert lists to tensors
    x1_tensor = torch.stack(x1, dim=0)
    x2_tensor = torch.stack(x2, dim=0)
    y_tensor = torch.stack(y, dim=0)
    x1_target_tensor = torch.stack(x1_target, dim=0)
    x2_target_tensor = torch.stack(x2_target, dim=0)

    return x1_tensor, x2_tensor, y_tensor, x1_target_tensor, x2_target_tensor, infos
class Container(object):
    '''A container class that can be used to store any attributes.'''

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def load_dict(self, dict):
        for key, value in dict.items():
            if getattr(self, key, None) is None:
                setattr(self, key, value)

    def print_attributes(self):
        for key, value in vars(self).items():
            print(f"{key}: {value}")

    def get_dict(self):
        return self.__dict__


def load_checkpoints_ddp(model, checkpoint_path, prefix='', load_backbone=False):
    print(f"****Loading  checkpoint - {checkpoint_path}****")
    state_dict = torch.load(f'{checkpoint_path}', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # print(key)
        while key.startswith('module.'):
            key = key[7:]
        if load_backbone:
            if key.startswith('backbone.'):
                key = key[9:]
            else:
                continue
        key = prefix + key
        # print(key, value.shape)
        new_state_dict[key] = value
    state_dict = new_state_dict

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("number of keys in state dict and model: ", len(state_dict), len(model.state_dict()))
    print("number of missing keys: ", len(missing))
    print("number of unexpected keys: ", len(unexpected))
    print("missing keys: ", missing)
    print("unexpected keys: ", unexpected)
    return model