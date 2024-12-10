# Contains stuff for loading in an old iXnos model.
# I'm not sure how much I'll expand upon this. 
import torch
from torch import nn
import pickle
from collections import OrderedDict


class iXnos(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(760, 200),
            nn.Tanh(),
            nn.Linear(200, 1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        # x = self.flatten(x)
        output = self.layers(x)
        return output 


def load_ixnos(pklpath):
    # 
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = iXnos().to(device)

    with open(pklpath, 'rb') as file:
        # Load the pickled data
        data = pickle.load(file, encoding='bytes')
    for idx, val in enumerate(data):
        data[idx] = torch.from_numpy(val).T
    layer_name = "layers"
    labels = [
        f"{layer_name}.0.weight", f"{layer_name}.0.bias", 
        f"{layer_name}.2.weight", f"{layer_name}.2.bias"]

    old_model = OrderedDict(zip(labels, data))

    model.load_state_dict(old_model)
    return model
