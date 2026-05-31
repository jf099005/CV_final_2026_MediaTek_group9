import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.RLR_dataset import (
    YOnlySRDataset,
    # YOnlySRValidationDataset,
)
from models.edsr import EDSREnd2EndModel


from ptflops import get_model_complexity_info

def input_constructor(input_res):
    base = torch.randn(1, 1, input_res[0], input_res[1])
    prv  = torch.randn(1, 1, input_res[0], input_res[1])
    nxt  = torch.randn(1, 1, input_res[0], input_res[1])
    return dict(base=base, prv=prv, nxt=nxt)

model = EDSREnd2EndModel(
    in_channels=1,
    out_channels=1,
    num_features=64,
    num_blocks=8,
    scale=2,
)#.to('cuda')


macs, params = get_model_complexity_info(
    model,
    (96, 96),
    input_constructor=input_constructor,
    as_strings=True,
    print_per_layer_stat=True
)

print(macs, params)