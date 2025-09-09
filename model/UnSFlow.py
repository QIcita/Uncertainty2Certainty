import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
from einops import rearrange

import torch.nn.functional as F
import torch.fft
from math import sqrt
import numpy as np

from timm.layers import DropPath

from utils import *


from .block.BHTN import HTemporalNet
from .block.UAAHTN import BMoHAttention

###############
class UnSFlow(nn.Module):
    """The proposed M3DFEL framework

    Args:
        args
    """

    def __init__(self, args):
        super(UnSFlow, self).__init__()

        self.args = args
        self.device = torch.device(
            'cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        self.window_size = 4
        self.step_size = 2

        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.features = nn.Sequential(
            *list(model.children())[:-1])  # after avgpool 512x1

        self.bhtn = HTemporalNet(input_dim=512, hidden_dim=512, seq_len=7, period=3)
        
        self.battn_layer = BMoHAttention(
        dim=1024,
        num_heads=8,
        qkv_bias=True,
        qk_norm=True,
        attn_drop=0.1,
        proj_drop=0.1,
        shared_head=2,
        routed_head=3,
        head_dim=16
    )

        self.pwconv = nn.Conv1d(7, 1, 3, 1, 1)
        # classifier
        self.fc = nn.Linear(1024, self.args.num_classes)
        self.Softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = create_sliding_windows(x, window_size=self.window_size, step_size=self.step_size)
        x = rearrange(x, 'b w c t h d -> (b w) c t h d')
        x = self.features(x).squeeze()
        x = rearrange(x, '(b w) c -> b w c', w=7)

        x = self.bhtn(x)       
        ori_x = x
        x = self.battn_layer(x)
        x = torch.sigmoid(x)
        x = ori_x * x
        x = self.pwconv(x).squeeze()
        ##features before the FC layer are used to form t-SNE
        #x_FER =x
        out = self.fc(x)

        return out#, x_FER
    
def create_sliding_windows(x, window_size, step_size):
    b, c, t, h, w = x.shape
    windows = []
    for i in range(0, t - window_size + 1, step_size):
        windows.append(x[:, :, i:i + window_size, :, :])
    return torch.stack(windows, dim=1)

