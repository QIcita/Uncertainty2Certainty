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
###############
from .block.MoH import MoHAttention
from .block.BHTN import HTemporalNet

from .block.HTN import HybridTemporalNet
from .block.UAAHTN import BMoHAttention

###############
class M3DFEL(nn.Module):
    """The proposed M3DFEL framework

    Args:
        args
    """

    def __init__(self, args):
        super(M3DFEL, self).__init__()

        self.args = args
        self.device = torch.device(
            'cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        #self.bag_size = self.args.num_frames // self.args.instance_length
        #self.instance_length = self.args.instance_length
        self.window_size = 4
        self.step_size = 2

        # backbone networks
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.features = nn.Sequential(
            *list(model.children())[:-1])  # after avgpool 512x1
        #self.lstm = nn.LSTM(input_size=512, hidden_size=512,
         #                 num_layers=2, batch_first=True, bidirectional=True)
        
        self.bhtn = HTemporalNet(input_dim=512, hidden_dim=512, seq_len=7, period=3)
        #self.htn = HybridTemporalNet(input_dim=512, hidden_dim=512, seq_len=7, period=3)
        
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

    #     self.battn_layer = MoHAttention(
    #     dim=1024,
    #     num_heads=8,
    #     qkv_bias=True,
    #     qk_norm=True,
    #     attn_drop=0.1,
    #     proj_drop=0.3,
    #     shared_head=2,
    #     routed_head=2,
    #     head_dim=16
    # )




        # multi head self attention
        self.heads = 8
        self.dim_head = 1024 // self.heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(
            1024, (self.dim_head * self.heads) * 3, bias=False)

        self.norm = DMIN(num_features=1024)
        self.pwconv = nn.Conv1d(7, 1, 3, 1, 1)

        # classifier
        self.fc = nn.Linear(1024, self.args.num_classes)
        #self.fc1 = nn.Linear(1024, 512)
        #self.fc2 = nn.Linear(512, self.args.num_classes)
        self.Softmax = nn.Softmax(dim=-1)



    def MIL(self, x):
        """The Multi Instance Learning Agregation of instances

        Inputs:
            x: [batch, bag_size, 512]
        """
        #self.lstm.flatten_parameters()
        #x, _ = self.lstm(x)
        
        #x = self.htn(x)

        #x, x_short_uncertainty, x_long_uncertainty = self.bhtn(x)
        x = self.bhtn(x)
       
        # [batch, bag_size, 1024]
        ori_x = x

        x = self.battn_layer(x)
        
        # MHSA
        # qkv = self.to_qkv(x).chunk(3, dim=-1)
        # q, k, v = map(lambda t: rearrange(
        #     t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attn = self.attend(dots)
        # x = torch.matmul(attn, v)
        # x = rearrange(x, 'b h n d -> b n (h d)')

        x = self.norm(x)
        x = torch.sigmoid(x)

        x = ori_x * x

        return x

    def forward(self, x):
        """Forward pass through the model
        
        Inputs:
            x: [batch, 16, 3, 112, 112]
        """
       # Initial input shape: [batch, 16, 3, 112, 112]
        x = rearrange(x, 'b t c h w -> b c t h w')

        # After rearrange: [batch, 3, 16, 112, 112]

        x = create_sliding_windows(x, window_size=self.window_size, step_size=self.step_size)
        # After create_sliding_windows: [batch, num_windows, 3, window_size, 112, 112]
        # Example shape: [batch, 7, 3, 4, 112, 112]

        x = rearrange(x, 'b w c t h d -> (b w) c t h d')
        # After rearrange: [(batch * num_windows), 3, window_size, 112, 112]
        # Example shape: [batch * 7, 3, 4, 112, 112]
        
        x = self.features(x).squeeze()
        # After features extraction and squeeze: [(batch * num_windows), 512]
        # Example shape: [batch * 7, 512]
        
        x = rearrange(x, '(b w) c -> b w c', w=7)
        # After rearrange: [batch, num_windows, 512]
        # Example shape: [batch, 7, 512]
        
        x = self.MIL(x)
        # After MIL: [batch, num_windows, 1024]
        # Example shape: [batch, 7, 1024]
        #x_FER =x
        x = self.pwconv(x).squeeze()
        # After pwconv and squeeze: [batch, 1024]
        ##features before the FC layer are used to form t-SNE
        ##temporal avg pooling
        #x_FER =x
        #x = self.fc1(x)
        #out = self.fc2(x)
        out = self.fc(x)
        # After fc: [batch, num_classes]

        return out#, x_FER
    
def create_sliding_windows(x, window_size, step_size):
    b, c, t, h, w = x.shape
    windows = []
    for i in range(0, t - window_size + 1, step_size):
        windows.append(x[:, :, i:i + window_size, :, :])
    return torch.stack(windows, dim=1)
########################################################################################
