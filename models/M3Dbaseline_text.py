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
        self.lstm = nn.LSTM(input_size=512, hidden_size=512,
                           num_layers=2, batch_first=True, bidirectional=True)
        
        
        #self.MLLA = MLLABlock(dim=512, input_resolution=7, num_heads=4, mlp_ratio=4.0) #1
        #self.graph_block = GraphBlock(c_out=512, d_model=512,  seq_len=7, gcn_depth=2) #2

        #self.BiIlstm = BiImprovedLSTM(input_dim=512, hidden_dim=512, num_layers=3)
        #self.BIGRU = BiMSSGCellWithGRU(512, 512, 3)
        #self.BiAlstm = BiAdaptiveLSTMWithRes(512, 512, 2)
        #self.BiAlstmSSG = BiAdaptiveLSTMWithSSGRes(512, 512, 3)
        #self.BiAlstmWSSG = BiAdaptiveLSTMWithWeakSignalSSGRes(512, 512, 2)
        

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
        self.Softmax = nn.Softmax(dim=-1)



    def MIL(self, x):
        """The Multi Instance Learning Agregation of instances

        Inputs:
            x: [batch, bag_size, 512]
        """
        #self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        #x = self.BiAlstm(x)
        # [batch, bag_size, 1024]
        ori_x = x
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
        x_FER =x
        out = self.fc(x)
        # After fc: [batch, num_classes]

        return out, x_FER
    
def create_sliding_windows(x, window_size, step_size):
    b, c, t, h, w = x.shape
    windows = []
    for i in range(0, t - window_size + 1, step_size):
        windows.append(x[:, :, i:i + window_size, :, :])
    return torch.stack(windows, dim=1)
########################################################################################


class ImprovedLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ImprovedLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim)
        
        # 使用多层感知器的弱信号门
        self.W_s = nn.Linear(input_dim, hidden_dim)
        self.U_s = nn.Linear(hidden_dim, hidden_dim)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state

        i_t = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
        c_hat_t = torch.tanh(self.W_c(x) + self.U_c(h_prev))
        
        # 弱信号门
        s_t = torch.sigmoid(self.W_s(x) + self.U_s(h_prev))
        s_t = self.MLP(s_t)
        
        c_t = f_t * c_prev + i_t * c_hat_t * s_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class BiImprovedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(BiImprovedLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fwd_layers = nn.ModuleList([ImprovedLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.bwd_layers = nn.ModuleList([ImprovedLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        h_fwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        c_fwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        
        h_bwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        c_bwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]

        output_fwd = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h_fwd[layer], c_fwd[layer] = self.fwd_layers[layer](x_t, (h_fwd[layer], c_fwd[layer]))
                x_t = h_fwd[layer]
            output_fwd.append(h_fwd[-1])

        output_bwd = []
        for t in range(seq_len-1, -1, -1):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h_bwd[layer], c_bwd[layer] = self.bwd_layers[layer](x_t, (h_bwd[layer], c_bwd[layer]))
                x_t = h_bwd[layer]
            output_bwd.append(h_bwd[-1])

        output_bwd.reverse()
        output = torch.cat([torch.stack(output_fwd, dim=1), torch.stack(output_bwd, dim=1)], dim=-1)

        return output




########################################################################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RoPE(nn.Module):
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class LinearAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution, dim))

    def forward(self, x):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k = qk[0], qk[1]

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, n, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, n, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = x.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, n, c).permute(0, 2, 1)
        x = x + self.lepe(v).permute(0, 2, 1).reshape(b, n, c)

        return x


class MLLABlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=4, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, L, C = x.shape
        H = self.input_resolution
        assert L == H, "input feature has wrong size"

        x = x + self.cpe1(x.permute(0, 2, 1)).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).permute(0, 2, 1)
        x = self.act(self.dwc(x)).permute(0, 2, 1)

        x = self.attn(x)

        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

########################################################################################

# https://github.com/YoZhibo/MSGNet
# MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting, AAAI 2024
# https://arxiv.org/pdf/2401.00423
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class self_attention(nn.Module):
    def __init__(self, attention, d_model ,n_heads):
        super(self_attention, self).__init__()
        d_keys =  d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention( attention_dropout = 0.1)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads


    def forward(self, queries ,keys ,values, attn_mask= None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
                    queries,
                    keys,
                    values,
                    attn_mask
                )
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        return out , attn


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        # return V.contiguous()
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

# multi-head attention module
class Attention_Block(nn.Module):
    def __init__(self,  d_model, d_ff=None, n_heads=8, dropout=0.1, activation="relu"):
        super(Attention_Block, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = self_attention(FullAttention, d_model, n_heads=n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        # x = torch.einsum('ncwl,wv->nclv',(x,A)
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho


# adaptive graph convolution module
class GraphBlock(nn.Module):
    def __init__(self, c_out , d_model, seq_len , conv_channel=32, skip_channel=32,
                        gcn_depth=32, dropout=0.1, propalpha=0.1,node_dim=10):
        super(GraphBlock, self).__init__()

        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out), requires_grad=True)
        self.start_conv = nn.Conv2d(1 , conv_channel, (d_model - c_out + 1, 1))
        self.gconv1 = mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        self.gelu = nn.GELU()
        self.end_conv = nn.Conv2d(skip_channel, seq_len , (1, seq_len ))
        self.linear = nn.Linear(c_out, d_model)
        self.norm = nn.LayerNorm(d_model)


    # x in (B, T, d_model)
    # Here we use a mlp to fit a complex mapping f (x)
    def forward(self, x):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        out = x.unsqueeze(1).transpose(2, 3)
        out = self.start_conv(out)
        out = self.gelu(self.gconv1(out , adp))
        out = self.end_conv(out).squeeze()
        out = self.linear(out)

        return self.norm(x + out)
    
#######################################################################

class AdaptiveLSTMCellWithSSGRes(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AdaptiveLSTMCellWithSSGRes, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # LSTM gates
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim)

        # Weak signal gate
        self.W_s = nn.Linear(input_dim, hidden_dim)
        self.U_s = nn.Linear(hidden_dim, hidden_dim)

        # Adaptive weight computation
        self.alpha_mlp = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Residual path (deep network)
        self.residual_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # State-space update layer
        self.ssg_update = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, hidden_state, ssg_state=None):
        h_prev, c_prev = hidden_state

        i_t = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
        c_hat_t = torch.tanh(self.W_c(x) + self.U_c(h_prev))

        # Adaptive weight α_t
        combined_input = torch.cat([x, h_prev], dim=-1)
        alpha_t = self.alpha_mlp(combined_input)

        # Weak signal gate and state-space update
        s_t = torch.sigmoid(self.W_s(x) + self.U_s(h_prev))
        if ssg_state is not None:
            ssg_state = self.ssg_update(ssg_state + h_prev)
        else:
            ssg_state = s_t  # Initialize if not provided

        # Combine state-space and adaptive gating
        c_t = f_t * c_prev + i_t * c_hat_t * s_t * alpha_t * ssg_state

        # Deep residual connection
        residual = self.residual_layers(h_prev)
        c_t = c_t + residual

        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t, ssg_state


class BiAdaptiveLSTMWithSSGRes(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(BiAdaptiveLSTMWithSSGRes, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define forward and backward adaptive LSTM cells with state-space integration
        self.fwd_layers = nn.ModuleList(
            [AdaptiveLSTMCellWithSSGRes(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )
        self.bwd_layers = nn.ModuleList(
            [AdaptiveLSTMCellWithSSGRes(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Initialize forward and backward hidden states and state-space states
        h_fwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        c_fwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        ssg_fwd = [None for _ in range(self.num_layers)]

        h_bwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        c_bwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        ssg_bwd = [None for _ in range(self.num_layers)]

        # Forward pass
        output_fwd = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h_fwd[layer], c_fwd[layer], ssg_fwd[layer] = self.fwd_layers[layer](x_t, (h_fwd[layer], c_fwd[layer]), ssg_fwd[layer])
                x_t = h_fwd[layer]
            output_fwd.append(h_fwd[-1])

        # Backward pass
        output_bwd = []
        for t in range(seq_len - 1, -1, -1):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h_bwd[layer], c_bwd[layer], ssg_bwd[layer] = self.bwd_layers[layer](x_t, (h_bwd[layer], c_bwd[layer]), ssg_bwd[layer])
                x_t = h_bwd[layer]
            output_bwd.append(h_bwd[-1])

        output_bwd.reverse()

        # Concatenate forward and backward outputs
        output = torch.cat([torch.stack(output_fwd, dim=1), torch.stack(output_bwd, dim=1)], dim=-1)

        return output



#########################################################
class AdaptiveLSTMCellWithRes(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AdaptiveLSTMCellWithRes, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim)

        # 弱信号门
        self.W_s = nn.Linear(input_dim, hidden_dim)
        self.U_s = nn.Linear(hidden_dim, hidden_dim)

        # 自适应权重计算
        self.alpha_mlp = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 残差路径（多层网络）
        self.residual_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state

        i_t = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
        c_hat_t = torch.tanh(self.W_c(x) + self.U_c(h_prev))

        # 计算自适应权重α_t
        combined_input = torch.cat([x, h_prev], dim=-1)
        alpha_t = self.alpha_mlp(combined_input)

        # 弱信号门和自适应调节
        s_t = torch.sigmoid(self.W_s(x) + self.U_s(h_prev))
        c_t = f_t * c_prev + i_t * c_hat_t * s_t * alpha_t

        # 添加深度残差连接
        residual = self.residual_layers(h_prev)
        c_t = c_t + residual  # 残差路径的结合

        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class BiAdaptiveLSTMWithRes(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(BiAdaptiveLSTMWithRes, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 定义前向和后向的自适应 LSTM 单元
        self.fwd_layers = nn.ModuleList(
            [AdaptiveLSTMCellWithRes(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )
        self.bwd_layers = nn.ModuleList(
            [AdaptiveLSTMCellWithRes(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 初始化前向和后向的隐藏状态和细胞状态
        h_fwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        c_fwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]

        h_bwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        c_bwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]

        # 前向传播
        output_fwd = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h_fwd[layer], c_fwd[layer] = self.fwd_layers[layer](x_t, (h_fwd[layer], c_fwd[layer]))
                x_t = h_fwd[layer]
            output_fwd.append(h_fwd[-1])

        # 反向传播
        output_bwd = []
        for t in range(seq_len - 1, -1, -1):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h_bwd[layer], c_bwd[layer] = self.bwd_layers[layer](x_t, (h_bwd[layer], c_bwd[layer]))
                x_t = h_bwd[layer]
            output_bwd.append(h_bwd[-1])

        output_bwd.reverse()

        # 将前向和后向的输出拼接
        output = torch.cat([torch.stack(output_fwd, dim=1), torch.stack(output_bwd, dim=1)], dim=-1)

        return output
########################################################################################
class AdaptiveLSTMCellWithWeakSignalSSGRes(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AdaptiveLSTMCellWithWeakSignalSSGRes, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # LSTM gates
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim)

        # Weak signal gate
        self.W_s = nn.Linear(input_dim, hidden_dim)
        self.U_s = nn.Linear(hidden_dim, hidden_dim)

        # Adaptive weight computation
        self.alpha_mlp = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Residual path (deep network)
        self.residual_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # State-space update layer
        self.ssg_update = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, hidden_state, ssg_state=None):
        h_prev, c_prev = hidden_state

        i_t = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
        c_hat_t = torch.tanh(self.W_c(x) + self.U_c(h_prev))

        # Adaptive weight α_t
        combined_input = torch.cat([x, h_prev], dim=-1)
        alpha_t = self.alpha_mlp(combined_input)

        # Weak signal gate and state-space update
        s_t = torch.sigmoid(self.W_s(x) + self.U_s(h_prev))
        if ssg_state is not None:
            ssg_state = self.ssg_update(ssg_state + h_prev)
        else:
            ssg_state = s_t  # Initialize if not provided

        # Combine state-space and adaptive gating
        c_t = f_t * c_prev + i_t * c_hat_t * s_t * alpha_t * ssg_state

        # Deep residual connection
        residual = self.residual_layers(h_prev)
        c_t = c_t + residual

        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t, ssg_state


class BiAdaptiveLSTMWithWeakSignalSSGRes(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(BiAdaptiveLSTMWithWeakSignalSSGRes, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define forward and backward adaptive LSTM cells with state-space integration
        self.fwd_layers = nn.ModuleList(
            [AdaptiveLSTMCellWithWeakSignalSSGRes(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )
        self.bwd_layers = nn.ModuleList(
            [AdaptiveLSTMCellWithWeakSignalSSGRes(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Initialize forward and backward hidden states and state-space states
        h_fwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        c_fwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        ssg_fwd = [None for _ in range(self.num_layers)]

        h_bwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        c_bwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        ssg_bwd = [None for _ in range(self.num_layers)]

        # Forward pass
        output_fwd = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h_fwd[layer], c_fwd[layer], ssg_fwd[layer] = self.fwd_layers[layer](x_t, (h_fwd[layer], c_fwd[layer]), ssg_fwd[layer])
                x_t = h_fwd[layer]
            output_fwd.append(h_fwd[-1])

        # Backward pass
        output_bwd = []
        for t in range(seq_len - 1, -1, -1):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                h_bwd[layer], c_bwd[layer], ssg_bwd[layer] = self.bwd_layers[layer](x_t, (h_bwd[layer], c_bwd[layer]), ssg_bwd[layer])
                x_t = h_bwd[layer]
            output_bwd.append(h_bwd[-1])

        output_bwd.reverse()

        # Concatenate forward and backward outputs
        output = torch.cat([torch.stack(output_fwd, dim=1), torch.stack(output_bwd, dim=1)], dim=-1)

        return output