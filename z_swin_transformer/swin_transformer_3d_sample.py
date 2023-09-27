import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_

# from mmcv.runner import load_checkpoint
# from mmaction.utils import get_root_logger
# from ..builder import BACKBONES
# from mmaction.models.builder import BACKBONES

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features #如果没有指定out_features，则默认为in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        '''
        input_shape: B D H W C
        out_shape: B D H W C
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    #形状变换：(B, D, H, W, C)->(B, nwd, wd, nwh, wh, nww, ww, C)
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    #形状变换：(B, nwd, wd, nwh, wh, nww, ww, C)->(B, nwd, nwh, nww, wd, wh, ww, C)->(B*nwd*nwh*nww, wd*wh*ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    #形状变换：(B*nw, wd, wh, ww, C)->(B, D//wd, H//wh, W//hw, wd, wh, ww, C)
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    #形状变换：(B, D//wd, H//wh, W//hw, wd, wh, ww, C)->(B, D//wd, wd, H//wh, wh, W//hw, ww, C)->(B, D, H, W, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

#判断指定窗口尺寸是否合理（小于等于输入数据尺寸），如何不合理（大于输出尺寸），令window_size=x.size,shift_size=0。
def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads #降维
        self.scale = qk_scale or head_dim ** -0.5 #是否除以根号dk

        # define a parameter table of relative position bias
        #相对位置偏差参数（head个列向量）
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        #求相对偏差编码

        #获取窗口内每个patch的绝对位置
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij')) #3, Wd, Wh, Ww ，得到三个维度的独立坐标张量，并打包成一个更大的张量
        coords_flatten = torch.flatten(coords, 1) #3, Wd*Wh*Ww，将每个独立坐标张量展平成一维。最终coords_flatten的维度是2维，大小为(3, Wd*Wh*Ww)
        #计算相对位置之差
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] #3, Wd*Wh*Ww, Wd*Wh*Ww；先进行维度扩张变换，再进行相减；(3, Wd*Wh*Ww, 1) - (3, 1, Wd*Wh*Ww) = (3, Wd*Wh*Ww, Wd*Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() #Wd*Wh*Ww, Wd*Wh*Ww, 3；维度调换，(3, Wd*Wh*Ww, Wd*Wh*Ww) -> (Wd*Wh*Ww, Wd*Wh*Ww, 3)，每个点的(x,y,z)坐标放在一起
        #将位置之差的范围转为成[0, window_size-1]
        relative_coords[:, :, 0] += self.window_size[0] - 1 #shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        #计算偏差编码（把三维张量展平成一维向量，计算每个位置的编号）
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww；对最后一维求和，即每个点的三个坐标值求和。
        self.register_buffer("relative_position_index", relative_position_index) #将位置编码表注册到模型缓冲区（不进行梯度更新）

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #线性计算qkv
        self.attn_drop = nn.Dropout(attn_drop) #注意力权重结果随机失活
        self.proj = nn.Linear(dim, dim) #线性计算最终结果
        self.proj_drop = nn.Dropout(proj_drop) #最终结果随机失活

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        input_shape: B*nw Wd*Wh*Ww C
        out_shape:  B*nw Wd*Wh*Ww C

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        #形状变换：(B*n_w, N, C)->(B*n_w, N, 3C)->(B*n_w, N, 3, H, C//H)->(3, B*n_w, H, N, C//H)。H是head的数目，B*n_w=B_
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C//H #得到多头的qkv

        #计算注意力
        q = q * self.scale #若scale不为None，则令q除以根号(C//H)
        attn = q @ k.transpose(-2, -1) #计算注意力q×k.T（矩阵乘法）。形状变换：(B*n_w, H, N, C//H)->(B*n_w, H, N, N)

        #根据相对位置偏差索引表查找对应的相对位置偏差值，最后将结果形状转换成3维
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1) #Wd*Wh*Ww,Wd*Wh*Ww,nH(nH=1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() #nH, Wd*Wh*Ww, Wd*Wh*Ww。
        #加上局部相对位置偏差
        attn = attn + relative_position_bias.unsqueeze(0) #B_(1), nH(1), N, N。在0维之前增加一个维度，为了和注意力结果维度保持一致。

        if mask is not None:
            nW = mask.shape[0]
            #带有掩码的注意力机制。形状变换：attn-(B*n_w, H, N, N)->(B*n_w // nW, nW, H, N, N)；mask-(nW, N, N)->(1, nW, 1, N, N)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N) #重新变换成原来的形状(B*n_w, H, N, N)
            attn = self.softmax(attn) #归一化
        else:
            attn = self.softmax(attn) #不带掩码的注意力机制

        attn = self.attn_drop(attn) #注意力失活

        #计算最终的输出值。形状变换：(B*n_w, H, N, N).(B*n_w, H, N, C//H)->(B*n_w, H, N, C//H)->(B*n_w, N, H, C//H)->(B*n_w, N, C)。
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C) #权重与v相乘结果，并把每个head结果按列拼在一起
        x = self.proj(x) #线性变换，输出结果
        x = self.proj_drop(x) #输出结果随机失活
        return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size #窗口移动单元，单数块不移动
        self.mlp_ratio = mlp_ratio #MLP隐层数据维度扩张倍数
        self.use_checkpoint = use_checkpoint #！？

        #窗口移动大小范围必须在0-window_size之间
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        #层归一化
        self.norm1 = norm_layer(dim)

        #注意力计算
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() #随机丢失样本（某些样本的所有特征置0）
        self.norm2 = norm_layer(dim) #层归一化
        mlp_hidden_dim = int(dim * mlp_ratio) #mlp模块的隐层维度（通道数）
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop) #mlp

    def forward_part1(self, x, mask_matrix):
        '''
        input_shape: B D H W C
        out_shape: B D H W C
        '''
        B, D, H, W, C = x.shape
        #处理x的某维度小于指定window对应维度的情况（win_size=x_size, shift_size=0）
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        #使用0对x进行填充，使得x的尺寸能够被window_size整除
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1)) #填充操作
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size): #需要移位操作
            #将x的Dp, Hp, Wp三个维度沿指定方向shifts的反方向移动（将窗口沿shifts方向移动）
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else: #不需要移位操作
            shifted_x = x
            attn_mask = None
        # partition windows
        #窗口划分。数据尺寸变换：(B, Dp, Hp, Wp, C)->(B*nW, Wd*Wh*Ww, C)
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        #注意力机制（如果移位，需要加掩码）。形状保持不变：(B*nW, Wd*Wh*Ww, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        #还原窗口，将数据转换成未划分窗口时的形状。(B*nW, Wd*Wh*Ww, C)->(B*nW, Wd, Wh, Ww, C)->(B, Dp, Hp, Wp, C)
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        #反向循环移位进行数据还原
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        #把一开始填充的0元素删除
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        '''
        input_shape: B D H W C
        out_shape: B D H W C
        '''
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.
        input_shape: B D H W C
        out_shape: B D H W C

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        #MSA
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix) #不保存forward_part1中的激活值
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        #MLP
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x) #不保存forward_part2中的激活值
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2)) #对W维和H维进行填充，以便进行x2下采样。F.pad()是从最低维开始填充。
        #对H和W维，均每隔2个单元选一次，分别得到4个尺寸相同的张量，每个张量尺寸的H和W维度均为原来的1/2。
        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        #四个张量的最后一维（C维）进行拼接，得到一个新张量。尺寸变换：(B, D, H, W, C)->(B, D, H/2, W/2, 4*C)
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        #降维，形状变换：(B, D, H/2, W/2, 4*C)->(B, D, H/2, W/2, 2*C)
        x = self.reduction(x)

        return x


# cache each stage results
@lru_cache() #缓存函数的返回值
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1。#创建一个5维全0张量,D,H,W为单个数据的尺寸
    cnt = 0 #每一块的编号（移位后的块）
    #给每一块（窗口，大小不同）进行编号，d、h、w为每一维上的切片对象
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt #同一块（窗口），赋值相同数字
                cnt += 1
    #划分窗口，形状变换：(1, Dp, Hp, Wp, 1)->(nw, wd*wh*ww, 1)
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    #把最后一维压缩，形状变换：(nw, wd*wh*ww, 1)->(nw, wd*wh*ww*1)
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    #按2个不同维度进行拉伸，然后相减。形状变换：(nw, wd*wh*ww*1)->(nw, 1, wd*wh*ww) - (nw, wd*wh*ww*1)->(nw, wd*wh*ww, 1) -> (nw, wd*wh*ww, wd*wh*ww)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) #nw, wd*wh*ww, wd*wh*ww
    #掩码填充，不等于0（不在同一窗口）填充-100.0，等于0（在同一窗口）填充0.0。
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim, #输入数据维度（通道数）
                 depth, #当前层的深度（基本块的个数）
                 num_heads, #多头注意力的头数目（h的值）
                 window_size=(1, 7, 7), #窗口大小
                 mlp_ratio=4., #mlp隐层的数据维度扩张倍数
                 qkv_bias=False, #是否为qkv添加偏差
                 qk_scale=None, #qk的放缩倍数
                 drop=0., #每一个模块输出结果的失活率
                 attn_drop=0., #注意力权重结果失活率
                 drop_path=0., #每层的随机衰减率
                 norm_layer=nn.LayerNorm, #层归一化
                 downsample=None, #下采样
                 use_checkpoint=False #是否保存激活数据
                 ):
        super().__init__()
        self.window_size = window_size #窗口大小
        self.shift_size = tuple(i // 2 for i in window_size) #每个维度的移动单元为对应窗口维度的1/2
        self.depth = depth #当前层的深度（基本块数目）
        self.use_checkpoint = use_checkpoint #是否保存激活数据

        # build blocks
        #构建每层（stage）的各个swin块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim, #输入数据维度（通道数）
                num_heads=num_heads, #head数（多头注意力）
                window_size=window_size, #窗口尺寸
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size, #奇数块移位，偶数块不移位（从0块开始）
                mlp_ratio=mlp_ratio, #mlp的隐层扩展率
                qkv_bias=qkv_bias, #计算qkv时是否添加偏置项
                qk_scale=qk_scale, #qk放缩率
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, #每层的全样本失活率
                norm_layer=norm_layer, #归一化函数
                use_checkpoint=use_checkpoint, #是否保存激活数据
            )
            for i in range(depth)])

        #下采样函数（patch mergeing）
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.
        input_shape: B C D H W
        out_shape: B 2C D H//2 W//2

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        #calculate attention mask for SW-MSA
        #掩码设计
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        #变换x的形状，等价于x.permute(0, 2, 3, 4, 1)。形状变换：(B, C, D, H, W)->(B, D, H, W, C)
        x = rearrange(x, 'b c d h w -> b d h w c')
        #计算单个数据（3D）的掩码张量的尺寸
        Dp = int(np.ceil(D / window_size[0])) * window_size[0] #np.ceil表示向上取整
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        #计算掩码。attn_mask尺寸：(nw, wd*wh*ww, wd*wh*ww)，nw为单个数据中的窗口数目，wd、wh、ww分别为窗口各个维度大小
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        #基本块计算。形状不变：(B, D, H, W, C)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        #形状变换。形状未改变：(B, D, H, W, C)
        x = x.view(B, D, H, W, -1)
        #下采样（patch mergeing）。形状变换(B, D, H, W, C)->(B, D, H//2, W//2, 2C)
        if self.downsample is not None:
            x = self.downsample(x)
        #形状变换：(B, D, H//2, W//2, 2C)->(B, 2C, D, H//2, W//2)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans #输入通道
        self.embed_dim = embed_dim #输出通道

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size) #使用卷积进行不重叠patch划分
        if norm_layer is not None: #使用指定标准化函数进行标准化
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        '''
        input_shape: B 3 D H W
        out_shape: B C D//patch_size[0] H//patch_size[1] W//patch_size[2]
        '''
        # padding

        # 获取输入数据x的D,H,W（帧数，高，宽）
        _, _, D, H, W = x.size()
        #判断x的每个维度是否能够被patch_size整除，如果不能进行相应的填充
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2])) #在x的最后一维（W维）的第一个方向上不填充，第二个方向上填充self.patch_size[2] - W % self.patch_size[2]个单元
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        #划分patch，3D卷积。形状变换：(B, 3, D, H, W)->(B, C, D//patch_size[0], H//patch_size[1], W//patch_size[2])，C=embed_dim。
        x = self.proj(x)  # B C Dd Hh Ww

        #是否进行标准化（归一化）
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            # 将x的第二维（D维）及之后维度展平成向量，并进行1,2维度转置。形状变换：(B,C,Dd,Hh,Ww)->(B,C,Dd×Hh×Ww)->(B,Dd×Hh×Ww,C)
            #这样做的目的是为了与MSA计算统一
            x = x.flatten(2).transpose(1, 2)
            #归一化（LN）
            x = self.norm(x)
            #将x还原成初始形状（便于进行后面的卷积或线性操作）。形状变换：(B,Dd×Hh×Ww,C)->(B,C,Dd×Hh×Ww)->(B,C,Dd,Hh,Ww)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class SwinTransformer3D(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (tuple[int]): Window size. Default: (2, 7, 7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None, #预训练权重（3D）
                 pretrained2d=True, #预训练权重（2D）
                 patch_size=(4, 4, 4), #每个patch包含的像素个数
                 in_chans=3, #数据的输入通道
                 embed_dim=96, #模型的输入维度（通道数）
                 depths=(2, 2, 6, 2), #每个stage的层数（block数）
                 num_heads=(3, 6, 12, 24), #每个stage的head数目（多头注意力）
                 window_size=(2, 7, 7), #窗口尺寸（每个标准窗口包含的patch数目）
                 mlp_ratio=4., #MLP模块的隐层扩展率
                 qkv_bias=True, #计算qkv时是否加偏置项
                 qk_scale=None, #qk计算结果的放缩倍数
                 drop_rate=0., #模型每一个模块的输出结果的失活率
                 attn_drop_rate=0., #注意力机制的失活率
                 drop_path_rate=0.2, #各层的随机衰减率（整个样本失活）
                 norm_layer=nn.LayerNorm, #层归一化
                 patch_norm=False, #是否在embed之后进行归一化
                 frozen_stages=-1, #停止某些层参数更新（固定参数）
                 use_checkpoint=False #是否存储激活数据
                 ):
        super().__init__()

        self.pretrained = pretrained  #3D预训练
        self.pretrained2d = pretrained2d  #2D预训练（用swin transformer在imagenet上预训练）
        self.num_layers = len(depths)  #模型层数（多少个stage）
        self.embed_dim = embed_dim  #模型输入的维度（通道数），C的值
        self.patch_norm = patch_norm  #是否在patch embedding后进行标准化
        self.frozen_stages = frozen_stages  #停止某些参数更新
        self.window_size = window_size  #局部窗口大小（每个窗口中的patch数目）
        self.patch_size = patch_size  #patch的大小（每个patch中的像素个数）

        #split image into non-overlapping patches
        #划分patch，并将维度（通道数）变为设定的数值embed_dim（使用的是3D卷积）
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        #随机失活
        self.pos_drop = nn.Dropout(p=drop_rate)

        #stochastic depth
        #随机深度衰减，每一层随机失活几个样本
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  #stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        #依次构建每一层（每个stage）
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer), #当前stage的输出维度（通道数）
                depth=depths[i_layer], #当前stage的层数（block的数目）
                num_heads=num_heads[i_layer], #当前stage的head数（多头注意力）
                window_size=window_size, #窗口尺寸
                mlp_ratio=mlp_ratio, #mlp隐层的扩展率
                qkv_bias=qkv_bias, #计算q、k、v时，是否加偏置
                qk_scale=qk_scale, #计算q.(k.T)时，是否进行数值的放缩
                drop=drop_rate, #每个模块输出结果的失活率
                attn_drop=attn_drop_rate, #注意力结果的失活率
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], #每一层（block）的随机衰减率
                norm_layer=norm_layer, #归一化函数
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None, #patch合并，下采样
                use_checkpoint=use_checkpoint #不保存激活数据
            )
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) #模型最后输出的数据维度（通道数）

        # add a norm layer for each output
        #层归一化
        self.norm = norm_layer(self.num_features)

        #固定参数，停止某些层参数更新
        self._freeze_stages()

    #固定参数（不更新）
    def _freeze_stages(self):
        if self.frozen_stages >= 0: #停止patch_embed中的参数更新
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1: #停止pos_drop和前frozen_stages层的参数更新
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """Forward function.
        input_shape: B 3 D H W
        out_shape: B C*2^(N-1) Dd Hh//2^(N-1) Ww//2^(N-1)). N为模型层数（stage数目）
        """

        #划分patch并进行维度（通道）变换。形状变换：(B, 3, D, H, W)->(B, C, Dd:D//patch_size[0], Hh:H//patch_size[1], Ww:W//patch_size[2])
        x = self.patch_embed(x)

        #随机失活
        x = self.pos_drop(x)

        #计算每一层。形状变换：(B, C, Dd, Hh, Ww)->(B, 2C, Dd, Hh//2, Ww//2)->......->(B, C':C*2^(N-1), Dd, Hd':Hh//2^(N-1), Ww':Ww//2^(N-1))
        for layer in self.layers:
            x = layer(x.contiguous())

        #改变形状，进行层归一化。形状变换：(B, C', Dd, Hd', Ww')->(B, Dd, Hd', Ww', C')
        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        #形状还原：(B, Dd, Hd', Ww', C')->(B, C', Dd, Hd', Ww')
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x


class ClassifyHead(nn.Module):
    '''分类头
    Args:
        num_class (int): 类别数目
        in_features (int): 输入特征数目
    '''

    def __init__(self, num_class=101, in_features=768):
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=num_class)

    def forward(self, x):
        return self.fc(x)


class ClassifySwinTransformer3D(nn.Module):
    '''分类任务的SwinTransformer3D
    Args:
        num_class (int): 类别数目
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int] | list[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int] | list[int]): Number of attention head of each stage.
        window_size (tuple[int]): Window size. Default: (2, 7, 7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    '''

    def __init__(self,
                 num_class=101,
                 pretrained=None,  # 预训练权重（3D）
                 pretrained2d=True,  # 预训练权重（2D）
                 patch_size=(4, 4, 4),  # 每个patch包含的像素个数
                 in_chans=3,  # 数据的输入通道
                 embed_dim=96,  # 模型的输入维度（通道数）
                 depths=(2, 2, 6, 2),  # 每个stage的层数（block数）
                 num_heads=(3, 6, 12, 24),  # 每个stage的head数目（多头注意力）
                 window_size=(2, 7, 7),  # 窗口尺寸（每个标准窗口包含的patch数目）
                 mlp_ratio=4.,  # MLP模块的隐层扩展率
                 qkv_bias=True,  # 计算qkv时是否加偏置项
                 qk_scale=None,  # qk计算结果的放缩倍数
                 drop_rate=0.,  # 模型每一个模块的输出结果的失活率
                 attn_drop_rate=0.,  # 注意力机制的失活率
                 drop_path_rate=0.2,  # 各层的随机衰减率（整个样本失活）
                 norm_layer=nn.LayerNorm,  # 层归一化
                 patch_norm=False,  # 是否在embed之后进行归一化
                 frozen_stages=-1,  # 停止某些层参数更新（固定参数）
                 use_checkpoint=False  # 是否存储激活数据
                 ):
        super().__init__()
        #实例化swin3D对象
        self.swin = SwinTransformer3D(pretrained=pretrained,  # 预训练权重（3D）
                                      pretrained2d=pretrained2d,  # 预训练权重（2D）
                                      patch_size=patch_size,  # 每个patch包含的像素个数
                                      in_chans=in_chans,  # 数据的输入通道
                                      embed_dim=embed_dim,  # 模型的输入维度（通道数）
                                      depths=depths,  # 每个stage的层数（block数）
                                      num_heads=num_heads,  # 每个stage的head数目（多头注意力）
                                      window_size=window_size,  # 窗口尺寸（每个标准窗口包含的patch数目）
                                      mlp_ratio=mlp_ratio,  # MLP模块的隐层扩展率
                                      qkv_bias=qkv_bias,  # 计算qkv时是否加偏置项
                                      qk_scale=qk_scale,  # qk计算结果的放缩倍数
                                      drop_rate=drop_rate,  # 模型每一个模块的输出结果的失活率
                                      attn_drop_rate=attn_drop_rate,  # 注意力机制的失活率
                                      drop_path_rate=drop_path_rate,  # 各层的随机衰减率（整个样本失活）
                                      norm_layer=norm_layer,  # 层归一化
                                      patch_norm=patch_norm,  # 是否在embed之后进行归一化
                                      frozen_stages=frozen_stages,  # 停止某些层参数更新（固定参数）
                                      use_checkpoint=use_checkpoint  # 是否存储激活数据
                                      )

        #对swin的输出结果进行全局平均池化。形状变化：(B, C, D, H, W)->(B, C, 1, 1, 1)
        self.avgepool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        #实例化分类头对象
        self.classify_head = ClassifyHead(num_class=num_class, in_features=self.swin.num_features)
        #初始化权重
        self.apply(self.init_weights)

    def forward(self, x):
        '''
        input_shape: B C D H W
        out_shape: B num_class
        '''
        x = self.swin(x)
        x = self.avgepool(x)
        x = x.view(x.shape[0], -1)
        x = self.classify_head(x)

        return x

    # 初始化权重
    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # 截断的正态分布初始化（均值为0，标准差为0.02），并且将值控制在一定范围。
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 常量初始化
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self.swin._freeze_stages() #停止某些层的更新



def swin_t(num_class=101, patch_size=(2, 4, 4), window_size=(2, 7, 7), drop_rate=0.5):
    return ClassifySwinTransformer3D(num_class=num_class, patch_size=patch_size, window_size=window_size,
                                     embed_dim=96, depths=[2, 2, 6, 2], num_heads=(3, 6, 12, 24), drop_rate=drop_rate)


def swin_s(num_class=101, patch_size=(2, 4, 4), window_size=(2, 7, 7), drop_rate=0.5):
    return ClassifySwinTransformer3D(num_class=num_class, patch_size=patch_size, window_size=window_size,
                                     embed_dim=96, depths=[2, 2, 18, 2], num_heads=(3, 6, 12, 24), drop_rate=drop_rate)


def swin_b(num_class=101, patch_size=(2, 4, 4), window_size=(2, 7, 7), drop_rate=0.5):
    return ClassifySwinTransformer3D(num_class=num_class, patch_size=patch_size, window_size=window_size,
                                     embed_dim=128, depths=[2, 2, 18, 2], num_heads=(4, 8, 16, 32), drop_rate=drop_rate)


def swin_l(num_class=101, patch_size=(2, 4, 4), window_size=(2, 7, 7), drop_rate=0.5):
    return ClassifySwinTransformer3D(num_class=num_class, patch_size=patch_size, window_size=window_size,
                                     embed_dim=192, depths=[2, 2, 18, 2], num_heads=(6, 12, 24, 48), drop_rate=drop_rate)
