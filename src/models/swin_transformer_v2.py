# 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import collections.abc
from itertools import repeat

import numpy as np
from mindspore import Parameter
from mindspore import dtype as mstype
from mindspore import nn, Tensor
from mindspore import ops as P
from mindspore.common import initializer as weight_init
from mindspore.numpy import roll

from src.models.layers.drop_path import DropPath1D
from src.models.layers.identity import Identity


class Roll(nn.Cell):
    """Roll Cell"""

    def __init__(self, shift, axis=(1, 2)):
        super(Roll, self).__init__()
        self.shift_size = to_2tuple(shift)
        self.shift_axis = axis

    def construct(self, x):
        x = roll(x, self.shift_size, self.shift_axis)
        return x


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = windows.shape[0] // (H * W / window_size / window_size)
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x


class WindowAttention(nn.Cell):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = Parameter(Tensor(np.log(10 * np.ones((num_heads, 1, 1))), dtype=mstype.float32),
                                     requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.SequentialCell([nn.Dense(in_channels=2, out_channels=512, has_bias=True),
                                          nn.ReLU(),
                                          nn.Dense(in_channels=512, out_channels=num_heads, has_bias=False)])

        # get relative_coords_table
        relative_coords_h = np.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=np.float32)
        relative_coords_w = np.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=np.float32)
        zeros = np.zeros([int(2 * window_size[0] - 1), int(2 * window_size[1] - 1)], dtype=np.float32)
        relative_coords_h = relative_coords_h.reshape(-1, 1) + zeros
        relative_coords_w = relative_coords_w.reshape(1, -1) + zeros
        relative_coords_table = np.stack([relative_coords_h, relative_coords_w], axis=-1)[np.newaxis, :]
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = np.sign(relative_coords_table) * np.log2(
            np.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.relative_coords_table = Parameter(Tensor(relative_coords_table, mstype.float32), requires_grad=False)

        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(self.window_size[0]).reshape(self.window_size[0], 1)
        coords_h = coords_h.repeat(self.window_size[0], 1).reshape(1, -1)
        coords_w = np.arange(self.window_size[1]).reshape(1, self.window_size[1])
        coords_w = coords_w.repeat(self.window_size[1], 0).reshape(1, -1)
        coords_flatten = np.concatenate([coords_h, coords_w], axis=0)  # 2, Wh, Ww
        relative_coords = coords_flatten[:, :, np.newaxis] - coords_flatten[:, np.newaxis, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = Parameter(Tensor(relative_coords.sum(-1).reshape(-1), dtype=mstype.int32),
                                                 requires_grad=False)  # Wh*Ww, Wh*Ww
        # self.one_hot = nn.OneHot(axis=-1, depth=(2 * window_size[0] - 1) * (2 * window_size[1] - 1),
        #                          dtype=mstype.float32)
        # print((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.relative_position_index.shape, self.relative_position_index.max(), self.relative_position_index.min())
        # exit()
        # self.index = Parameter(self.one_hot(self.relative_position_index), requires_grad=False)
        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=False)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)
        self.max_value = P.Log()(Tensor(1.0 / 0.01, dtype=mstype.float32))
        self.min_value = Tensor(-10000000., dtype=mstype.float32)
        self.batch_matmul = P.BatchMatMul()
        self.matmul = P.MatMul(transpose_b=True)

    def construct(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        q = self.q(x).reshape(B_, N, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B_, N, self.num_heads, -1).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B_, N, self.num_heads, -1).transpose(0, 2, 1, 3)

        # cosine attention
        attn = P.BatchMatMul()(P.L2Normalize(-1, 1e-12)(q), P.L2Normalize(-1, 1e-12)(k).transpose(0, 1, 3, 2))
        logit_scale = P.clip_by_value(self.logit_scale, clip_value_max=self.max_value, clip_value_min=self.min_value)
        logit_scale = P.Exp()(logit_scale)
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).reshape(-1, self.num_heads)
        relative_position_bias = P.Gather()(relative_position_bias_table,
                                            self.relative_position_index.reshape(-1), 0)
        relative_position_bias = P.Reshape()(relative_position_bias, (self.window_size[0] * self.window_size[1],
                                                                      self.window_size[0] * self.window_size[1], -1))
        relative_position_bias = P.Transpose()(relative_position_bias, (2, 0, 1,))  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * P.Sigmoid()(relative_position_bias)
        attn = attn + P.ExpandDims()(relative_position_bias, 0)

        if mask is not None:
            nW = mask.shape[0]
            attn = P.Reshape()(attn, (B_ // nW, nW, self.num_heads, N, N,)) + P.ExpandDims()(P.ExpandDims()(mask, 1), 0)
            attn = P.Reshape()(attn, (-1, self.num_heads, N, N,))
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = self.batch_matmul(attn, v).transpose(0, 2, 1, 3).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Cell):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer((dim,))
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath1D(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = np.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)
            attn_mask = mask_windows[:, np.newaxis] - mask_windows[:, :, np.newaxis]
            attn_mask = np.where(attn_mask == 0, 0., -100.)
        else:
            attn_mask = None
        self.roll_pos = Roll(shift=(shift_size, shift_size), axis=(1, 2))
        self.roll_neg = Roll(shift=(-shift_size, -shift_size), axis=(1, 2))
        if attn_mask is not None:
            self.attn_mask = Parameter(Tensor(attn_mask, mstype.float32), requires_grad=False)
        else:
            self.attn_mask = None

    def construct(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = P.Reshape()(x, (B, H, W, C,))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = self.roll_neg(x)
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        # reverse cyclic shift
        if self.shift_size > 0:
            x = self.roll_pos(shifted_x)
        else:
            x = shifted_x
        x = P.Reshape()(x, (B, H * W, C))
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class PatchMerging(nn.Cell):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Dense(in_channels=4 * dim, out_channels=2 * dim, has_bias=False)
        self.norm = norm_layer((2 * dim,))

    def construct(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        x = P.Reshape()(x, (B, H, W, C,))
        x = P.Reshape()(x, (B, H // 2, 2, W // 2, 2, self.dim))
        x = P.Transpose()(x, (0, 1, 3, 4, 2, 5))
        x = P.Reshape()(x, (B, -1, 4 * C))
        x = self.reduction(x)
        x = self.norm(x)

        return x


class BasicLayer(nn.Cell):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.CellList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def construct(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def init_respostnorm(self):
        for blk in self.blocks:
            blk.norm1.gamma.set_data(
                weight_init.initializer(weight_init.Zero(), blk.norm1.gamma.shape, blk.norm1.gamma.dtype))
            blk.norm2.gamma.set_data(
                weight_init.initializer(weight_init.Zero(), blk.norm2.gamma.shape, blk.norm2.gamma.dtype))


class PatchEmbed(nn.Cell):
    r""" Image to Patch Embedding

    Args:
        image_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, image_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size,
                              pad_mode='pad', has_bias=True)
        if norm_layer is not None:
            self.norm = norm_layer((embed_dim,))
        else:
            self.norm = None

    def construct(self, x):

        x = self.proj(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(0, 2, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerV2(nn.Cell):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        image_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, image_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            image_size=image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = Parameter(np.zeros([1, num_patches, embed_dim]))
            self.absolute_pos_embed.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02)))
        self.pos_drop = nn.Dropout(keep_prob=1.0 - drop_rate)

        # stochastic depth
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.CellList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer((self.num_features,))
        self.head = nn.Dense(in_channels=self.num_features,
                             out_channels=num_classes) if num_classes > 0 else Identity()

        self.init_weights()
        for bly in self.layers:
            bly.init_respostnorm()

    def init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        x = P.ReduceMean()(x, 1)
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def swinv2_base_patch4_window12to16_192to256_22kto1k_ft(image_size, drop_path_rate, num_classes):
    model = SwinTransformerV2(image_size=image_size,
                              patch_size=4,
                              in_chans=3,
                              num_classes=num_classes,
                              embed_dim=128,
                              depths=[2, 2, 18, 2],
                              num_heads=[4, 8, 16, 32],
                              window_size=16,
                              mlp_ratio=4,
                              qkv_bias=True,
                              drop_rate=0.0,
                              drop_path_rate=drop_path_rate,
                              ape=False,
                              patch_norm=True,
                              pretrained_window_sizes=[12, 12, 12, 6])
    return model


if __name__ == "__main__":
    from mindspore import context

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    data = Tensor(np.ones([3, 3, 256, 256]), dtype=mstype.float32)
    model = swinv2_base_patch4_window12to16_192to256_22kto1k_ft(image_size=256, drop_path_rate=0.1, num_classes=1000)
    # print(model(data).shape)
    # model.update_parameters_name(prefix="model")
    # params_num = sum([np.prod(data.shape) for _, data in model.parameters_and_names()])
    # num = sum([1 for _, data in model.parameters_and_names()])
    # print(params_num)
    # print(num, params_num)
    params_num = 0
    for name, data in model.parameters_and_names():
        if "relative_position_index" not in name and "attn_mask" not in name:
            params_num += np.prod(data.shape)
    print(params_num)
