import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .video_swintransformer2 import SwinTransformerBlock3D


class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, num_frames=5, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.num_frames = num_frames

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        b, c, h, w = x.shape
        b_new = b // self.num_frames
        x = x.reshape(b_new, self.num_frames, h, w, c)
        B, D, H, W, C = x.shape
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)

        B, D, H, W, C = x.shape
        x = x.reshape(B*D, C, H, W)

        return x


class my_video_swin(nn.Module):
    #  对于r18   输入应该是 128, 256, 512  存疑
    #  如果是encoder输入，应该先过一个卷积，减少通道维度
    def __init__(self, dims=[128, 256, 512], depths=[2, 4, 2], drop_path_rate=0.0):
        super().__init__()
        self.dpr = drop_path_rate
        if self.dpr == 0.0:
            my_dpr = [dd.item() for dd in torch.linspace(0.0, self.dpr, sum(depths))]
        else:
            my_dpr = [dd.item() for dd in torch.linspace(0.05, self.dpr, sum(depths))]

        
        
        self.block1 = SwinTransformerBlock3D(c1=dims[0], num_frames=5, num_layers=depths[0],
                                             num_heads_=32, drop_path_rate=my_dpr[:depths[0]])
        self.block2 = SwinTransformerBlock3D(c1=dims[1], num_frames=5, num_layers=depths[1],
                                             num_heads_=32, drop_path_rate=my_dpr[depths[0]:depths[0]+depths[1]])
        self.block3 = SwinTransformerBlock3D(c1=dims[2], num_frames=5, num_layers=depths[2],
                                             num_heads_=32, drop_path_rate=my_dpr[depths[0]+depths[1]:depths[0]+depths[1]+depths[2]])
        self.downsample1 = PatchMerging(dim=dims[0], num_frames=5)
        self.downsample2 = PatchMerging(dim=dims[1], num_frames=5)

    def forward_1(self, x):

       
        x1 = self.block1(x)

        x2 = self.downsample1(x1)
        x2 = self.block2(x2)

        x3 = self.downsample2(x2)
        x3 = self.block3(x3)

        return [x1, x2, x3]

    def forward(self, x):
        x = self.forward_1(x)
        return x


if __name__ == "__main__":
    model = my_video_swin(dims=[128, 256, 512], depths=[2, 4, 2], drop_path_rate=0.0)
    x = torch.randn([5, 128, 80, 80])
    x = model(x)
    print(x[0].shape)
    print(x[1].shape)
    print(x[2].shape)