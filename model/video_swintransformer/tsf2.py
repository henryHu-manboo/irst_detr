import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepthwiseSeparableConv1x1(nn.Module):
    def __init__(self, spatial_channels):
        super().__init__()
        # 深度卷积阶段
        self.depthwise = nn.Conv2d(
            in_channels=spatial_channels,
            out_channels=spatial_channels,
            kernel_size=1,
            groups=spatial_channels,
            bias=False
        )
        # 逐点卷积阶段
        self.pointwise = nn.Conv2d(
            in_channels=spatial_channels,
            out_channels=spatial_channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SpatioTemporalFusion(nn.Module):
    def __init__(self, temporal_channels=512, spatial_channels=256, num_heads=8, dropout=0.1):
        super().__init__()

        self.temporal_proj = nn.Conv2d(temporal_channels, spatial_channels, kernel_size=1)
        self.spatial_proj = nn.Conv2d(spatial_channels, spatial_channels, kernel_size=1)
        self.pos_encoder = PositionalEncoding(spatial_channels, dropout)
        self.spatial_proj2 = DepthwiseSeparableConv1x1(spatial_channels)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=spatial_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )

        self.ffn = nn.Sequential(
            nn.Linear(spatial_channels, spatial_channels * 4),
            nn.ReLU(),
            nn.Linear(spatial_channels * 4, spatial_channels),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(spatial_channels)
        self.norm2 = nn.LayerNorm(spatial_channels)

        self.out_proj = nn.Conv2d(spatial_channels, spatial_channels, kernel_size=1)

    def forward(self, temporal, spatial):
        """
        temporal: [T, C_t, H, W] (T=5, C_t=512)
        spatial: [1, C_s, H, W] (C_s=256)
        """
        T, C_t, H, W = temporal.shape
        _, C_s, H, W = spatial.shape

        temporal = self.temporal_proj(temporal)
        temporal = temporal.permute(2, 3, 0, 1)  # [H, W, T, C_s]
        temporal = temporal.reshape(H * W, T, C_s)  # [H*W, T, C_s]
        temporal = self.pos_encoder(temporal)

        spatial = self.spatial_proj(spatial)
        spatial = self.spatial_proj2(spatial)
        spatial = spatial.squeeze(0)  # [C_s, H, W]
        spatial = spatial.permute(1, 2, 0)  # [H, W, C_s]
        spatial = spatial.reshape(H * W, 1, C_s)  # [H*W, 1, C_s]

        attn_output, _ = self.cross_attn(
            query=spatial,  # [H*W, 1, C_s]
            key=temporal,  # [H*W, T, C_s]
            value=temporal  # [H*W, T, C_s]
        )

        attn_output = self.norm1(attn_output + spatial)

        ffn_output = self.ffn(attn_output)
        ffn_output = self.norm2(ffn_output + attn_output)

        output = ffn_output.view(H, W, 1, C_s)  # [H, W, 1, C_s]
        output = output.permute(3, 2, 0, 1)  # [C_s, 1, H, W]
        output = self.out_proj(output)

        return output.unsqueeze(0)


class PositionalEncoding(nn.Module):
    """时间位置编码"""

    def __init__(self, d_model, dropout=0.1, max_len=5):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [N, T, C] (H*W, T, C_s)
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    temporal = torch.randn(5, 512, 20, 20)
    spatial = torch.randn(1, 256, 20, 20)

    model = SpatioTemporalFusion()
    output = model(temporal, spatial)
    print(output.shape)