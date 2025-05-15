import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalGateFusion(nn.Module):
    def __init__(self, x1_channel, x2_channel):
        super().__init__()

        # 通道对齐
        self.time_conv = nn.Conv2d(x1_channel, x2_channel, 1)
        self.spatial_conv = nn.Conv2d(x2_channel, x2_channel, 1)

        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(
                in_channels=x2_channel * 2,
                out_channels=x2_channel * 2,
                kernel_size=3,
                padding=1,
                groups=x2_channel * 2,
                bias=False
            ),
            nn.BatchNorm2d(x2_channel * 2),
            nn.ReLU(inplace=True),
            # 逐点卷积
            nn.Conv2d(
                in_channels=x2_channel * 2,
                out_channels=x2_channel,
                kernel_size=1,
                bias=True
            ),
            nn.Sigmoid()
        )

        # 特征投影
        self.projection = nn.Conv2d(x2_channel, x2_channel, 1)

    def forward(self, temporal, spatial):
        # 通道对齐
        temporal = self.time_conv(temporal)  # [5, 256, 80, 80]
        spatial = self.spatial_conv(spatial)  # [1, 256, 80, 80]

        # 时间特征聚合
        temporal_pool = temporal.mean(dim=0, keepdim=True)  # [1, 256, 80, 80]

        # 门控融合
        gate_input = torch.cat([temporal_pool, spatial], dim=1)
        fusion_gate = self.gate(gate_input)  # [1, 256, 80, 80]

        # 加权融合
        fused = fusion_gate * temporal_pool + (1 - fusion_gate) * spatial

        return self.projection(fused)

if __name__ == "__main__":
    temporal = torch.randn(5, 128, 80, 80)
    spatial = torch.randn(1, 256, 80, 80)
    model = TemporalGateFusion(128, 256)
    output = model(temporal, spatial)
    print(output.shape)  # torch.Size([1, 256, 80, 80])