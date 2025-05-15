import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

from video_swintransformer.my_vst4 import my_video_swin
from video_swintransformer.presnet2 import PResNet2
from video_swintransformer.tsf import TemporalGateFusion
from video_swintransformer.tsf2 import SpatioTemporalFusion
from backbone.presnet import PResNet
from compsite_encoder import HybridEncoder
from detr_decoder import DETRTransformer


# 主模型
class IRST_DETR(nn.Module):

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.multi_scale = multi_scale
        self.multi_scale = None
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.resnet = PResNet2(depth=18, freeze_at=-1, freeze_norm=False, pretrained=False, num_stages=2)
        self.vst = my_video_swin(dims=[128, 256, 512], depths=[4, 6, 2])
        self.tsf1 = TemporalGateFusion(128,256)
        self.tsf2 = TemporalGateFusion(256, 256)
        self.tsf3 = SpatioTemporalFusion(512, 256)

    def forward(self, x, targets=None):
        x = x.permute(1, 0, 2, 3)

        # single frame
        x1 = x[2:3, :, :, :]
        x1 = self.backbone(x1)
        x1 = self.encoder(x1)

        # multi-frame
        x2 = self.resnet(x)
        x2 = self.vst(x2)

        # fusion
        g1 = self.tsf1(x2[0], x1[0])
        g2 = self.tsf2(x2[1], x1[1])
        g3 = self.tsf3(x2[2], x1[2])

        # decoder
        feats = [g1, g2, g3]
        x = self.decoder(feats, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self







