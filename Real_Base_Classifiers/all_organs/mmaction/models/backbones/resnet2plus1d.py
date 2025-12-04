# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import BACKBONES
from .resnet3d import ResNet3d
import torch.nn as nn
from mmcv.cnn import kaiming_init, constant_init
from mmcv.utils import _BatchNorm

def init_weights(m):
    if isinstance(m, nn.Conv3d):
        kaiming_init(m)
    elif isinstance(m, _BatchNorm):
        constant_init(m, 1)

class last_Conv2plus1d(nn.Module):
    def __init__(self):
        super(last_Conv2plus1d, self).__init__()
        M = int((3 * 3 * 3 * 512 * 512) / (3 * 3 * 512 + 3 * 512))
        self.conv_s = nn.Conv3d(512, M, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.bn_s = nn.BatchNorm3d(M, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv_t = nn.Conv3d(M, 512, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=False)
    
    def forward(self, x):
        x = self.conv_t(self.relu(self.bn_s(self.conv_s(x))))
        return x
    
@BACKBONES.register_module()
class ResNet2Plus1d(ResNet3d):
    """ResNet (2+1)d backbone.

    This model is proposed in `A Closer Look at Spatiotemporal Convolutions for
    Action Recognition <https://arxiv.org/abs/1711.11248>`_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.pretrained2d is False
        assert self.conv_cfg['type'] == 'Conv2plus1d'
        # bn = nn.SyncBatchNorm(512, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        # activate = nn.ReLU(inplace=True)
        # self.last_ConvModule = nn.Sequential(last_Conv2plus1d(), bn, activate)
        # self.last_ConvModule.apply(init_weights)
        
    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        # h, w = ori
        x = self.conv1(x) # conv_s (bs, M, clip_len, ori/2, ori/2)+ bn_s + relu + conv_t (bs, 64, clip_len, ori/2, ori/2)
        x = self.maxpool(x) # maxpool 3d (bs, 64, clip_len, ori/4, ori/4)
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            # no pool2 in R(2+1)d
            x = res_layer(x) # after layer 1: (bs, 64, clip_len, ori/4, ori/4). after layer 2: (bs, 128, clip_len/2, ori/8, ori/8)
                             # after layer 3: (bs, 256, clip_len/4, ori/16, ori/16). after layer 4: (bs, 512, clip_len/8, ori/32, ori/32)

        # 自定义最后一层
        # x = self.last_ConvModule(x)
        return x
