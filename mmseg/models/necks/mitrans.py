import torch.nn as nn, torch

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from mmcv.cnn import NonLocal2d

from mmseg.ops import resize
from ..builder import NECKS

@NECKS.register_module()
class MITransModule(BaseModule):
    def __init__(self, 
        num_convs=2,
        kernel_size=3,
        concat_input=True,
        dilation=1,
        in_channels=2048,
        channels=2048,
        reduction=2,
        use_scale=True,
        mode='embedded_gaussian',
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),):
        super().__init__()
        
        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                in_channels,
                channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        
        self.concat_input = concat_input
        if concat_input:
            self.conv_cat = ConvModule(
                in_channels + channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            
        self.nl_block = NonLocal2d(
            in_channels=channels,
            reduction=reduction,
            use_scale=use_scale,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            mode=mode)
    
    def forward(self, x):
        """Forward function."""
        # x = inputs
        output = self.convs[0](x)
        output = self.nl_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))

        return output