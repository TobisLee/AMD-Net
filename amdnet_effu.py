# Copyright (c) OpenMMLab. All rights reserved.
from statistics import mode
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (UPSAMPLE_LAYERS, ConvModule, build_activation_layer,
                      build_norm_layer, DepthwiseSeparableConvModule)
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.models.utils import CBAMBlock, SELayer
from mmseg.ops import Upsample
from ..builder import BACKBONES
from ..utils import UpConvBlock


class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None):
        super(BasicConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out


@BACKBONES.register_module()
class AMDNet_EFFU(BaseModule):
    """AMDNet_EFFU backbone.

    This backbone is the implementation of `U-Net: Convolutional Networks
    for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_.

    Args:
        in_channels (int): Number of input image channels. Default" 3.
        base_channels (int): Number of base channels of each stage.
            The output channels of the first stage. Default: 64.
        num_stages (int): Number of stages in encoder, normally 5. Default: 5.
        strides (Sequence[int 1 | 2]): Strides of each stage in encoder.
            len(strides) is equal to num_stages. Normally the stride of the
            first stage in encoder is 1. If strides[i]=2, it uses stride
            convolution to downsample in the correspondence encoder stage.
            Default: (1, 1, 1, 1, 1).
        enc_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondence encoder stage.
            Default: (2, 2, 2, 2, 2).
        dec_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondence decoder stage.
            Default: (2, 2, 2, 2).
        downsamples (Sequence[int]): Whether use MaxPool to downsample the
            feature map after the first stage of encoder
            (stages: [1, num_stages)). If the correspondence encoder stage use
            stride convolution (strides[i]=2), it will never use MaxPool to
            downsample, even downsamples[i-1]=True.
            Default: (True, True, True, True).
        enc_dilations (Sequence[int]): Dilation rate of each stage in encoder.
            Default: (1, 1, 1, 1, 1).
        dec_dilations (Sequence[int]): Dilation rate of each stage in decoder.
            Default: (1, 1, 1, 1).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Notice:
        The input image size should be divisible by the whole downsample rate
        of the encoder. More detail of the whole downsample rate can be found
        in UNet._check_input_divisible.
    """

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 norm_eval=False,
                 dcn=None,
                 plugins=None,
                 pretrained=None,
                 init_cfg=None):
        super(AMDNet_EFFU, self).__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        assert len(strides) == num_stages, \
            'The length of strides should be equal to num_stages, '\
            f'while the strides is {strides}, the length of '\
            f'strides is {len(strides)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(enc_num_convs) == num_stages, \
            'The length of enc_num_convs should be equal to num_stages, '\
            f'while the enc_num_convs is {enc_num_convs}, the length of '\
            f'enc_num_convs is {len(enc_num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dec_num_convs) == (num_stages-1), \
            'The length of dec_num_convs should be equal to (num_stages-1), '\
            f'while the dec_num_convs is {dec_num_convs}, the length of '\
            f'dec_num_convs is {len(dec_num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(downsamples) == (num_stages-1), \
            'The length of downsamples should be equal to (num_stages-1), '\
            f'while the downsamples is {downsamples}, the length of '\
            f'downsamples is {len(downsamples)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(enc_dilations) == num_stages, \
            'The length of enc_dilations should be equal to num_stages, '\
            f'while the enc_dilations is {enc_dilations}, the length of '\
            f'enc_dilations is {len(enc_dilations)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dec_dilations) == (num_stages-1), \
            'The length of dec_dilations should be equal to (num_stages-1), '\
            f'while the dec_dilations is {dec_dilations}, the length of '\
            f'dec_dilations is {len(dec_dilations)}, and the num_stages is '\
            f'{num_stages}.'
        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval
        self.base_channels = base_channels

        # self.encoder = nn.ModuleList()
        # self.decoder = nn.ModuleList()

        self.pool_16= nn.MaxPool2d(16, 16, ceil_mode=True)
        self.pool_8 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.pool_4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.pool_2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.up_2   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4   = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8   = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16  = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        enc_channels = []
        for i in range(num_stages):
            inp_channels = base_channels * 2**i
            enc_channels.append(inp_channels)
        enc_channels = np.array(enc_channels)

        # Encoder Feature Fuse Block (EFFU): CBAM + Conv1*1 + (Conv3*3)*2
        self.effu_cbam = nn.ModuleList()
        for i in range(1, num_stages):
            self.effu_cbam.append(CBAMBlock(np.sum(enc_channels[:i])))

        self.effu_c1 = nn.ModuleList()
        for i in range(1, num_stages):
            effu_c1_block = []
            effu_c1_block.append(
                ConvModule(
                    in_channels=np.sum(enc_channels[:i]),
                    out_channels=enc_channels[i-1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.effu_c1.append((nn.Sequential(*effu_c1_block)))

        # effu (conv3*3)*2
        self.effu_c2 = nn.ModuleList()
        for i in range(num_stages):
            enc_conv_block = []
            enc_conv_block.append(
                BasicConvBlock(
                    in_channels=in_channels,
                    out_channels=base_channels * 2**i,
                    num_convs=enc_num_convs[i],
                    stride=strides[i],
                    dilation=enc_dilations[i],
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    dcn=None,
                    plugins=None))
            self.effu_c2.append((nn.Sequential(*enc_conv_block)))
            in_channels = base_channels * 2**i

        self.decoder = nn.ModuleList()
        for i in range(num_stages-1):
            self.decoder.append(
                BasicConvBlock(
                    in_channels=enc_channels[i]+enc_channels[i+1],
                    out_channels=enc_channels[i],
                    num_convs=2,
                    stride=1,
                    dilation=1,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    dcn=None,
                    plugins=None))


    def forward(self, x):
        self._check_input_divisible(x)
        x0 = self.effu_c2[0](x)

        # print("x0 shape: ", x0.shape)

        x1 = self.effu_c2[1](
            self.effu_c1[0](
                self.effu_cbam[0](
                    self.pool_2(x0)
                )
            )
        )

        x2 = self.effu_c2[2](
            self.effu_c1[1](
                self.effu_cbam[1](
                    torch.cat([
                        self.pool_4(x0),
                        self.pool_2(x1)
                    ], dim=1)
                )
            )
        )

        x3 = self.effu_c2[3](
            self.effu_c1[2](
                self.effu_cbam[2](
                    torch.cat([
                        self.pool_8(x0),
                        self.pool_4(x1),
                        self.pool_2(x2)
                    ], dim=1)
                )
            )
        )

        x4 = self.effu_c2[4](
            self.effu_c1[3](
                self.effu_cbam[3](
                    torch.cat([
                        self.pool_16(x0),
                        self.pool_8(x1),
                        self.pool_4(x2),
                        self.pool_2(x3)
                    ], dim=1)
                )
            )
        )

        dec3 = self.decoder[3](
            torch.cat([
                x3,
                self.up_2(x4)
            ], dim=1)
        )

        dec2 = self.decoder[2](
            torch.cat([
                x2,
                self.up_2(dec3)
            ], dim=1)
        )

        dec1 = self.decoder[1](
            torch.cat([
                x1,
                self.up_2(dec2)
            ], dim=1)
        )

        dec0 = self.decoder[0](
            torch.cat([
                x0,
                self.up_2(dec1)
            ], dim=1)
        )

        dec_outs = [x4, dec3, dec2, dec1, dec0]

        return dec_outs


    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(AMDNet_EFFU, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) \
            and (w % whole_downsample_rate == 0),\
            f'The input image size {(h, w)} should be divisible by the whole '\
            f'downsample rate {whole_downsample_rate}, when num_stages is '\
            f'{self.num_stages}, strides is {self.strides}, and downsamples '\
            f'is {self.downsamples}.'
