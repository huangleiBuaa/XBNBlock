# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import torch
from torch import nn
from torch.nn import functional as F
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import GroupItN
from maskrcnn_benchmark.modeling.poolers import Pooler


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, \
            "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, \
            "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn


def group_norm(out_channels, affine=True, divisor=1, dim=4):
    out_channels = out_channels // divisor
    dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
    eps = cfg.MODEL.GROUP_NORM.EPSILON # default: 1e-5
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups), 
        out_channels, 
        eps, 
        affine
    )


def Whitening_IGWItN(out_channels, dim=4, affine=True, divisor=1):
    out_channels = out_channels // divisor
    num_channels = cfg.MODEL.WHITENING.DIM_PER_GP // divisor
    num_groups = cfg.MODEL.WHITENING.NUM_GROUPS // divisor
    T = cfg.MODEL.WHITENING.T
    eps = cfg.MODEL.WHITENING.EPSILON # default: 1e-5
    return GroupItN(out_channels,
                         num_channels=num_channels,
                         num_groups=num_groups,
                         dim=dim,
                         T=T,
                         affine=affine,
                         eps=eps
                         )


class _Norm_config:
    normActivation = {'GN': group_norm, 'IGWItN': Whitening_IGWItN}





def make_conv3x3(
        in_channels,
        out_channels,
        dilation=1,
        stride=1,
        use_gn='No',
        use_relu=False,
        kaiming_init=True
):
    actNorm_flag = False if use_gn == 'No' else True
    conv = Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False if actNorm_flag else True
    )
    if kaiming_init:
        nn.init.kaiming_normal_(
            conv.weight, mode="fan_out", nonlinearity="relu"
        )
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
    if not actNorm_flag:
        nn.init.constant_(conv.bias, 0)
    module = [conv,]
    if actNorm_flag:
        module.append(_Norm_config.normActivation[use_gn](out_channels))
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv



def make_fc(dim_in, hidden_dim, use_gn='No'):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    '''
    if use_gn != 'No':
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        return nn.Sequential(fc, _Norm_config.normActivation[use_gn](hidden_dim, dim=2)) # use fully coonected- Itn
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc



def conv_with_kaiming_uniform(use_gn='No', use_relu=False):
    def make_conv(
            in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        actNorm_flag = False if use_gn=='No' else True
        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if actNorm_flag else True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not actNorm_flag:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if actNorm_flag:
            module.append(_Norm_config.normActivation[use_gn](out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv

