import argparse
import torch
import torch.nn as nn
from .iterative_normalization_FlexGroup import IterNorm
from .group_whitening import GroupItN
from .group_whitening_SVD import GroupSVD
from .iterative_normalization_FlexGroupSigma import IterNormSigma


from ..utils import str2dict


class IdentityModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(IdentityModule, self).__init__()

    def forward(self, input: torch.Tensor):
        return input

def _GroupNorm(num_features, num_groups=32, eps=1e-5, affine=True, *args, **kwargs):
    if num_groups>num_features:
        print('------arrive maxum groub numbers of:', num_features)
        num_groups=num_features
    return nn.GroupNorm(num_groups, num_features, eps=eps, affine=affine)


def _LayerNorm(normalized_shape, eps=1e-5, affine=True, *args, **kwargs):
    return nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=affine)


def _BatchNorm(num_features, dim=4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, *args, **kwargs):
    return (nn.BatchNorm2d if dim == 4 else nn.BatchNorm1d)(num_features, eps=eps, momentum=momentum, affine=affine,
                                                            track_running_stats=track_running_stats)


def _InstanceNorm(num_features, dim=4, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, *args,
                  **kwargs):
    return (nn.InstanceNorm2d if dim == 4 else nn.InstanceNorm1d)(num_features, eps=eps, momentum=momentum,
                                                                  affine=affine,
                                                                  track_running_stats=track_running_stats)

def _Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, *args, **kwargs):
    """return first input"""
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


def _IdentityModule(x, *args, **kwargs):
    """return first input"""
    return IdentityModule()

def _Identity_fn(x, *args, **kwargs):
    """return first input"""
    return x


class _config:
    norm = 'BN'
    norm_cfg = {}
    norm_methods = {'No': _IdentityModule, 'BN': _BatchNorm, 'GN': _GroupNorm, 'LN': _LayerNorm, 'IN': _InstanceNorm, 'None': None, 'ItN': IterNorm, 'ItNSigma': IterNormSigma, 'IGWItN': GroupItN, 'IGWSVD':GroupSVD}


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Normalization Options')
    group.add_argument('--norm', default='No', help='Use which normalization layers? {' + ', '.join(
        _config.norm_methods.keys()) + '}' + ' (defalut: {})'.format(_config.norm))
    group.add_argument('--norm-cfg', type=str2dict, default={}, metavar='DICT', help='layers config.')
    return group

def getNormConfigFlag():
    flag = ''
    flag += _config.norm
    if str.find(_config.norm, 'GW')>-1 or str.find(_config.norm, 'GN')>-1:
        if _config.norm_cfg.get('num_groups') != None:
            flag += '_NG' + str(_config.norm_cfg.get('num_groups'))
    if str.find(_config.norm,'ItN') > -1:
        if _config.norm_cfg.get('T') != None:
            flag += '_T' + str(_config.norm_cfg.get('T'))
        if _config.norm_cfg.get('num_channels') != None:
            flag += '_NC' + str(_config.norm_cfg.get('num_channels'))

    if str.find(_config.norm,'DBN') > -1:
        flag += '_NC' + str(_config.norm_cfg.get('num_channels'))
    if _config.norm_cfg.get('affine') == False:
        flag += '_NoA'
    if _config.norm_cfg.get('momentum') != None:
        flag += '_MM' + str(_config.norm_cfg.get('momentum'))
    #print(_config.normConv_cfg)
    return flag

def setting(cfg: argparse.Namespace):
    print(_config.__dict__)
    for key, value in vars(cfg).items():
        #print(key)
        #print(value)
        if key in _config.__dict__:
            setattr(_config, key, value)
    #print(_config.__dict__)
    flagName =  getNormConfigFlag()
    print(flagName)
    return flagName


def Norm(*args, **kwargs):
    kwargs.update(_config.norm_cfg)
    if _config.norm == 'None':
        return None
    return _config.norm_methods[_config.norm](*args, **kwargs)

