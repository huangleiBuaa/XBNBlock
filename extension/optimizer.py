import argparse
import torch
from .utils import str2dict
from .logger import get_logger

_methods = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam, 'adamax': torch.optim.Adamax,
            'RMSprop': torch.optim.RMSprop}


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Optimizer Option:')
    group.add_argument('-oo', '--optimizer', default='sgd', choices=_methods.keys(),
                       help='the optimizer method to train network {' + ', '.join(_methods.keys()) + '}')
    group.add_argument('-oc', '--optimizer-config', default={}, type=str2dict, metavar='DICT',
                       help='The configure for optimizer')
    group.add_argument('-wd', '--weight-decay', default=0, type=float, metavar='FLOAT',
                       help='weight decay (default: 0).')
    return

def add_grouped_weight_decay(model, weight_decay=1e-4):
    decay = []
    no_decay = []
    print(model.named_parameters())
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.find('WNScale') != -1:
            print('-----------WNScale no weight decay----------------')
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.}]

def setting(model: torch.nn.Module, cfg: argparse.Namespace, **kwargs):
    if cfg.optimizer == 'sgd':
        kwargs.setdefault('momentum', 0.9)
    if hasattr(cfg, 'lr'):
        kwargs['lr'] = cfg.lr
    kwargs['weight_decay'] = cfg.weight_decay
    kwargs.update(cfg.optimizer_config)
    #params = model.parameters()
    params = add_grouped_weight_decay(model, weight_decay=cfg.weight_decay)
    logger = get_logger()
    optimizer = _methods[cfg.optimizer](params, **kwargs)
    logger('==> Optimizer {}'.format(optimizer))
    return optimizer

