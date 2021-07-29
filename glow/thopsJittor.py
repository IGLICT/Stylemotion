
import jittor as jt
from jittor import init
from jittor import nn

def onehot(y, num_classes):
    y_onehot = torch.zeros(y.shape[0], num_classes).to(y.device)
    if (len(y.shape) == 1):
        y_onehot = y_onehot.scatter_(1, y.unsqueeze((- 1)), 1)
    elif (len(y.shape) == 2):
        y_onehot = y_onehot.scatter_(1, y, 1)
    else:
        raise ValueError('[onehot]: y should be in shape [B], or [B, C]')
    return y_onehot

def sum(tensor, dim=None, keepdim=False):
    if (dim is None):
        return torch.sum(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.sum(dim=d, keepdim=True)
        if (not keepdim):
            for (i, d) in enumerate(dim):
                tensor.squeeze_((d - i))
        return tensor

def mean(tensor, dim=None, keepdim=False):
    if (dim is None):
        return torch.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if (not keepdim):
            for (i, d) in enumerate(dim):
                tensor.squeeze_((d - i))
        return tensor

def split_feature(tensor, type='split'):
    '\n    type = ["split", "cross"]\n    '
    C = tensor.shape[1]
    if (type == 'split'):
        return (tensor[:, :(C // 2), ...], tensor[:, (C // 2):, ...])
    elif (type == 'cross'):
        return (tensor[:, 0::2, ...], tensor[:, 1::2, ...])

def cat_feature(tensor_a, tensor_b):
    return jt.contrib.concat((tensor_a, tensor_b), dim=1)

def pixels(tensor):
    return int((tensor.shape[2] * tensor.shape[3]))

def timesteps(tensor):
    return int(tensor.shape[2])

