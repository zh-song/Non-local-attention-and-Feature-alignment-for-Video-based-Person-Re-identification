from __future__ import absolute_import

from .ResNet import *

__factory = {
    'ap3dres50': AP3DResNet50,
    'ap3dnlres50': AP3DNLResNet50,
    'Baseline':Baseline,
    'NLResNet50':NLResNet50,
    # 'resnet50tp': ResNet50TP,
    'resnet50ta': ResNet50TP,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
