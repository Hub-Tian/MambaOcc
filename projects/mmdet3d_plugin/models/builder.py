# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, build_from_cfg
from torch import nn
from mmcv.cnn import MODELS as MMCV_MODELS

DISTILL = Registry('distill', parent=MMCV_MODELS)
DISTILL_LOSSES = DISTILL
DISTILL_ADAPTERS = DISTILL
# DISTILL_LOSSES = Registry('distill_loss')
# DISTILL_ADAPTERS = Registry['distill_adpater']

def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_distill_loss(cfg):
    """Build distill loss."""
    return DISTILL_LOSSES.build(cfg)

def build_distill_adpater(cfg):
    """Build distill adpater."""
    return DISTILL_ADAPTERS.build(cfg)
