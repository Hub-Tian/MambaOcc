import torch.utils.checkpoint as checkpoint
from torch import nn

from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet3d.models import BACKBONES
# import VMamba
from VMamba.classification.models.vmamba import Backbone_VSSM

@BACKBONES.register_module()
class MM_VSSM(Backbone_VSSM):
    def __init__(self, *args, **kwargs):
        Backbone_VSSM.__init__(self, *args, **kwargs)