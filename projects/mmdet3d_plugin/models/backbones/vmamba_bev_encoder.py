import torch.utils.checkpoint as checkpoint
from torch import nn

from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet3d.models import BACKBONES
# import VMamba
from VMamba.classification.models.vmamba import Backbone_VSSM, VSSBlock, LayerNorm2d
from timm.models.layers import DropPath, trunc_normal_

import torch.utils.checkpoint as checkpoint
from torch import nn

from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmengine.model import BaseModule
from mmdet3d.models import BACKBONES
import torch
from collections import OrderedDict
import time
from mmcv.cnn import build_conv_layer, build_norm_layer
import sys
from ops_dcnv3 import modules as opsm

def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

class MLPLayer_dcnv3(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class InternImageLayer_dcnv3(nn.Module):
    r""" Basic layer of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 channels,
                 groups,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 offset_scale=1.0,
                 with_cp=False,
                #  dcn_output_bias=False,
                #  mlp_fc2_bias=False,
                 dw_kernel_size=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False,
                 remove_center=False,
                 kernel_size=3): # for InternImage-H/G
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1_name, norm1 = build_norm_layer(dict(type='LN'), channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.post_norm = post_norm
        core_op = getattr(opsm, 'DCNv3')
        self.dcn = core_op(
            channels=channels,
            kernel_size=kernel_size,#5
            stride=1,
            pad=kernel_size//2,#2
            dilation=1,
            group=groups,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size, # for InternImage-H/G
            center_feature_scale=center_feature_scale, # for InternImage-H/G
            remove_center=remove_center,  # for InternImage-H/G
        )
        # self.dcn = core_op(
        #     channels=channels,
        #     kernel_size=5,
        #     stride=2,
        #     group=groups,
        #     offset_scale=offset_scale,
        #     dw_kernel_size=dw_kernel_size,
        #     # output_bias=dcn_output_bias,
        # )
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2_name, norm2 = build_norm_layer(dict(type='LN'), channels, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.mlp = MLPLayer_dcnv3(in_features=channels,
                            hidden_features=int(channels * mlp_ratio),
                            act_layer=act_layer,
                            drop=drop,
                            # mlp_fc2_bias=mlp_fc2_bias
                            )
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
        self.res_post_norm = res_post_norm
        if res_post_norm:
            self.norm3_name, res_post_norm1 = build_norm_layer(dict(type='LN'), channels, postfix=3)
            self.norm4_name, res_post_norm2 = build_norm_layer(dict(type='LN'), channels, postfix=4)
            self.add_module(self.norm3_name, res_post_norm1)
            self.add_module(self.norm4_name, res_post_norm2)
    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                elif self.res_post_norm: # for InternImage-H/G
                    x = x + self.drop_path(self.res_post_norm1(self.dcn(self.norm1(x))))
                    x = x + self.drop_path(self.res_post_norm2(self.mlp(self.norm2(x))))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

class DCNBlock_dcnv3(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None,
                 kernel_size=3,
                 group=4,
                 layer_scale=1e-5,
                 offset_scale=1.0,
                 mlp_ratio=4.0,
                 post_norm=True,
                 dw_kernel_size=3):
        super(DCNBlock_dcnv3, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        # self.conv2 = build_conv_layer(
        #     conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.dcnv3 = InternImageLayer_dcnv3(
                    channels=planes,
                    groups=group,
                    mlp_ratio=mlp_ratio,
                    drop=0.0,
                    drop_path=0.0,
                    act_layer='GELU',
                    norm_layer='LN',
                    post_norm=post_norm,
                    layer_scale=layer_scale,
                    offset_scale=offset_scale,
                    with_cp=False,
                    # dcn_output_bias=False,
                    # mlp_fc2_bias=False,
                    dw_kernel_size=dw_kernel_size, # for InternImage-H/G
                    res_post_norm=False, # for InternImage-H/G
                    center_feature_scale=False, # for InternImage-H/G,
                    kernel_size=kernel_size
                )
        
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            N, C, H, W = out.shape
            # out_shape = [H, W]
            out = out.permute(0,2,3,1)
            out = self.dcnv3(out)
            out = out.permute(0,3,1,2)
            # out = out.permute(0,2,1).reshape(N, C, H, W)
            
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

@BACKBONES.register_module()
class CustomResNetMambaPosDCN_dcnv3(nn.Module):
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
            drop_path_rate=0.1,
            norm_layer="ln2d",
            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            use_checkpoint=False, 
            posembed=False, 
            posembed_list=[0],
            ### DCNv3 config
            kernel_size_list=[3, 3, 3],
            groups=[4,4,4],
            LAYER_SCALE=1e-5,
            OFFSET_SCALE=1.0,
            MLP_RATIO=4.0,
            POST_NORM=True,
            dw_kernel_size=3,
            # for 4D
            time4D=False,
            

    ):
        super(CustomResNetMambaPosDCN_dcnv3, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids

        depths = [d-1 for d in num_layer]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.channel_first = channel_first
        self.pos_embed_dict = None
        if posembed:
            self.pos_embed_dict = nn.ParameterDict()
            self.posembed_list = posembed_list
            for i in posembed_list:
                self.pos_embed_dict[str(i)] = self._pos_embed(numC_input*(2**(i+1)), 1, 100//(2**i)) if not time4D else self._pos_embed(numC_input*(2**(i)), 1, 100//(2**i))

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)
        
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                # 在第一个block中对输入进行downsample
                layer = [Bottleneck(inplanes=curr_numC, planes=num_channels[i]//4, stride=stride[i],
                                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([Bottleneck(inplanes=curr_numC, planes=num_channels[i]//4, stride=1,
                                         downsample=None, norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
                
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                # 在第一个block中对输入进行downsample
                cnn_layer= [DCNBlock_dcnv3(inplanes=curr_numC, planes=num_channels[i], stride=stride[i],
                                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                    norm_cfg=norm_cfg, kernel_size=kernel_size_list[i], group=groups[i], layer_scale=LAYER_SCALE, offset_scale=OFFSET_SCALE, 
                                    mlp_ratio=MLP_RATIO, post_norm=POST_NORM, dw_kernel_size=dw_kernel_size)]
                # cnn_layer = [BasicBlock(inplanes=curr_numC, planes=num_channels[i], stride=stride[i],
                #                     downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                #                     norm_cfg=norm_cfg)]
                vss_layer = []
                curr_numC = num_channels[i]
                # layer.extend([BasicBlock(inplanes=curr_numC, planes=num_channels[i], stride=1,
                #                           downsample=None, norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
                vss_layer.extend([VSSBlock(hidden_dim=curr_numC, 
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        channel_first=channel_first,
                        ssm_d_state=ssm_d_state,
                        ssm_ratio=ssm_ratio,
                        ssm_dt_rank=ssm_dt_rank,
                        ssm_act_layer=ssm_act_layer,
                        ssm_conv=ssm_conv,
                        ssm_conv_bias=ssm_conv_bias,
                        ssm_drop_rate=ssm_drop_rate,
                        ssm_init=ssm_init,
                        forward_type=forward_type,
                        mlp_ratio=mlp_ratio,
                        mlp_act_layer=mlp_act_layer,
                        mlp_drop_rate=mlp_drop_rate,
                        gmlp=gmlp,
                        use_checkpoint=use_checkpoint,)])
                layers.append(nn.Sequential(OrderedDict(
                                cnn_layer=nn.Sequential(*cnn_layer,),
                                vss_layer=nn.Sequential(*vss_layer,),
                            )))
                # layers.append(nn.Sequential(*layer))
        else:
            assert False

        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp
        
    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def forward(self, x):
        """
        Args:
            x: (B, C=64, Dy, Dx)
        Returns:
            feats: List[
                (B, 2*C, Dy/2, Dx/2),
                (B, 4*C, Dy/4, Dx/4),
                (B, 8*C, Dy/8, Dx/8),
            ]
        """
        def layer_forward(l, x, i):
            x = l.cnn_layer(x)
            if self.pos_embed_dict is not None and i in self.posembed_list:
                pos_embed = self.pos_embed_dict[str(i)]
                pos_embed = pos_embed.permute(0, 2, 3, 1) if not self.channel_first else pos_embed
                x = x + pos_embed
            x = l.vss_layer(x)
            return x

        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer_forward(layer, x_tmp, lid)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

@BACKBONES.register_module()
class CustomResNetMambaPosDCN4D_dcnv3(nn.Module):
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
            drop_path_rate=0.1,
            norm_layer="ln2d",
            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            use_checkpoint=False, 
            posembed=False, 
            posembed_list=[],

    ):
        super(CustomResNetMambaPosDCN4D_dcnv3, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids

        depths = [d-1 for d in num_layer]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.channel_first = channel_first
        self.pos_embed_dict = None
        if posembed:
            self.pos_embed_dict = nn.ParameterDict()
            self.posembed_list = posembed_list
            for i in posembed_list:
                self.pos_embed_dict[str(i)] = self._pos_embed(numC_input*(2**(i)), 1, 100//(2**i))

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)
        
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                # 在第一个block中对输入进行downsample
                layer = [Bottleneck(inplanes=curr_numC, planes=num_channels[i]//4, stride=stride[i],
                                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([Bottleneck(inplanes=curr_numC, planes=num_channels[i]//4, stride=1,
                                         downsample=None, norm_cfg=norm_cfg) for _ in range(num_layer[i] - 1)])
                
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                # 在第一个block中对输入进行downsample
                cnn_layer= [DCNBlock_dcnv3(inplanes=curr_numC, planes=num_channels[i], stride=stride[i],
                                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                                    norm_cfg=norm_cfg)]
                vss_layer = []
                curr_numC = num_channels[i]
                vss_layer.extend([VSSBlock(hidden_dim=curr_numC, 
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        channel_first=channel_first,
                        ssm_d_state=ssm_d_state,
                        ssm_ratio=ssm_ratio,
                        ssm_dt_rank=ssm_dt_rank,
                        ssm_act_layer=ssm_act_layer,
                        ssm_conv=ssm_conv,
                        ssm_conv_bias=ssm_conv_bias,
                        ssm_drop_rate=ssm_drop_rate,
                        ssm_init=ssm_init,
                        forward_type=forward_type,
                        mlp_ratio=mlp_ratio,
                        mlp_act_layer=mlp_act_layer,
                        mlp_drop_rate=mlp_drop_rate,
                        gmlp=gmlp,
                        use_checkpoint=use_checkpoint,)])
                layers.append(nn.Sequential(OrderedDict(
                                cnn_layer=nn.Sequential(*cnn_layer,),
                                vss_layer=nn.Sequential(*vss_layer,),
                            )))
        else:
            assert False

        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp
        
    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def forward(self, x):
        """
        Args:
            x: (B, C=64, Dy, Dx)
        Returns:
            feats: List[
                (B, 2*C, Dy/2, Dx/2),
                (B, 4*C, Dy/4, Dx/4),
                (B, 8*C, Dy/8, Dx/8),
            ]
        """
        def layer_forward(l, x, i):
            x = l.cnn_layer(x)
            if self.pos_embed_dict is not None and i in self.posembed_list:
                pos_embed = self.pos_embed_dict[str(i)]
                pos_embed = pos_embed.permute(0, 2, 3, 1) if not self.channel_first else pos_embed
                x = x + pos_embed
            x = l.vss_layer(x)

            return x
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer_forward(layer, x_tmp, lid)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats