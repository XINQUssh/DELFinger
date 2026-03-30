# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import math

from layers import (
    CMul, 
    Flatten, 
    ConcatTable, 
    Identity, 
    Reshape, 
    SpatialAttention2d, 
    WeightedSum2d,
    AutoEncoder,
    GeM)

__all__ = ['ghost_net']

def __freeze_weights__(module_dict, freeze=[]):
    for _, v in enumerate(freeze):
        module = module_dict[v]
        for param in module.parameters():
            param.requires_grad = False

def __load_weights_from__(module_dict, load_dict, modulenames):
    for modulename in modulenames:
        module = module_dict[modulename]
        print('loaded weights from module "{}" ...'.format(modulename))
        module.load_state_dict(load_dict[modulename])

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    

    
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    def __init__(self, atten, load_from = None, num_classes=1000, width=1.0):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs1 = [
            # k, t, c, SE, s 
            # stage1
            [[3,  16,  16, 0, 1]],
            # stage2
            [[3,  48,  24, 0, 2]],
            [[3,  72,  24, 0, 1]],
            # stage3
            [[5,  72,  40, 0.25, 2]],
            [[5, 120,  40, 0.25, 1]],
            # stage4
            [[3, 240,  80, 0, 2]],
            [[3, 200,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 480, 112, 0.25, 1],
             [3, 672, 112, 0.25, 1]
            ]
        ]
        self.cfgs2 = [
            # k, t, c, SE, s 
            # stage5
            [[5, 672, 160, 0.25, 2]],
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 0.25, 1]
            ]
        ]

        self.atten = atten
        self.load_from = load_from
        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        stages2 = []
        block = GhostBottleneck
        for cfg in self.cfgs1:
            layers1 = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers1.append(block(input_channel, hidden_channel, output_channel, k, s,
                              se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers1))
        
        self.blocks = nn.Sequential(*stages)     

        for cfg in self.cfgs2:
            layers2 = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers2.append(block(input_channel, hidden_channel, output_channel, k, s,
                              se_ratio=se_ratio))
                input_channel = output_channel
            stages2.append(nn.Sequential(*layers2))

        output_channel = _make_divisible(exp_size * width, 4)
        stages2.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        self.blocks2 = nn.Sequential(*stages2)

        self.attn = SpatialAttention2d(in_c=112, act_fn='relu')
        self.ae = AutoEncoder(in_c=112)
        self.attn_pooling = WeightedSum2d()

        # building last several layers
        output_channel = 960 #1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.gem_pool = GeM()
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)

        self.Flatten = Flatten()
        self.classifier = nn.Linear(output_channel, num_classes)

        self.attn_classifier = nn.Linear(112, num_classes)

        self.module_dict = {}
        self.module_dict['base'] = nn.Sequential(self.conv_stem, self.bn1, self.act1, self.blocks)
        self.module_dict['post'] = nn.Sequential(self.blocks2)
        self.module_dict['GeM'] = nn.Sequential(self.global_pool, self.conv_head, self.act2, self.Flatten, self.classifier)
        self.module_dict['attn'] = nn.Sequential(self.attn, self.ae, self.attn_pooling)
        self.module_dict['logits'] = nn.Sequential(self.attn_classifier)
        
        if atten == True:
            load_dict = torch.load(self.load_from)
            __load_weights_from__(self.module_dict, load_dict, modulenames=['base'])
            __freeze_weights__(self.module_dict, freeze=['base'])
            print('load model from "{}"'.format(load_from))
 
    def write_to(self, state):
        if self.atten == False:
            state['base'] = self.module_dict['base'].state_dict()
            state['post'] = self.module_dict['post'].state_dict()
            state['GeM']  = self.module_dict['GeM'].state_dict()
        elif self.atten == True:
            state['base'] = self.module_dict['base'].state_dict()
            state['attn'] = self.module_dict['attn'].state_dict()
            state['logits'] = self.module_dict['logits'].state_dict()
        else:
            assert self.stage in ['inference']
            raise ValueError('inference does not support model saving!')

    def forward_for_serving(self, x):
        '''
        This function directly returns attention score and raw features
        without saving to endpoint dict.
        '''
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        
        _,attn_x = self.ae(x)
        attn_score = self.attn(x)
        return attn_x.cpu(), attn_score.cpu()

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        
        if self.atten == True:
            feature = x
            _,attn_x = self.ae(x)
            attn_score = self.attn(x)
            x = self.attn_pooling([attn_x, attn_score])
            x = self.Flatten(x)
            x = self.attn_classifier(x)
            return x, feature, attn_x

        x = self.blocks2(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.Flatten(x)
        x = self.classifier(x)
        return x



def ghostnet(ncls, load_from, **kwargs):
    """
    Constructs a GhostNet model
    """
    return GhostNet(atten = False, load_from = load_from, num_classes = ncls, **kwargs)

def ghostnet_frozen(ncls, load_from, **kwargs):
    return GhostNet(atten = True, load_from = load_from, num_classes = ncls, **kwargs)


if __name__=='__main__':
    model = ghostnet()
    model.eval()
    print(model)
    input = torch.randn(32,3,160,160)
    y = model(input)
    print(y.size())