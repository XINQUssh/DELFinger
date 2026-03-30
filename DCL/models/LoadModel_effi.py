import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import math

import pdb
from efficientnet_pytorch import EfficientNet


class MainModel(nn.Module):
    def __init__(self, config):
        super(MainModel, self).__init__()
        self.use_dcl = config.use_dcl
        self.num_classes = config.numcls
        #self.backbone_arch = config.backbone
        self.use_Asoftmax = config.use_Asoftmax
        #print(self.backbone_arch)

        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes = self.num_classes)
        self.model = nn.Sequential(*list(self.model.children())[:-3])  #5*5*1280
        
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        #self._dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1280, self.num_classes)


        if self.use_dcl:
            if config.cls_2:
                self.classifier_swap = nn.Linear(1280, 2, bias=False)
            if config.cls_2xmul:
                self.classifier_swap = nn.Linear(1280, 2*self.num_classes, bias=False)
            self.Convmask = nn.Conv2d(1280, 1, 1, stride=1, padding=1, bias=True)
            self.avgpool2 = nn.AvgPool2d(1, stride=1)

        if self.use_Asoftmax:
            self.Aclassifier = AngleLinear(output_channel, self.num_classes, bias=False)

    def forward(self, x, last_cont=None):
        x = self.model(x)
        if self.use_dcl:
            mask = self.Convmask(x)
            mask = self.avgpool2(mask)
            mask = torch.tanh(mask)
            mask = mask.view(mask.size(0), -1)

        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
        out = []
        out.append(self.classifier(x))

        if self.use_dcl:
            out.append(self.classifier_swap(x))
            out.append(mask)

        if self.use_Asoftmax:
            if last_cont is None:
                x_size = x.size(0)
                out.append(self.Aclassifier(x[0:x_size:2]))
            else:
                last_x = self.model(last_cont)
                last_x = self.avgpool(last_x)
                last_x = last_x.view(last_x.size(0), -1)
                out.append(self.Aclassifier(last_x))

        return out

#######################################################################################################

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


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 2, groups=inp, dilation=2, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

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
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, 3, stride, relu=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        layers = [nn.Sequential(
            nn.Conv2d(3, output_channel, 5, 2, 1, dilation=2, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )]
        input_channel = output_channel

        # building inverted residual blocks
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x) #[batch_size, 20, 20, 160] 20*20*160    
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
  
def ghost_net(num_classes, **kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, s 
        [3,  16,  16, 0, 2],
        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],
        [5,  72,  40, 1, 1],
        [5, 120,  40, 1, 1],
        [3, 240,  80, 0, 1],
        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]
    return GhostNet(cfgs, num_classes,  **kwargs)