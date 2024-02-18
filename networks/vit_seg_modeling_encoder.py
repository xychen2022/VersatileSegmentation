from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class StdConv3d(nn.Conv3d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3, 4], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv3d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)

def conv3x3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv3d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)

def conv1x1x1(cin, cout, stride=1, bias=False):
    return StdConv3d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)

class Conv3dBlock(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, down_factor=1):
        super().__init__()
        
        self.change_dim = False
        if (cin != cout):
            self.change_dim = True
            self.proj = conv1x1x1(cin, cout, bias=True)
        
        self.conv1 = conv3x3x3(cin, cmid, bias=True)
        self.gn1 = nn.GroupNorm(cmid, cmid, eps=1e-6)
        
        self.conv2 = conv3x3x3(cmid, cout, bias=True)
        self.gn2 = nn.GroupNorm(cout, cout, eps=1e-6)
        
        self.relu = nn.ReLU(inplace=True)
        
        if down_factor != 1:
            self.downsample = nn.Upsample(scale_factor=down_factor, mode='trilinear', align_corners=True)
    
    def forward(self, x):
        
        residual = x
        if self.change_dim == True:
            residual = self.proj(residual)
            
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))

        y = self.relu(residual + y)
        
        if hasattr(self, 'downsample'):
            y = self.downsample(y)
        
        return y

class FeatureExtractor(nn.Module):
    
    def __init__(self, base_width, block_units, width_factor, in_channels):
        super().__init__()
        width = int(base_width * width_factor)
        self.width = width
        
        self.root = nn.Sequential(OrderedDict([
            ('conv1', StdConv3d(in_channels, width//2, kernel_size=5, stride=1, bias=True, padding=2)),
            ('gn1', nn.GroupNorm(width//2, width//2, eps=1e-6)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', StdConv3d(width//2, width, kernel_size=5, stride=1, bias=True, padding=2)),
            ('gn2', nn.GroupNorm(width, width, eps=1e-6)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        
        # self.pool = nn.Sequential(OrderedDict([
        #     ('conv', StdConv3d(width//2, width//2, kernel_size=5, stride=2, bias=True, padding=2)),
        #     ('gn', nn.GroupNorm(width//2, width//2, eps=1e-6)),
        #     ('relu', nn.ReLU(inplace=True)),
        # ]))
        
        self.pool = nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True)
    
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', Conv3dBlock(cin=width, cout=width*2, cmid=width))] +
                [(f'unit{i:d}', Conv3dBlock(cin=width*2, cout=width*2, cmid=width)) for i in range(2, block_units[0])] +
                [('unit'+str(block_units[0]), Conv3dBlock(cin=width*2, cout=width*4, cmid=width*2, down_factor=0.5))],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', Conv3dBlock(cin=width*4, cout=width*4, cmid=width*2))] +
                [(f'unit{i:d}', Conv3dBlock(cin=width*4, cout=width*4, cmid=width*2)) for i in range(2, block_units[1])] +
                [('unit'+str(block_units[1]), Conv3dBlock(cin=width*4, cout=width*8, cmid=width*2, down_factor=0.5))],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', Conv3dBlock(cin=width*8, cout=width*8, cmid=width*4))] +
                [(f'unit{i:d}', Conv3dBlock(cin=width*8, cout=width*8, cmid=width*4)) for i in range(2, block_units[2] + 1)] + 
                [('unit'+str(block_units[2]), Conv3dBlock(cin=width*8, cout=width*16, cmid=width*4, down_factor=0.5))],
                ))),
        ]))

    def forward(self, x):
        features = []
        
        b, c, in_size_z, in_size_y, in_size_x = x.size()
        full = x = self.root(x)
        x = self.pool(x)
        features.append(x)
        
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            features.append(x)
        
        x = self.body[-1](x)
        
        return x, features[::-1], full
