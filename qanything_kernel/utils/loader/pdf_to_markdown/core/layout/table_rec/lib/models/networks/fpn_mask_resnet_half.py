# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=False)

def save_map_g(x,name):
    path = '/home/rujiao.lrj/CenterNet_4point_Mask_4_rotate_offset/src/save_map.txt'
    f = open(path,'a+')
    f.write('---------------%s-------------------\n'%name)
    shape = list(x.shape)
    if shape[0]>3:
        c=3 
    else:
        c=shape[0]
    if shape[1]>16:
        w=16
    else:
        w=shape[1]
    if shape[2]>16:
        h=16
    else:
        h=shape[2]
    for i in range(c):
        for j in range(w):
             f.write('cln:%d,line:%d\n'%(i,j))
             string = ''
             for k in range(h):
                 string = string + str(x[i][j][k]) + ' ' 
             f.write(string+'\n')

def pad_same(x,ksize,stride=1,Pool=False):
    shape = x.shape
    n_in,w,h = shape[1],shape[2],shape[3]
    if h % stride == 0:
        pad_along_height = max(ksize - stride, 0)
    else:
        pad_along_height = max(ksize - (h % stride), 0)
    if w % stride == 0:
        pad_along_width = max(ksize - stride, 0)
    else:
        pad_along_width = max(ksize - (w % stride), 0)
    pad_bottom = pad_along_height // 2
    pad_top = pad_along_height - pad_bottom
    pad_right = pad_along_width // 2
    pad_left = pad_along_width - pad_right
    dim = (pad_left,pad_right,pad_top,pad_bottom)
    if Pool:
        dim = (pad_right,pad_left,pad_bottom,pad_top)
    x = F.pad(x,dim,"constant",value=0)
    return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        out = avg_out + max_out

        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2,1,3,padding=1,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        self.planes = planes

    def forward(self, x):
        shape = list(x.shape)
        residual = x
        x = pad_same(x,3,self.stride)

        out = self.conv1(x)
        '''
        if self.planes==64:
            if self.downsample is not None:
                save_map_g(out.cpu().numpy()[0],'layer1.0.conv1')
            else:
                save_map_g(out.cpu().numpy()[0],'layer1.1.conv1')
        '''
        out = self.bn1(out)
        '''
        if self.planes==64:
            if self.downsample is not None:
                save_map_g(out.cpu().numpy()[0],'layer1.0.bn1')
            else:
                save_map_g(out.cpu().numpy()[0],'layer1.1.bn1')
        '''
        out = self.relu(out)
        '''
        if self.planes==64:
            if self.downsample is not None:
                save_map_g(out.cpu().numpy()[0],'layer1.0.relu')
            else:
                save_map_g(out.cpu().numpy()[0],'layer1.1.relu')
        '''
        out = pad_same(out,3,1)
        out = self.conv2(out)
        '''
        if self.planes==64:
            if self.downsample is not None:
                save_map_g(out.cpu().numpy()[0],'layer1.0.conv2')
            else:
                save_map_g(out.cpu().numpy()[0],'layer1.1.conv2')
        '''
        out = self.bn2(out)
        '''
        if self.planes==64:
            if self.downsample is not None:
                save_map_g(out.cpu().numpy()[0],'layer1.0.bn2')
            else:
                save_map_g(out.cpu().numpy()[0],'layer1.1.bn2')
        '''
        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        '''
        if self.planes==64:
            if self.downsample is not None:
                save_map_g(out.cpu().numpy()[0],'res2a.relu')
            else:
                save_map_g(out.cpu().numpy()[0],'res2b.relu')
        '''
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)

        self.adaption3 = nn.Conv2d(256,256, kernel_size=1, stride=1, padding=0,bias=False)
        self.adaption2 = nn.Conv2d(128,256, kernel_size=1, stride=1, padding=0,bias=False)
        self.adaption1 = nn.Conv2d(64 ,256, kernel_size=1, stride=1, padding=0,bias=False)
        self.adaption0 = nn.Conv2d(64 ,256, kernel_size=1, stride=1, padding=0,bias=False)

        self.adaptionU1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0,bias=False)

        # used for deconv layers
        self.deconv_layers1 = self._make_deconv_layer(1, [256], [4],)
        self.deconv_layers2 = self._make_deconv_layer(1, [256], [4],)
        self.deconv_layers3 = self._make_deconv_layer(1, [256], [4],)
        self.deconv_layers4 = self._make_deconv_layer(1, [256], [4],)

        # self.final_layer = []

        for head in sorted(self.heads):
          num_output = self.heads[head]
          if head_conv > 0:
            inchannel = 256
            fc = nn.Sequential(
                nn.Conv2d(inchannel, head_conv,
                  kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output, 
                  kernel_size=1, stride=1, padding=0))
          else:
            inchannel = 256
            fc = nn.Conv2d(
              in_channels=inchannel,
              out_channels=num_output,
              kernel_size=1,
              stride=1,
              padding=0
          )
          self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        elif deconv_kernel == 7:
            padding = 3
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)
    
    def save_map(self,x,name):
        path = '/home/rujiao.lrj/CenterNet_4point_Mask_4_rotate_offset/src/save_map.txt'
        f = open(path,'a+')
        f.write('---------------%s-------------------\n'%name)
        shape = list(x.shape)
        if shape[0]>3:
            c=3 
        else:
            c=shape[0]
        if shape[1]>16:
            w=16
        else:
            w=shape[1]
        if shape[2]>16:
            h=16
        else:
            h=shape[2]
        for i in range(c):
            for j in range(w):
                 f.write('cln:%d,line:%d\n'%(i,j))
                 string = ''
                 for k in range(h):
                     string = string + str(x[i][j][k]) + ' ' 
                 f.write(string+'\n')

    def forward(self, x):
        #self.save_map(x.cpu().numpy()[0],'input')
        x = pad_same(x,7,2)
        x = self.conv1(x)
        #self.save_map(x.cpu().numpy()[0],'conv1')
        x = self.bn1(x)
        #self.save_map(x.cpu().numpy()[0],'bn1')
        x = self.relu(x)
        #self.save_map(x.cpu().numpy()[0],'conv1.relu')
        x0 = pad_same(x,3,2,True)
        x0 = self.maxpool(x0)
        #self.save_map(x0.cpu().numpy()[0],'pool1')
        x1 = self.layer1(x0)
        #self.save_map(x1.cpu().numpy()[0],'res2b.relu')
        x2 = self.layer2(x1)
        #self.save_map(x2.cpu().numpy()[0],'res3b.relu')
        x3 = self.layer3(x2)
        #self.save_map(x3.cpu().numpy()[0],'res4b.relu')
        x4 = self.layer4(x3)
        #self.save_map(x4.cpu().numpy()[0],'res5b.relu')
 
        x3_ = self.deconv_layers1(x4)
        #self.save_map(x3_.cpu().numpy()[0],'deconv_layers1.0')
        x3_ = self.adaption3(x3) + x3_
        #self.save_map(x3_.cpu().numpy()[0],'res4b_')
      
        x2_ = self.deconv_layers2(x3_)
        #self.save_map(x2_.cpu().numpy()[0],'deconv_layers2.0')
        x2_ = self.adaption2(x2) + x2_
        #self.save_map(x2_.cpu().numpy()[0],'res3b_')
 
        x1_ = self.deconv_layers3(x2_)
        #self.save_map(x1_.cpu().numpy()[0],'deconv_layers3.0')
        x1_ = self.adaption1(x1) + x1_
        #self.save_map(x1_.cpu().numpy()[0],'res2b_')
 
        x0_ = self.deconv_layers4(x1_) + self.adaption0(x0)
        x0_ = self.adaptionU1(x0_)
        #self.save_map(x0_.cpu().numpy()[0],'adaptionU1')

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x0_)
        return [ret]

        #for head in self.heads:
        #   self.save_map(ret[head].cpu().numpy()[0],head)

    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            for deconv_layer in [self.deconv_layers1,self.deconv_layers2,self.deconv_layers3]:
                for _, m in deconv_layer.named_modules():
                    if isinstance(m, nn.ConvTranspose2d):
                        nn.init.normal_(m.weight, std=0.001)
                        if self.deconv_with_bias:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                for head in self.heads:
                  final_layer = self.__getattr__(head)
                  for i, m in enumerate(final_layer.modules()):
                      if isinstance(m, nn.Conv2d):
                          if m.weight.shape[0] == self.heads[head]:
                              if 'hm' in head:
                                  nn.init.constant_(m.bias, -2.19)
                              else:
                                  nn.init.normal_(m.weight, std=0.001)
                                  nn.init.constant_(m.bias, 0)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            #self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net_fpn_mask_half(num_layers, heads, head_conv):
  block_class, layers = resnet_spec[num_layers]

  model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
  model.init_weights(num_layers, pretrained=True)
  return model
