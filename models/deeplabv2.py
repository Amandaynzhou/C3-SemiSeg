"""
This is the implementation of DeepLabv2 without multi-scale inputs. This implementation uses ResNet-101 by default.
"""

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from base import BaseModel
import numpy as np
affine_par = True
from models.pretrain import PRETRAIN_PATH
import pdb
from collections import OrderedDict
from utils.checkpoint import load_state_dict,load_coco_resnet_101

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        # print(out.shape)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
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

class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out


from torchvision.models._utils import IntermediateLayerGetter
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes,  return_feat_and_logit = False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.return_feat_and_logit = return_feat_and_logit

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)


    def forward(self, x):
        if self.return_feat_and_logit:
            out = OrderedDict()
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            out['layer1'] = x
            x = self.layer2(x)
            out['layer2'] = x
            x = self.layer3(x)
            out['layer3'] = x
            x = self.layer4(x)
            out['layer4'] = x
            return out, x

        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)


        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj+=1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10*args.learning_rate}]

class DeepLabV2(BaseModel):

    def __init__(self, num_classes,  backbone='resnet101', pretrained=True, distributed = False, cloud = False,
                 pretrain_from= 'imagenet', global_branch = None,return_feat_and_logit = False,**_):
        super(DeepLabV2, self).__init__()
        assert ('resnet' in backbone)
        self.body = ResNet(Bottleneck,[3, 4, 23, 3], num_classes, return_feat_and_logit)
        self.decoder = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        self.return_feat_and_logit = return_feat_and_logit


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # for i in m.parameters():
                #     i.requires_grad = False

        if pretrained:
            pretrain_weight_name ='resnet101_'+pretrain_from
            pretrain_path = PRETRAIN_PATH[pretrain_weight_name]
            if distributed or cloud: pretrain_path = pretrain_path.replace('../pretrains', '/cache/pretrains')
            print('load from path', pretrain_path)
            weight = torch.load(pretrain_path, map_location=torch.device("cpu"))
            if 'coco' in pretrain_weight_name:
                print('load pretrain from coco...')
                # strip backbone in model name
                load_coco_resnet_101(self.body, weight)
            else:
                print('load pretrain from imagenet...')
                load_state_dict(self.body, weight)

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
            elif isinstance(module,nn.SyncBatchNorm):module.eval()

    def unfreeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
            elif isinstance(module, nn.SyncBatchNorm): module.train()
                # for i in module.parameters():
                #     i.requires_grad = True

    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        inter = self.body(x) # 1/8 scale

        if self.return_feat_and_logit:
            feats, out = inter
            x = self.decoder(out)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
            return feats, x
        else:
            x = self.decoder(inter)
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

            return x

    def get_backbone_params(self):
        return self.body.get_1x_lr_params_NOscale()

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.decoder.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def get_decoder_params(self):
        return self.get_10x_lr_params()

    def get_modulate_params(self):
        b = [self.body, self.decoder]
        for part in b:
            for name, para in part.named_parameters():
                if 'modulate' in name:
                    yield para

    def tuning_modulate_params(self):
        b = [self.body, self.decoder]
        for part in b:
            for name, para in part.named_parameters():
                if 'modulate' in name:
                    para.requires_grad = True

    def get_bn_params(self):
        for name, para in self.body.named_parameters():
            if 'bn' in name:
                yield para
