import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deeplabv3_plus import assp_branch
from utils.helpers import initialize_weights
import pdb
from utils.dist_utils import concat_all_gather, get_world_size

class Debug(nn.Module):
    def __init__(self):
        super(Debug, self).__init__()
        self.c1 = nn.Conv2d(3, 256, (1, 1), 1, 0, 1, bias=False)
        self.bn = nn.BatchNorm2d(256, affine=True, track_running_stats=True)

        self.relu= nn.ReLU(False)

    def forward(self,data):
        a = self.c1(data)
        a = self.bn(a)
        a = F.adaptive_avg_pool2d(a, (1,1))
        a = self.relu(a)
        return  a
    def freeze_bn(self):
        pass

class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 32, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Table 2, best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(32+32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(32, num_classes, 1, stride=1),
        )
        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x

class EmbedLayers(nn.Module):
    def __init__(self, multiple_layer = False, embed_dim = 32, bn = False, **kwargs):
        super(EmbedLayers, self).__init__()
        self.multiple_layer = multiple_layer
        self.bn = bn
        if multiple_layer:
            self.emb_1 = nn.Conv2d(256, embed_dim, (1, 1), 1, 0, 1, bias=False)
            self.emb_2 = nn.Conv2d(512, embed_dim, (1, 1), 1, 0, 1, bias=False)
            self.emb_3 = nn.Conv2d(1024, embed_dim, (1, 1), 1, 0, 1, bias=False)
            self.emb_4 = nn.Conv2d(2048, embed_dim, (1, 1), 1, 0, 1, bias=False)
            if bn:
                self.bn1 = nn.BatchNorm2d(embed_dim,affine=True, track_running_stats=True)
                self.bn2 = nn.BatchNorm2d(embed_dim, affine=True, track_running_stats=True)
                self.bn3 = nn.BatchNorm2d(embed_dim, affine=True, track_running_stats=True)
                self.bn4 = nn.BatchNorm2d(embed_dim, affine=True, track_running_stats=True)
        else:
            self.emb_1 = nn.Conv2d(2048, embed_dim, (1, 1), 1, 0, 1, bias=False)
            if bn:
                self.bn1 = nn.BatchNorm2d(embed_dim,affine=True, track_running_stats=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat):
        if self.multiple_layer:
            emb1 = self.emb_1(feat['layer1'])
            emb2 = self.emb_2(feat['layer2'])
            emb3 = self.emb_3(feat['layer3'])
            emb4 = self.emb_4(feat['layer4'])
            if self.bn:
                emb1 = self.bn1(emb1)
                emb2 = self.bn2(emb2)
                emb3 = self.bn3(emb3)
                emb4 = self.bn4(emb4)
            return emb1, emb2, emb3, emb4
        else:
            if isinstance(feat, dict):
                emb1 = self.emb_1(feat['layer4'])
            else:
                emb1 = self.emb_1(feat)
            if self.bn:
                emb1 = self.bn1(emb1)
                # emb1 = F.relu(emb1)
            return emb1


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.conv_1 = nn.Conv2d(input_dim, output_dim, (1, 1), 1, 0, 1, bias=True)
        # self.bn1 = nn.BatchNorm2d(output_dim, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(output_dim, output_dim, (1, 1), 1, 0, 1, bias=True)
        # self.bn2 = nn.BatchNorm2d(output_dim, affine=True, track_running_stats=True)

    def forward(self, feature):
        feature = self.conv_1(feature)
        # feature = self.bn1(feature)
        feature = self.relu1(feature)
        feature = self.conv_2(feature)
        # feature = self.bn2(feature)

        return feature

class MLP4C(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP4C, self).__init__()
        self.conv_1 = nn.Conv2d(input_dim, output_dim, (1, 1), 1, 0, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(output_dim, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(output_dim, output_dim, (1, 1), 1, 0, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(output_dim, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv_3 = nn.Conv2d(output_dim, output_dim, (1, 1), 1, 0, 1, bias=True)
        self.bn3 = nn.BatchNorm2d(output_dim, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv_4 = nn.Conv2d(output_dim, output_dim, (1, 1), 1, 0, 1, bias=True)
        self.bn4 = nn.BatchNorm2d(output_dim, affine=True, track_running_stats=True)


    def forward(self, feature):
        feature = self.conv_1(feature)
        feature = self.bn1(feature)
        feature = self.relu1(feature)
        feature = self.conv_2(feature)
        feature = self.bn2(feature)
        feature = self.relu2(feature)
        feature = self.conv_3(feature)
        feature = self.bn3(feature)
        feature = self.relu3(feature)
        feature = self.conv_4(feature)
        feature = self.bn4(feature)

        return feature


class ASSPv3(nn.Module):
    def __init__(self, in_channels,embed_dim, output_stride):
        super(ASSPv3, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = assp_branch(in_channels, embed_dim, 1, dilation=dilations[0])
        self.aspp2 = assp_branch(in_channels, embed_dim, 3, dilation=dilations[1])
        self.aspp3 = assp_branch(in_channels, embed_dim, 3, dilation=dilations[2])
        self.aspp4 = assp_branch(in_channels, embed_dim, 3, dilation=dilations[3])

        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(embed_dim * 5, 32, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x = self.bn1(x)
        x = self.dropout(self.relu(x))

        return x


class ASPPEmbedLayers(nn.Module):
    def __init__(self, embed_dim = 32, **kwargs):
        super(ASPPEmbedLayers, self).__init__()
        self.emb_1 = ASSPv3(2048, embed_dim, 16)
        self.decoder = Decoder(256, embed_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat):

        emb1 = self.emb_1(feat['layer4'])
        out = self.decoder(emb1, feat['layer1'])
        return out


class OCREmbedLayers(nn.Module):
    def __init__(self, embed_dim = 32, num_class = 19,scale = 1, **kwargs):
        super(OCREmbedLayers, self).__init__()
        self.cls_num = num_class
        self.scale = scale
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(2048, embed_dim,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, feats, probs, is_teacher= False):
        # gather prototype
        if isinstance(feats, dict):
            feat = feats['layer4']
        else:
            feat = feats
        if get_world_size() > 1 and is_teacher:
            feat, idx_unshuffle = self._batch_shuffle_ddp(feat)
        probs = F.interpolate(probs, (feat.size(2),feat.size(3)), mode='bilinear', align_corners=True)
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feat = self.conv3x3_ocr(feat)
        feat = feat.view(batch_size, feat.size(1), -1)
        feat = feat.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feat)
        # ocr_context = torch.matmul(probs, feat) \
        #     .permute(0, 2, 1).unsqueeze(3)  # batch x k x c
        k = nn.functional.normalize(ocr_context, dim=2)
        # undo shuffle
        if get_world_size() > 1 and is_teacher:
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        return k


class MLPEmbedLayers(nn.Module):
    def __init__(self,embed_dim = 32, **kwargs):
        super(MLPEmbedLayers, self).__init__()
        MLPfunc = MLP
        self.emb_1 = MLPfunc(2048, embed_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat):
        if isinstance( feat,dict):
            emb1 = self.emb_1(feat['layer4'])
        else:
            emb1 = self.emb_1(feat)
        return emb1


class ReconstructModule(nn.Module):
    def __init__(self, multiple_layer = False, embed_dim = 32, bn = False, **kwargs):
        super(ReconstructModule, self).__init__()
        self.multiple_layer = multiple_layer
        self.bn = bn
        if multiple_layer:
            self.emb_1 = nn.Conv2d(embed_dim, 256, (1, 1), 1, 0, 1, bias=False)
            self.emb_2 = nn.Conv2d(embed_dim, 512, (1, 1), 1, 0, 1, bias=False)
            self.emb_3 = nn.Conv2d( embed_dim,1024, (1, 1), 1, 0, 1, bias=False)
            self.emb_4 = nn.Conv2d( embed_dim,2048, (1, 1), 1, 0, 1, bias=False)
            if bn:
                self.bn1 = nn.BatchNorm2d(embed_dim, affine=True, track_running_stats=True)
                self.bn2 = nn.BatchNorm2d(embed_dim, affine=True, track_running_stats=True)
                self.bn3 = nn.BatchNorm2d(embed_dim, affine=True, track_running_stats=True)
                self.bn4 = nn.BatchNorm2d(embed_dim, affine=True, track_running_stats=True)
        else:
            self.emb_1 = nn.Conv2d(embed_dim, 2048,  (1, 1), 1, 0, 1, bias=False)
            if bn:
                self.bn1 = nn.BatchNorm2d(2048, affine=True, track_running_stats=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, emb):
        if self.multiple_layer:
            emb1,emb2,emb3,emb4 = emb
            emb1 = self.emb_1(emb1)
            emb2 = self.emb_2(emb2)
            emb3 = self.emb_3(emb3)
            emb4 = self.emb_4(emb4)
            if self.bn:
                emb1 = self.bn1(emb1)
                emb2 = self.bn2(emb2)
                emb3 = self.bn3(emb3)
                emb4 = self.bn4(emb4)
            return emb1, emb2, emb3, emb4
        else:
            emb1 = self.emb_1(emb)
            if self.bn:
                emb1 = self.bn1(emb1)
                emb1 = F.relu(emb1)
            return emb1


class MLPReconstructModule(nn.Module):
    def __init__(self, multiple_layer = False, embed_dim = 32, deeper = False, **kwargs):
        super(MLPReconstructModule, self).__init__()
        self.multiple_layer = multiple_layer
        if deeper:
            MLPfunc = MLP4C
        else:
            MLPfunc = MLP

        if self.multiple_layer:
            self.emb_1 = MLPfunc(embed_dim, 256)
            self.emb_2 = MLPfunc(embed_dim, 512)
            self.emb_3 = MLPfunc(embed_dim, 1024)
            self.emb_4 = MLPfunc(embed_dim, 2048)
        else:
            self.emb_1 = MLPfunc(embed_dim, 2048)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, emb):
        if self.multiple_layer:
            emb1,emb2,emb3,emb4 = emb
            emb1 = self.emb_1(emb1)
            emb2 = self.emb_2(emb2)

            emb3 = self.emb_3(emb3)
            emb4 = self.emb_4(emb4)
            return emb1, emb2, emb3, emb4
        else:
            emb1 = self.emb_1(emb)
            emb1 = F.relu(emb1)
            return emb1

class ExtraClassifier(nn.Module):
    def __init__(self, embed_dim, out = 19, H = 256, W = 512):
        super(ExtraClassifier, self).__init__()
        self.cls = nn.Conv2d(embed_dim, out, (1,1), bias=False)
        self.H = H
        self.W = W
    def forward(self, inputs):
        out = self.cls(inputs)
        out = F.interpolate(out, size=(self.H, self.W), mode='bilinear', align_corners=True)
        return out

__embed_factory = {'mlp': MLPEmbedLayers,
             'aspp': ASPPEmbedLayers,
             'ocr': OCREmbedLayers,
            }

def create_embedding_layer(name, *args, **kwargs):
    """
    Create a embedding module instance.
    Parameters
    ----------
    name : str
    """
    if name not in __embed_factory:
        raise KeyError("Unknown loader:", name)
    return __embed_factory[name]( *args, **kwargs)
