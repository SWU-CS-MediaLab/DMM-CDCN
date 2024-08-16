import copy
import torch
import torch.nn as nn
from torch.nn import init, BatchNorm1d
from resnet import resnet50
from model_utils import Incremental_Classifier, Normalize, GCN_3ADJ, ADJ_ZeroLayer, ADJ_FirstLayer, ADJ_SecondLayer, BiasCorrectionLayer, GeneralizedMeanPooling, GCN_Layer, MetaGraph_fd
from DSBN.dsbn import DomainSpecificBatchNorm1d

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        if classname.find('Linear') != -1:
            init.normal_(m.weight.data, 0, 0.001)
            if m.bias:
                init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


class Embed_Module(nn.Module):
    def __init__(self, arch, feature_in=3, pretrained=True):
        super(Embed_Module, self).__init__()
        model = resnet50(pretrained=pretrained, last_conv_stride=1, last_conv_dilation=1, feature_in=feature_in)
        if arch == 'visible' or arch == 'infrared':
            self.layer1 = copy.deepcopy(model.conv1)
            self.layer2 = copy.deepcopy(model.bn1)
            self.layer3 = copy.deepcopy(model.relu)
            self.layer4 = copy.deepcopy(model.maxpool)
        elif arch == 'base':
            self.layer1 = copy.deepcopy(model.layer1)
            self.layer2 = copy.deepcopy(model.layer2)
            self.layer3 = copy.deepcopy(model.layer3)
            self.layer4 = copy.deepcopy(model.layer4)
        del model

    def forward(self, x):
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class ResNet50(nn.Module):
    def __init__(self, out_features, device):
        super(ResNet50, self).__init__()

        self.visible_module = Embed_Module('visible')
        self.infrared_module = Embed_Module('infrared')
        self.base_module = Embed_Module('base')
        self.in_features = 2048
        self.out_features = out_features

        self.l2norm = Normalize(2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.BatchNorm1d(self.in_features)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = Incremental_Classifier(self.in_features, out_features, device)

    def increase_step(self, increment_class):
        self.classifier.increase_step(self.classifier.fc, increment_class)
        self.out_features += increment_class

    def frozen_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.infrared_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.infrared_module(x2)
        del x1, x2

        x = self.base_module(x)
        x = self.pool(x)
        x = x.view(x.size(0), x.size(1))
        feat = self.bottleneck(x)
        if self.training:
            return x, self.classifier(feat)
        else:
            return self.l2norm(x), self.l2norm(feat)


class CAJ_ResNet50(nn.Module):
    def __init__(self, out_features, device, method=''):
        super(CAJ_ResNet50, self).__init__()

        self.visible_module = Embed_Module('visible')
        self.infrared_module = Embed_Module('infrared')
        self.base_module = Embed_Module('base')
        self.method = method
        self.in_features = 2048
        self.out_features = out_features

        self.l2norm = Normalize(2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.BatchNorm1d(self.in_features)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = Incremental_Classifier(self.in_features, out_features, device)
        self.domain_bns = nn.BatchNorm1d(self.in_features)

        if method == 'bic':
            self.bias_split = [0]
            self.bias_correct_list = nn.ModuleList()

    def increase_step(self, increment_class):
        self.classifier.increase_step(self.classifier.fc, increment_class)
        self.out_features += increment_class

    def bic_increase_step(self):
        self.bias_split.append(self.out_features)
        self.bias_correct_list.append(BiasCorrectionLayer())

    def bias_forward(self, x, step):
        x[:, self.bias_split[step]:self.bias_split[step + 1]] = self.bias_correct_list[step](x[:, self.bias_split[step]:self.bias_split[step + 1]])
        return x

    def bias_layer_frozen(self, step):
        self.bias_correct_list[step].frozen_all()

    def frozen_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x1, x2, modal=0, pcb='f', reply='f', step=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.infrared_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.infrared_module(x2)
        del x1, x2

        x = self.base_module(x)
        if pcb == 't':
            x41, x42, x43, x44 = torch.chunk(x, 4, 2)
            x41, x42, x43, x44 = self.pool(x41), self.pool(x42), self.pool(x43), self.pool(x44)
            x41, x42, x43, x44 = x41.view(x41.size(0), -1), x42.view(x42.size(0), -1), x43.view(x43.size(0), -1), x44.view(x44.size(0), -1)
            x41, x42, x43, x44 = self.bottleneck(x41), self.bottleneck(x42), self.bottleneck(x43), self.bottleneck(x44)
            x41, x42, x43, x44 = x41.unsqueeze(2), x42.unsqueeze(2), x43.unsqueeze(2), x44.unsqueeze(2)
            x = torch.cat((x41, x42, x43, x44), 2)
            return x
        x = self.pool(x)
        x = x.view(x.size(0), x.size(1))
        feat = self.bottleneck(x)
        if self.training:
            return x, self.classifier(feat)
        else:
            return self.l2norm(x), self.l2norm(feat)


class CAJ_GCN_ResNet50(nn.Module):
    def __init__(self, out_features, device=None, rgb_cams=1, ir_cams=2):
        super(CAJ_GCN_ResNet50, self).__init__()

        self.visible_module = Embed_Module('visible')
        self.infrared_module = Embed_Module('infrared')
        self.base_module = Embed_Module('base')
        self.in_features = 2048
        self.out_features = out_features

        self.l2norm = Normalize(2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = BatchNorm1d(self.in_features)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = Incremental_Classifier(self.in_features, out_features)
        self.domain_bns = BatchNorm1d(self.in_features)

        self.gcn_layer = GCN_Layer(self.in_features, self.out_features, modal_nodes=2, rgb_cam=rgb_cams, ir_cam=ir_cams)

    def increase_step(self, increment_class, rgb_cams=None, ir_cams=None):
        self.classifier.increase_step(self.classifier.fc, increment_class)
        self.out_features += increment_class
        self.gcn_layer.increase_step(increment_class, rgb_cams, ir_cams)

    def frozen_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x1, x2, modal=0, step=0, cams=None, pcb='f', reply='f'):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.infrared_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.infrared_module(x2)
        del x1, x2

        x = self.base_module(x)
        if pcb == 't':
            x41, x42, x43, x44 = torch.chunk(x, 4, 2)
            x41, x42, x43, x44 = self.pool(x41), self.pool(x42), self.pool(x43), self.pool(x44)
            x41, x42, x43, x44 = x41.view(x41.size(0), -1), x42.view(x42.size(0), -1), x43.view(x43.size(0), -1), x44.view(x44.size(0), -1)
            x41, x42, x43, x44 = self.bottleneck(x41), self.bottleneck(x42), self.bottleneck(x43), self.bottleneck(x44)
            x41, x42, x43, x44 = x41.unsqueeze(2), x42.unsqueeze(2), x43.unsqueeze(2), x44.unsqueeze(2)
            x = torch.cat((x41, x42, x43, x44), 2)
            return x
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        feat = self.bottleneck(x)
        if reply == 'f':
            x_gcn = x
            x_gcn = self.gcn_layer(x_gcn, modal, step, cams)
        else:
            x_gcn = x

        if self.training:
            return x, self.classifier(feat), x_gcn
        else:
            return self.l2norm(x), self.l2norm(feat), self.l2norm(x_gcn)


class PTKP_ResNet50(nn.Module):
    def __init__(self, out_features, device):
        super(PTKP_ResNet50, self).__init__()

        self.visible_module = Embed_Module('visible')
        self.infrared_module = Embed_Module('infrared')
        self.base_module = Embed_Module('base')
        self.in_features = 2048
        self.out_features = out_features

        self.l2norm = Normalize(2)
        self.pool = GeneralizedMeanPooling(3)
        self.bottleneck = nn.BatchNorm1d(self.in_features)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = Incremental_Classifier(self.in_features, out_features, device)
        self.domain_bns = DomainSpecificBatchNorm1d(self.in_features, bias=False)

    def increase_step(self, increment_class):
        self.classifier.increase_step(self.classifier.fc, increment_class)
        self.domain_bns.increase_step()
        self.out_features += increment_class

    def frozen_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def domain_specific_feat_list(self, x, reply):
        feat_list = []
        if reply is True:
            for i in range(len(self.domain_bns)):
                self.domain_bns.bns[i].eval()
                feat_list.append(self.domain_bns(x, i))
        else:
            for i in range(len(self.domain_bns) - 1):
                self.domain_bns.bns[i].eval()
                feat_list.append(self.domain_bns(x, i))
                self.domain_bns.bns[i].train()
            self.domain_bns.bns[-1].train()
            feat_list.append(self.domain_bns(x, len(self.domain_bns) - 1))
        return feat_list

    def forward(self, x1, x2, reply=False, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.infrared_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.infrared_module(x2)
        del x1, x2

        x = self.base_module(x)
        x = self.pool(x)
        x = x.view(x.size(0), x.size(1))
        feat = self.bottleneck(x)
        if self.training:
            cls = self.classifier(feat)
            feat_list = self.domain_specific_feat_list(x, reply)
            return x, cls, feat, feat_list
        else:
            return self.l2norm(x), self.l2norm(feat)


class AKA_ResNet50(nn.Module):
    def __init__(self, out_features, device):
        super(AKA_ResNet50, self).__init__()

        self.visible_module = Embed_Module('visible')
        self.infrared_module = Embed_Module('infrared')
        self.base_module = Embed_Module('base')
        self.in_features = 2048
        self.out_features = out_features

        self.l2norm = Normalize(2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.BatchNorm1d(self.in_features)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = Incremental_Classifier(self.in_features, out_features, device)

        self.gcn_layer = MetaGraph_fd(hidden_dim=2048, input_dim=2048, sigma=2.0, proto_graph_vertex_num=16, meta_graph_vertex_num=64)

    def increase_step(self, increment_class):
        self.classifier.increase_step(self.classifier.fc, increment_class)
        self.out_features += increment_class

    def frozen_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x1, x2, modal=0, reply='f', step=0, cams=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.infrared_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.infrared_module(x2)
        del x1, x2

        x = self.base_module(x)
        x = self.pool(x)
        x = x.view(x.size(0), x.size(1))
        feature_fuse = None
        correlation = None
        if reply == 'f':
            x_gcn = x.detach()
            protos, correlation = self.gcn_layer(x_gcn)
            feature_fuse = x_gcn + protos

        feat = self.bottleneck(x)
        if self.training:
            return x, self.classifier(feat), feature_fuse, correlation
        else:
            return self.l2norm(x), self.l2norm(feat), self.l2norm(feature_fuse)
