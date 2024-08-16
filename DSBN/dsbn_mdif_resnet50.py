import torch
import torchvision.models as models
from dsbn import DomainSpecificBatchNorm1d as BatchNorm1d
from model_utils import *


class DSBN_MDIF_ResNet50(nn.Module):

    def __init__(self, num_classes=0, pretrained=True, device=None):
        super(DSBN_MDIF_ResNet50, self).__init__()
        self.inplanes = 64
        self.device = device
        self.num_features = 2048

        # init_model
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(self.inplanes, device=self.device)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.reset_parameters()
        if pretrained:
            self.update_pretrained_weight(num_classes=num_classes)
            # self.remember_pretrained_weight = copy.deepcopy(self.state_dict())

        # cross_modal_reid_model
        self.visible_module = Separate_Model(self.conv1, self.bn1, self.relu, self.maxpool)
        self.infrared_module = Separate_Model(self.conv1, self.bn1, self.relu, self.maxpool)
        self.base_module = TwoInputSequential(copy.deepcopy(self.layer1), copy.deepcopy(self.layer2), copy.deepcopy(self.layer3), copy.deepcopy(self.layer4))
        self.bn = BatchNorm1d(self.num_features)
        self.fc = Incremental_Classifier(self.num_features, num_classes, self.device)
        self.l2norm = Normalize(2)
        del self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4

        # MDIF

        self.gcn = GCN_3ADJ(self.num_features, self.num_features, self.num_features)
        self.adj_0 = ADJ_ZeroLayer(device)
        self.adj_1 = ADJ_FirstLayer(device)
        self.adj_2 = ADJ_SecondLayer(device)
        self.gcn_bn = BatchNorm1d(self.num_features)
        self.gcn_classifier = Incremental_Classifier(self.num_features, num_classes, self.device)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BottleneckResnet50.expansion:
            downsample = TwoInputSequential(
                Conv2d(self.inplanes, planes * BottleneckResnet50.expansion, kernel_size=1, stride=stride),
                BatchNorm2d(planes * BottleneckResnet50.expansion, device=self.device))

        layers = [BottleneckResnet50(self.inplanes, planes, stride, downsample, device=self.device)]
        self.inplanes = planes * BottleneckResnet50.expansion
        for _ in range(1, blocks):
            layers.append(BottleneckResnet50(self.inplanes, planes, device=self.device))

        return TwoInputSequential(*layers)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d or BatchNorm1d):
                m.reset_parameters(0)

    # def update_pretrained_weight(self, num_classes=1000, dsbn_type='all'):
    #     pre_trained_dict = models.resnet50(pretrained=True).state_dict()
    #     new_state_dict = copy.deepcopy(pre_trained_dict)
    #
    #     for key, val in pre_trained_dict.items():
    #         if ('bn' in key or 'downsample.1' in key) and dsbn_type == 'all' and ('cfc' not in key):
    #             num_domains = 1
    #             if 'weight' in key:
    #                 for d in range(num_domains):
    #                     new_state_dict[key[0:-len('weight')] + 'bns1.{}.weight'.format(d)] = val.data.clone()
    #                     new_state_dict[key[0:-len('weight')] + 'bns2.{}.weight'.format(d)] = val.data.clone()
    #                     # new_state_dict[key[0:-len('weight')] + 'bns1.{}.cfc'.format(d)] = torch.zeros(val.size()[0], 2, requires_grad=True)
    #             elif 'bias' in key:
    #                 for d in range(num_domains):
    #                     new_state_dict[key[0:-len('bias')] + 'bns1.{}.bias'.format(d)] = val.data.clone()
    #                     new_state_dict[key[0:-len('bias')] + 'bns2.{}.bias'.format(d)] = val.data.clone()
    #             elif 'running_mean' in key:
    #                 for d in range(num_domains):
    #                     new_state_dict[key[0:-len('running_mean')] + 'bns1.{}.running_mean'.format(d)] = val.data.clone()
    #                     new_state_dict[key[0:-len('running_mean')] + 'bns2.{}.running_mean'.format(d)] = val.data.clone()
    #             elif 'running_var' in key:
    #                 for d in range(num_domains):
    #                     new_state_dict[key[0:-len('running_var')] + 'bns1.{}.running_var'.format(d)] = val.data.clone()
    #                     new_state_dict[key[0:-len('running_var')] + 'bns2.{}.running_var'.format(d)] = val.data.clone()
    #             elif 'num_batches_tracked' in key:
    #                 for d in range(num_domains):
    #                     new_state_dict[key[0:-len('num_batches_tracked')] + 'bns1.{}.num_batches_tracked'.format(
    #                         d)] = val.data.clone()
    #                     new_state_dict[key[0:-len('num_batches_tracked')] + 'bns2.{}.num_batches_tracked'.format(
    #                         d)] = val.data.clone()
    #             del new_state_dict[key]
    #     for key in list(new_state_dict.keys()):
    #         if 'fc' in key and 'cfc' not in key:
    #             del new_state_dict[key]
    #     self.load_state_dict(new_state_dict)

    def update_pretrained_weight(self, num_classes=1000, dsbn_type='all'):
        pre_trained_dict = models.resnet50(pretrained=True).state_dict()
        new_state_dict = copy.deepcopy(pre_trained_dict)

        for key, val in pre_trained_dict.items():
            if ('bn' in key or 'downsample.1' in key) and dsbn_type == 'all' and ('cfc' not in key):
                num_domains = 1
                if 'weight' in key:
                    for d in range(num_domains):
                        new_state_dict[key[0:-len('weight')] + 'bns.{}.weight'.format(d)] = val.data.clone()
                        # fc = nn.Linear(val.size()[0], val.size()[0] // 16, bias=False)
                        # new_state_dict[key[0:-len('weight')] + 'bns.{}.attention1.weight'.format(d)] = fc.weight.data.clone()
                        # fc = nn.Linear(val.size()[0] // 16, val.size()[0], bias=False)
                        # new_state_dict[key[0:-len('weight')] + 'bns.{}.attention2.weight'.format(d)] = fc.weight.data.clone()

                        # new_state_dict[key[0:-len('weight')] + 'bns.{}.cfc'.format(d)] = torch.zeros(val.size()[0], 2, requires_grad=True)
                        # conv = nn.Conv2d(val.size()[0], val.size()[0], kernel_size=1, stride=1, padding=0, bias=False)
                        # new_state_dict[key[0:-len('weight')] + 'bns.{}.Qconv.weight'.format(d)] = conv.weight.data.clone()
                        # conv = nn.Conv2d(val.size()[0], val.size()[0], kernel_size=1, stride=1, padding=0, bias=False)
                        # new_state_dict[key[0:-len('weight')] + 'bns.{}.Kconv.weight'.format(d)] = conv.weight.data.clone()
                        # conv = nn.Conv2d(val.size()[0], val.size()[0], kernel_size=1, stride=1, padding=0, bias=False)
                        # new_state_dict[key[0:-len('weight')] + 'bns.{}.Vconv.weight'.format(d)] = conv.weight.data.clone()
                elif 'bias' in key:
                    for d in range(num_domains):
                        new_state_dict[key[0:-len('bias')] + 'bns.{}.bias'.format(d)] = val.data.clone()
                elif 'running_mean' in key:
                    for d in range(num_domains):
                        new_state_dict[key[0:-len('running_mean')] + 'bns.{}.running_mean'.format(d)] = val.data.clone()
                elif 'running_var' in key:
                    for d in range(num_domains):
                        new_state_dict[key[0:-len('running_var')] + 'bns.{}.running_var'.format(d)] = val.data.clone()
                elif 'num_batches_tracked' in key:
                    for d in range(num_domains):
                        new_state_dict[key[0:-len('num_batches_tracked')] + 'bns.{}.num_batches_tracked'.format(
                            d)] = val.data.clone()
                del new_state_dict[key]
        for key in list(new_state_dict.keys()):
            if 'fc' in key and 'cfc' not in key and 'conv' not in key:
                del new_state_dict[key]
        self.load_state_dict(new_state_dict)

    def increase_step(self, old_classifier, incremental_class):
        for m in self.modules():
            if isinstance(m, BatchNorm2d or BatchNorm1d):
                m.increase_step()

        self.bn.increase_step()
        self.fc.increase_step(old_classifier, incremental_class)
        self.adj_0.increase_step()
        self.adj_1.increase_step()
        self.adj_2.increase_step()
        self.gcn_bn.increase_step()
        self.gcn_classifier.increase_step(old_classifier, incremental_class)

    def frozen_all(self):
        for param in self.parameters():
            param.requires_grad = False
        self.to(self.device)
        self.train()

    def weight_align(self, increment_class):
        self.fc.weight_align(increment_class)

    def forward(self, x1, x2, step, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1, step)
            x2 = self.infrared_module(x2, step)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1, step)
        elif modal == 2:
            x = self.infrared_module(x2, step)
        del x1, x2

        x, _ = self.base_module(x, step)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x_bn, _ = self.bn(x, step)
        x_prob = self.fc(x_bn)

        # MDIF
        adj_0 = self.adj_0(x)
        adj_1 = self.adj_1(x, step)
        x_new, adj_2 = self.adj_2(x, step, modal=modal)
        adj_0 = adj_0.detach()
        adj_1 = adj_1.detach()
        adj_2 = adj_2.detach()

        gcn_feat = self.gcn(x_new, adj_0, adj_1, adj_2)
        x_gcn = x + gcn_feat[:x.size()[0]]
        x_gcn_bn, _ = self.gcn_bn(x_gcn, step)
        x_gcn_prob = self.gcn_classifier(x_gcn_bn)

        if self.training:
            return x, x_prob, x_gcn, x_gcn_prob
        else:
            return self.l2norm(x), self.l2norm(x_bn), self.l2norm(x_gcn_bn)


# test
# device = torch.device('cuda:0')
# F = torch.randn(16, 3, 224, 224).to(device)
# print("As begin,shape:", format(F.shape))
# resnet = DSBN_MDIF_ResNet50(num_classes=651, pretrained=True, device=device)
# resnet.to(device)
# resnet.train()
# F, score, score2 = resnet(F, F, 0, 0)
# print(F.shape)
# print(score.shape)
# old_model = copy.deepcopy(resnet)
# old_model.frozen_all()
# old_model.to(device)
# old_model.train()
# resnet.increase_step(old_model.fc.classifier, 10)
# resnet.to(device)
# F = torch.randn(16, 3, 224, 224).to(device)
# F, score, score2 = resnet(F, F, 1, 0)
# print(F.shape)
# print(score.shape)
# resnet.weight_align(10)
# F = torch.randn(16, 3, 224, 224).to(device)
# F, score, score2 = resnet(F, F, 0, 0)
# print(F.shape)
# print(score.shape)
