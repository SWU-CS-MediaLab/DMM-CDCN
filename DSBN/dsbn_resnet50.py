import torchvision.models as models
from dsbn import DomainSpecificBatchNorm1d as BatchNorm1d
from model_utils import *


class DSBN_ResNet50(nn.Module):

    def __init__(self,  num_classes=1000, pretrained=True, device=None):
        super(DSBN_ResNet50, self).__init__()

        self.inplanes = 64
        self.device = device
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

        # rea_model
        self.visible_module = None
        self.infrared_module = None
        self.base_module = None
        self.separate_model()
        self.bn = BatchNorm1d(2048)
        self.fc = Incremental_Classifier(2048, num_classes, self.device)
        self.l2norm = Normalize(2)

    def separate_model(self):
        self.visible_module = Separate_Model(self.conv1, self.bn1, self.relu, self.maxpool)
        self.infrared_module = Separate_Model(self.conv1, self.bn1, self.relu, self.maxpool)
        self.base_module = TwoInputSequential(
            copy.deepcopy(self.layer1),
            copy.deepcopy(self.layer2),
            copy.deepcopy(self.layer3),
            copy.deepcopy(self.layer4))
        del self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4

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
                m.reset_parameters()

    def update_pretrained_weight(self, num_classes=1000, dsbn_type='all'):
        pre_trained_dict = models.resnet50(pretrained=True).state_dict()
        new_state_dict = copy.deepcopy(pre_trained_dict)

        for key, val in pre_trained_dict.items():
            if ('bn' in key or 'downsample.1' in key) and dsbn_type == 'all':
                num_domains = 1
                if 'weight' in key:
                    for d in range(num_domains):
                        new_state_dict[key[0:-len('weight')] + 'bns.{}.weight'.format(d)] = val.data.clone()

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
                        new_state_dict[key[0:-len('num_batches_tracked')] + 'bns.{}.num_batches_tracked'.format(d)] = val.data.clone()
                del new_state_dict[key]

        for key in list(new_state_dict.keys()):
            if 'fc' in key:
                del new_state_dict[key]
        self.load_state_dict(new_state_dict)

    def increase_step(self, old_classifier, incremental_class):
        for m in self.modules():
            if isinstance(m, BatchNorm2d or BatchNorm1d):
                m.increase_step()
        self.bn.increase_step()
        self.fc.increase_step(old_classifier, incremental_class)

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
        x, _ = self.base_module(x, step)

        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        feat, _ = self.bn(x, step)
        score = self.fc(feat)
        if self.training:
            return x, score
        else:
            return self.l2norm(x), self.l2norm(feat)


# test
# F = torch.randn(16, 3, 224, 224)
# print("As begin,shape:", format(F.shape))
# resnet = ResNet50(pretrained=True)
# F, score = resnet(F, F, 0, 0)
# print(F.shape)
# print(score.shape)
# old_model = copy.deepcopy(resnet)
# old_model.frozen_all()
# resnet.increase_step(old_model.fc.classifier, 10)
# F = torch.randn(16, 3, 224, 224)
# F, score = resnet(F, F, 1, 0)
# print(F.shape)
# print(score.shape)
# resnet.weight_align(10)
# F = torch.randn(16, 3, 224, 224)
# F, score = resnet(F, F, 0, 0)
# print(F.shape)
# print(score.shape)

