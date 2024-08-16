import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.batchnorm import _BatchNorm


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        if m.bias is not None:
            init.zeros_(m.bias.data)


class DomainSpecificBatchNorm1d(nn.Module):
    def __init__(self, num_channel, eps=1e-9, momentum=0.1, affine=True, track_running_stats=True, bias=True):
        super(DomainSpecificBatchNorm1d, self).__init__()
        self.num_channel = num_channel
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.bias_requires_grad = bias
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d(num_channel, eps, momentum, affine, track_running_stats)
        self.bns.append(bn)
        self.reset_parameters(0)
        if track_running_stats is True:
            self.reset_running_stats(0)

    def __len__(self):
        return len(self.bns)

    def reset_running_stats(self, step):
        self.bns[step].reset_running_stats()

    def requires_grad(self, flag, step):
        self.bns[step].requires_grad_(flag)
        self.bns[step].bias.requires_grad_(self.bias_requires_grad)

    def reset_parameters(self, step):
        self.bns[step].apply(weights_init_kaiming)
        self.requires_grad(True, step)

    def frozen_bn(self, step):
        self.requires_grad(False, step)

    # def weight_clone(self, step1, step2):
    #     self.bns[step2].weight.data = self.bns[step1].weight.data.clone()
    #     if self.bns[step1].bias is not None:
    #         self.bns[step2].bias.data = self.bns[step1].bias.data.clone()
    #     if self.bns[step1].track_running_stats is True and self.bns[step2].track_running_stats is True:
    #         self.bns[step2].running_mean.data = self.bns[step1].running_mean.data.clone()
    #         self.bns[step2].running_var.data = self.bns[step1].running_var.data.clone()
    #         self.bns[step2].num_batches_tracked.data = self.bns[step1].num_batches_tracked.data.clone()
    def weight_clone(self, clone_bn):
        for bn in self.bns:
            bn.weight.data = clone_bn.weight.data.clone()

    @staticmethod
    def check_input_dim(input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'.format(input.dim()))

    def increase_step(self):
        self.frozen_bn(len(self.bns) - 1)
        bn = nn.BatchNorm1d(self.num_channel, self.eps, self.momentum, self.affine, self.track_running_stats)
        self.bns.append(bn)
        self.reset_parameters(len(self.bns) - 1)

    def forward(self, x, step):
        self.check_input_dim(x)
        x = self.bns[step](x)

        return x


class DomainSpecificBatchNorm2d(nn.Module):

    def __init__(self, num_channel, eps=1e-9, momentum=0.1, affine=True, track_running_stats=True, device=None, bias=False):
        super(DomainSpecificBatchNorm2d, self).__init__()
        self.num_channel = num_channel
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm2d(num_channel, eps, momentum, affine, track_running_stats))
        self.reset_running_stats(0)
        self.reset_parameters(0)

    def reset_running_stats(self, step):
        self.bns[step].reset_running_stats()

    def requires_grad(self, flag, step):
        self.bns[step].requires_grad_(flag)
        self.bns[step].bias.requires_grad_(flag)

    def reset_parameters(self, step):
        self.bns[step].apply(weights_init_kaiming)
        self.requires_grad(True, step)

    def frozen_bn(self, step):
        self.requires_grad(False, step)

    def weight_clone(self, step1, step2):
        self.bns[step2].weight.data = self.bns[step1].weight.data.clone()
        if self.bns[step1].bias is not None:
            self.bns[step2].bias.data = self.bns[step1].bias.data.clone()
        self.bns[step2].running_mean.data = self.bns[step1].running_mean.data.clone()
        self.bns[step2].running_var.data = self.bns[step1].running_var.data.clone()
        self.bns[step2].num_batches_tracked.data = self.bns[step1].num_batches_tracked.data.clone()

    @staticmethod
    def check_input_dim(input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def increase_step(self):
        self.frozen_bn(len(self.bns) - 1)
        self.bns.append(nn.BatchNorm2d(self.num_channel, self.eps, self.momentum, self.affine, self.track_running_stats))
        self.reset_parameters(len(self.bns) - 1)
        self.weight_clone(len(self.bns) - 2, len(self.bns) - 1)

    def forward(self, x, step):
        self.check_input_dim(x)
        x = self.bns[step](x)
        return x, step


class RcBatchNorm2d(_BatchNorm):
    def __init__(self, num_features, eps=1e-9, momentum=0.1, affine=True, track_running_stats=True):
        super(RcBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

        # self.cfc = Parameter(torch.Tensor(num_features, 2))
        # self.cfc.data.fill_(0)
        self.activation = nn.Sigmoid()
        # self.softmax = nn.Sigmoid()
        # self.Qconv = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, padding=0, bias=False)
        # self.Kconv = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, padding=0, bias=False)
        # self.Vconv = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.attention1 = nn.Linear(num_features, num_features // 16, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.attention2 = nn.Linear(num_features // 16, num_features, bias=False)

    def recalibration(self, x, x_bn, eps=1e-9):

        # N, C, _, _ = x.size()
        # channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        # channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        # channel_std = channel_var.sqrt()
        # t = torch.cat((channel_mean, channel_std), dim=2)
        # z = t * self.cfc[None, :, :]  # B x C x 2
        # z = torch.sum(z, dim=2)[:, :, None, None]  # B x C x 1 x 1

        N, C, w, h = x.size()
        channle_weight = self.avgpool(x).view(N, C)
        channle_weight = self.attention1(channle_weight)
        channle_weight = self.relu(channle_weight)
        channle_weight = self.attention2(channle_weight)
        channle_weight = self.activation(channle_weight).view(N, C, 1, 1)
        out = x_bn * channle_weight.expand_as(x_bn)
        # Q = self.Qconv(x).view(N, -1, w*h).permute(0, 2, 1)
        # K = self.Kconv(x).view(N, -1, w*h)
        # V = self.Vconv(x_bn).view(N, -1, w*h)
        # attention = self.softmax(torch.bmm(K, Q))
        # out = torch.bmm(attention.permute(0, 2, 1), V).view(N, C, w, h)
        return out

    def forward(self, input, epochs=-1):

        out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)

        out = self.recalibration(input, out_bn)
        # out = out_bn
        # out = out_bn * g
        return out


class DomainSpecificRcBatchNorm2d(nn.Module):

    def __init__(self, num_channel, eps=1e-9, momentum=0.1, affine=True, track_running_stats=True, device=None, bias=False):
        super(DomainSpecificRcBatchNorm2d, self).__init__()
        self.num_channel = num_channel
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.bns = nn.ModuleList()
        self.bns.append(RcBatchNorm2d(num_channel, eps, momentum, affine, track_running_stats))
        self.reset_running_stats(0)
        self.reset_parameters(0)

    def reset_running_stats(self, step):
        self.bns[step].reset_running_stats()

    def requires_grad(self, flag, step):
        self.bns[step].requires_grad_(flag)
        self.bns[step].bias.requires_grad_(flag)

    def reset_parameters(self, step):
        self.bns[step].apply(weights_init_kaiming)
        self.requires_grad(True, step)

    def frozen_bn(self, step):
        self.requires_grad(False, step)

    def weight_clone(self, step1, step2):
        self.bns[step2].weight.data = self.bns[step1].weight.data.clone()
        if self.bns[step1].bias is not None:
            self.bns[step2].bias.data = self.bns[step1].bias.data.clone()
        self.bns[step2].running_mean.data = self.bns[step1].running_mean.data.clone()
        self.bns[step2].running_var.data = self.bns[step1].running_var.data.clone()
        self.bns[step2].num_batches_tracked.data = self.bns[step1].num_batches_tracked.data.clone()

    @staticmethod
    def check_input_dim(input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def increase_step(self):
        self.frozen_bn(len(self.bns) - 1)
        self.bns.append(nn.BatchNorm2d(self.num_channel, self.eps, self.momentum, self.affine, self.track_running_stats))
        self.reset_parameters(len(self.bns) - 1)
        self.weight_clone(len(self.bns) - 2, len(self.bns) - 1)

    def forward(self, x, step):
        self.check_input_dim(x)
        x = self.bns[step](x)
        return x, step


class DomainModalSpecificRcBatchNorm2d(nn.Module):

    def __init__(self, num_channel, eps=1e-9, momentum=0.1, affine=True, track_running_stats=True, device=None, bias=False):
        super(DomainModalSpecificRcBatchNorm2d, self).__init__()
        self.num_channel = num_channel
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.bns1 = nn.ModuleList()
        self.bns1.append(RcBatchNorm2d(num_channel, eps, momentum, affine, track_running_stats))
        self.bns2 = nn.ModuleList()
        self.bns2.append(RcBatchNorm2d(num_channel, eps, momentum, affine, track_running_stats))
        self.reset_running_stats(0)
        self.reset_parameters(0)

    def reset_running_stats(self, step):
        self.bns1[step].reset_running_stats()
        self.bns2[step].reset_running_stats()

    def requires_grad(self, flag, step):
        self.bns1[step].requires_grad_(flag)
        self.bns1[step].bias.requires_grad_(flag)
        self.bns2[step].requires_grad_(flag)
        self.bns2[step].bias.requires_grad_(flag)

    def reset_parameters(self, step):
        self.bns1[step].apply(weights_init_kaiming)
        self.bns2[step].apply(weights_init_kaiming)
        self.requires_grad(True, step)

    def frozen_bn(self, step):
        self.requires_grad(False, step)

    def weight_clone(self, step1, step2):
        self.bns1[step2].weight.data = self.bns1[step1].weight.data.clone()
        if self.bns1[step1].bias is not None:
            self.bns1[step2].bias.data = self.bns1[step1].bias.data.clone()
        self.bns1[step2].running_mean.data = self.bns1[step1].running_mean.data.clone()
        self.bns1[step2].running_var.data = self.bns1[step1].running_var.data.clone()
        self.bns1[step2].num_batches_tracked.data = self.bns1[step1].num_batches_tracked.data.clone()

        self.bns2[step2].weight.data = self.bns2[step1].weight.data.clone()
        if self.bns2[step1].bias is not None:
            self.bns2[step2].bias.data = self.bns2[step1].bias.data.clone()
        self.bns2[step2].running_mean.data = self.bns2[step1].running_mean.data.clone()
        self.bns2[step2].running_var.data = self.bns2[step1].running_var.data.clone()
        self.bns2[step2].num_batches_tracked.data = self.bns2[step1].num_batches_tracked.data.clone()

    @staticmethod
    def check_input_dim(input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def increase_step(self):
        self.frozen_bn(len(self.bns1) - 1)
        self.bns1.append(RcBatchNorm2d(self.num_channel, self.eps, self.momentum, self.affine, self.track_running_stats))
        self.bns2.append(RcBatchNorm2d(self.num_channel, self.eps, self.momentum, self.affine, self.track_running_stats))
        self.reset_parameters(len(self.bns1) - 1)
        self.weight_clone(len(self.bns1) - 2, len(self.bns) - 1)

    def forward(self, x, step):
        self.check_input_dim(x)
        b, c, w, h = x.size()
        x = torch.cat((self.bns1[step](x[:b//2]), self.bns2[step](x[b//2:])), 0)
        return x, step


class DomainModalSpecificBatchNorm2d(nn.Module):

    def __init__(self, num_channel, eps=1e-9, momentum=0.1, affine=True, track_running_stats=True, device=None, bias=False):
        super(DomainModalSpecificBatchNorm2d, self).__init__()
        self.num_channel = num_channel
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.bns1 = nn.ModuleList()
        self.bns1.append(nn.BatchNorm2d(num_channel, eps, momentum, affine, track_running_stats))
        self.bns2 = nn.ModuleList()
        self.bns2.append(nn.BatchNorm2d(num_channel, eps, momentum, affine, track_running_stats))
        self.reset_running_stats(0)
        self.reset_parameters(0)

    def reset_running_stats(self, step):
        self.bns1[step].reset_running_stats()
        self.bns2[step].reset_running_stats()

    def requires_grad(self, flag, step):
        self.bns1[step].requires_grad_(flag)
        self.bns1[step].bias.requires_grad_(flag)
        self.bns2[step].requires_grad_(flag)
        self.bns2[step].bias.requires_grad_(flag)

    def reset_parameters(self, step):
        self.bns1[step].apply(weights_init_kaiming)
        self.bns2[step].apply(weights_init_kaiming)
        self.requires_grad(True, step)

    def frozen_bn(self, step):
        self.requires_grad(False, step)

    def weight_clone(self, step1, step2):
        self.bns1[step2].weight.data = self.bns1[step1].weight.data.clone()
        if self.bns1[step1].bias is not None:
            self.bns1[step2].bias.data = self.bns1[step1].bias.data.clone()
        self.bns1[step2].running_mean.data = self.bns1[step1].running_mean.data.clone()
        self.bns1[step2].running_var.data = self.bns1[step1].running_var.data.clone()
        self.bns1[step2].num_batches_tracked.data = self.bns1[step1].num_batches_tracked.data.clone()

        self.bns2[step2].weight.data = self.bns2[step1].weight.data.clone()
        if self.bns2[step1].bias is not None:
            self.bns2[step2].bias.data = self.bns2[step1].bias.data.clone()
        self.bns2[step2].running_mean.data = self.bns2[step1].running_mean.data.clone()
        self.bns2[step2].running_var.data = self.bns2[step1].running_var.data.clone()
        self.bns2[step2].num_batches_tracked.data = self.bns2[step1].num_batches_tracked.data.clone()

    @staticmethod
    def check_input_dim(input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def increase_step(self):
        self.frozen_bn(len(self.bns1) - 1)
        self.bns1.append(nn.BatchNorm2d(self.num_channel, self.eps, self.momentum, self.affine, self.track_running_stats))
        self.bns2.append(nn.BatchNorm2d(self.num_channel, self.eps, self.momentum, self.affine, self.track_running_stats))
        self.reset_parameters(len(self.bns1) - 1)
        self.weight_clone(len(self.bns1) - 2, len(self.bns) - 1)

    def forward(self, x, step):
        self.check_input_dim(x)
        b, c, w, h = x.size()
        x = torch.cat((self.bns1[step](x[:b//2]), self.bns2[step](x[b//2:])), 0)
        return x, step


class DomainModalSpecificBatchNorm1d(nn.Module):

    def __init__(self, num_channel, eps=1e-9, momentum=0.1, affine=True, track_running_stats=True, device=None, bias=False):
        super(DomainModalSpecificBatchNorm1d, self).__init__()
        self.num_channel = num_channel
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.device = device
        self.bns1 = nn.ModuleList()
        self.bns1.append(nn.BatchNorm1d(num_channel, eps, momentum, affine, track_running_stats))
        self.bns2 = nn.ModuleList()
        self.bns2.append(nn.BatchNorm1d(num_channel, eps, momentum, affine, track_running_stats))
        self.reset_running_stats(0)
        self.reset_parameters(0)

    def reset_running_stats(self, step):
        self.bns1[step].reset_running_stats()
        self.bns2[step].reset_running_stats()

    def requires_grad(self, flag, step):
        self.bns1[step].requires_grad_(flag)
        self.bns1[step].bias.requires_grad_(flag)
        self.bns2[step].requires_grad_(flag)
        self.bns2[step].bias.requires_grad_(flag)

    def reset_parameters(self, step):
        self.bns1[step].apply(weights_init_kaiming)
        self.bns2[step].apply(weights_init_kaiming)
        self.requires_grad(True, step)

    def frozen_bn(self, step):
        self.requires_grad(False, step)

    def weight_clone(self, step1, step2):
        self.bns1[step2].weight.data = self.bns1[step1].weight.data.clone()
        if self.bns1[step1].bias is not None:
            self.bns1[step2].bias.data = self.bns1[step1].bias.data.clone()
        self.bns1[step2].running_mean.data = self.bns1[step1].running_mean.data.clone()
        self.bns1[step2].running_var.data = self.bns1[step1].running_var.data.clone()
        self.bns1[step2].num_batches_tracked.data = self.bns1[step1].num_batches_tracked.data.clone()

        self.bns2[step2].weight.data = self.bns2[step1].weight.data.clone()
        if self.bns2[step1].bias is not None:
            self.bns2[step2].bias.data = self.bns2[step1].bias.data.clone()
        self.bns2[step2].running_mean.data = self.bns2[step1].running_mean.data.clone()
        self.bns2[step2].running_var.data = self.bns2[step1].running_var.data.clone()
        self.bns2[step2].num_batches_tracked.data = self.bns2[step1].num_batches_tracked.data.clone()

    @staticmethod
    def check_input_dim(input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'.format(input.dim()))

    def increase_step(self):
        self.frozen_bn(len(self.bns1) - 1)
        self.bns1.append(nn.BatchNorm1d(self.num_channel, self.eps, self.momentum, self.affine, self.track_running_stats))
        self.bns2.append(nn.BatchNorm1d(self.num_channel, self.eps, self.momentum, self.affine, self.track_running_stats))
        self.reset_parameters(len(self.bns1) - 1)
        self.weight_clone(len(self.bns1) - 2, len(self.bns) - 1)

    def forward(self, x, step):
        self.check_input_dim(x)
        b, c = x.size()
        x = torch.cat((self.bns1[step](x[:b//2]), self.bns2[step](x[b//2:])), 0)
        return x, step