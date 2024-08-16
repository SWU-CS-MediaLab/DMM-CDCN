import copy
import math
from itertools import islice
from torch.nn import BatchNorm1d
import numpy as np
import torch
from torch.nn import init, functional as F
from torch.nn import Parameter
import torch.nn as nn
import operator
from collections import OrderedDict, defaultdict


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
        init.zeros_(m.bias.data)


def my_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)


def meta_grad(net, loss, lr):
    require_grad_params = [param for param in net.parameters() if param.requires_grad]
    # print(require_grad_params)
    grads = torch.autograd.grad(loss, require_grad_params, allow_unused=True)
    for param, grad in zip(require_grad_params, grads):
        if grad is not None:
            param.data = param.data - lr * grad.data
    return net


def gen_adj(graph):
    D = torch.pow(graph.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(graph, D).t(), D)
    return adj


class GeneralizedMeanPooling(nn.Module):
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)


class Incremental_MiddleModule(nn.Module):
    def __init__(self):
        super(Incremental_MiddleModule, self).__init__()
        self.now_module = nn.ModuleList()  # 定义一个模块列表
        self.now_module.append(MiddleModule())
        self.length = 1  # self.device = device

    def frozen_step(self, step):
        for p in self.now_module[step].parameters():
            p.requires_grad = False

    def increase_step(self):
        self.frozen_step(self.length - 1)
        self.now_module.append(MiddleModule())
        self.length = len(self.now_module)

    def forward(self, x1, x2, modal, step):
        return self.now_module[step](x1, x2, modal)


class MiddleModule(nn.Module):
    def __init__(self):
        super(MiddleModule, self).__init__()
        self.encoder1 = nn.Conv2d(3, 1, 1)
        self.encoder1.apply(my_weights_init)
        self.fc1 = nn.Conv2d(1, 1, 1)
        self.fc1.apply(weights_init_kaiming)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn1.apply(my_weights_init)

        self.encoder2 = nn.Conv2d(3, 1, 1)
        self.encoder2.apply(my_weights_init)
        self.fc2 = nn.Conv2d(1, 1, 1)
        self.fc2.apply(weights_init_kaiming)
        self.bn2 = nn.BatchNorm2d(1)
        self.bn2.apply(my_weights_init)

        self.decoder = nn.Conv2d(1, 3, 1)
        self.decoder.apply(my_weights_init)

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = F.relu(self.encoder1(x1))
            x1 = self.bn1(F.relu(self.fc1(x1)))
            x2 = F.relu(self.encoder2(x2))
            x2 = self.bn2(F.relu(self.fc2(x2)))
            x = F.relu(self.decoder(torch.cat((x1, x2), 0)))
        elif modal == 1:
            x1 = F.relu(self.encoder1(x1))
            x1 = self.bn1(F.relu(self.fc1(x1)))
            x = F.relu(self.decoder(x1))
        elif modal == 2:
            x2 = F.relu(self.encoder1(x2))
            x2 = self.bn1(F.relu(self.fc1(x2)))
            x = F.relu(self.decoder(x2))
        return x


class Incremental_Classifier(nn.Module):
    def __init__(self, in_features, out_features, device=None):
        super(Incremental_Classifier, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.init_classifier()

    def init_classifier(self):
        self.fc.apply(weights_init_kaiming)
        self.fc.cuda()

    def increase_step(self, old_classifier, incremental_class):
        out_features = old_classifier.out_features
        del self.fc
        self.fc = nn.Linear(old_classifier.in_features, out_features + incremental_class, bias=False)
        self.init_classifier()
        self.fc.weight.data[:out_features] = old_classifier.weight.data.clone()
        if old_classifier.bias is not None:
            self.fc.bias.data[:out_features] = old_classifier.bias.data.clone()
        del old_classifier, out_features

    def weight_align(self, increment):
        weights = self.fc.weight.data
        mean_new = torch.mean(torch.norm(weights[-increment:, :], p=2, dim=1))
        mean_old = torch.mean(torch.norm(weights[:-increment, :], p=2, dim=1))
        gamma = mean_old / mean_new
        self.fc.weight.data[-increment:, :] *= gamma
        del weights, mean_new, mean_old, gamma

    def forward(self, x):
        return self.fc(x)


class TwoInputSequential(nn.Module):
    r"""A sequential container forward with two inputs.
    """

    def __init__(self, *args):
        super(TwoInputSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TwoInputSequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(TwoInputSequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input1, input2):
        for module in self._modules.values():
            input1, input2 = module(input1, input2)
        return input1, input2


class Separate_Model(nn.Module):
    def __init__(self, conv1, bn1, relu, maxpool):
        super(Separate_Model, self).__init__()
        self.conv1 = copy.deepcopy(conv1)
        self.bn1 = copy.deepcopy(bn1)
        self.relu = copy.deepcopy(relu)
        self.maxpool = copy.deepcopy(maxpool)

    def forward(self, x, step):
        x = self.conv1(x)
        x, _ = self.bn1(x, step)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class ADJ_ZeroLayer(nn.Module):
    def __init__(self, modal_nodes=2):
        super(ADJ_ZeroLayer, self).__init__()
        self.step_domain = 0
        self.modal_nodes = modal_nodes

    def increase_step(self):
        self.step_domain += 1

    def forward(self, x, step):
        B, C = x.size()
        adj = torch.eye(B + self.modal_nodes * (self.step_domain + 1) + 1 + self.step_domain).cuda()
        # domain nodes connect with domain nodes
        adj[B + self.modal_nodes:, B + self.modal_nodes:] = 1.0
        # domain nodes connect with modal nodes
        for i in range(self.step_domain + 1):
            adj[B + i, B + self.step_domain * 2 + 1 + 1 + i] = 1.0
            adj[B + self.step_domain * 2 + 1 + 1 + i, B + i] = 1.0
            adj[B + i, B + self.step_domain + 1 + i] = 1.0
            adj[B + self.step_domain + 1 + i, B + i] = 1.0
            adj[B + self.step_domain + 1 + i, B + self.step_domain * 2 + 1 + 1 + i] = 1.0
            adj[B + self.step_domain * 2 + 1 + 1 + i, B + self.step_domain + 1 + i] = 1.0
        return adj


class ADJ_FirstLayer(nn.Module):
    def __init__(self, modal_nodes=2, rgb_cams=1, ir_cams=2):
        super(ADJ_FirstLayer, self).__init__()
        self.step_domain = 0
        self.modal_nodes = modal_nodes
        self.length_cam = 0
        rgb_cams = np.unique(rgb_cams)
        ir_cams = np.unique(ir_cams)
        self.rgb_cams = [rgb_cams]
        self.ir_cams = [ir_cams]
        self.cams_map = defaultdict(int)  # map for cams
        for i in rgb_cams:
            self.cams_map[i] = self.length_cam
            self.length_cam += 1
        for i in ir_cams:
            self.cams_map[i] = self.length_cam
            self.length_cam += 1

    def increase_step(self, rgb_cams, ir_cams):
        rgb_cams = np.unique(rgb_cams)
        ir_cams = np.unique(ir_cams)
        self.step_domain += 1
        self.rgb_cams.append(rgb_cams)
        self.ir_cams.append(ir_cams)
        for i in rgb_cams:
            self.cams_map[i] = self.length_cam
            self.length_cam += 1
        for i in ir_cams:
            self.cams_map[i] = self.length_cam
            self.length_cam += 1

    def forward(self, x, step, modal=0, cams=0):
        B, C = x.size()
        adj = torch.eye(B + self.length_cam).cuda()
        if modal == 0:
            adj[:B // 2, :B // 2] = 1.0
            adj[B // 2: B, B // 2: B] = 1.0
        elif modal == 1 or modal == 2:
            adj[:B, :B] = 1.0
        return adj


class ADJ_SecondLayer(nn.Module):
    def __init__(self, modal_nodes=2, rgb_cams=1, ir_cams=2):
        super(ADJ_SecondLayer, self).__init__()
        self.step_domain = 0
        self.momentum = 0.9
        self.modal_nodes = modal_nodes
        self.attention = nn.ModuleList()
        rgb_cams = np.unique(rgb_cams)
        ir_cams = np.unique(ir_cams)
        self.rgb_cams = [rgb_cams]
        self.ir_cams = [ir_cams]
        self.length_cam = 0
        self.cams_map = defaultdict(int)
        self.running_mean_cam = []
        for i in rgb_cams:
            self.cams_map[i] = self.length_cam
            self.length_cam += 1
        for i in ir_cams:
            self.cams_map[i] = self.length_cam
            self.length_cam += 1
        for i in range(self.length_cam):
            self.attention.append(nn.Linear(2048, 1))
            self.running_mean_cam.append(torch.zeros(1, 2048).cuda())

    def increase_step(self, rgb_cams, ir_cams):
        rgb_cams = np.unique(rgb_cams)
        ir_cams = np.unique(ir_cams)
        self.step_domain += 1
        for i in range(self.length_cam):
            self.attention[i].requires_grad_(False)
        self.rgb_cams.append(rgb_cams)
        self.ir_cams.append(ir_cams)
        incremental_length = 0
        for i in rgb_cams:
            self.cams_map[i] = self.length_cam
            self.length_cam += 1
            incremental_length += 1
        for i in ir_cams:
            self.cams_map[i] = self.length_cam
            self.length_cam += 1
            incremental_length += 1
        for i in range(incremental_length):
            self.attention.append(nn.Linear(2048, 1))
            self.running_mean_cam.append(torch.zeros(1, 2048).cuda())

    def forward(self, x, step, modal=0, cams=0):
        B, C = x.size()
        # adjacent matrix
        adj = torch.eye(B + self.length_cam).cuda()
        rgb_cams = self.rgb_cams[step]
        ir_cams = self.ir_cams[step]
        for i in range(self.length_cam):
            x = torch.cat([x, torch.autograd.Variable(self.running_mean_cam[i])], dim=0)
        if modal == 0:  # rgb + ir
            adj[:B // 2, :B // 2] = 1.0
            adj[B // 2: B, B // 2: B] = 1.0
            for cam in np.unique(cams):
                indices = np.where(cams == cam)[0]
                if cam in rgb_cams:
                    for item in ir_cams:
                        adj[indices, B + self.cams_map[item]] = 1.0
                        adj[B + self.cams_map[item], indices] = 1.0

                elif cam in ir_cams:
                    for item in rgb_cams:
                        adj[indices, B + self.cams_map[item]] = 1.0
                        adj[B + self.cams_map[item], indices] = 1.0

        elif modal == 1:  # rgb
            adj[:B, :B] = 1.0
            for item in ir_cams:
                adj[:B, B + self.cams_map[item]] = 1.0
                adj[B + self.cams_map[item], :B] = 1.0

        elif modal == 2:  # ir
            adj[:B, :B] = 1.0
            for item in rgb_cams:
                adj[:B, B + self.cams_map[item]] = 1.0
                adj[B + self.cams_map[item], :B] = 1.0

        adj = gen_adj(adj)

        if self.training:
            for cam in np.unique(cams):
                indices = np.where(cams == cam)[0]
                weight = self.attention[self.cams_map[cam]](x[indices])
                weight = weight / weight.sum(dim=0, keepdim=False)
                cam_center = torch.sum(x[indices] * weight, dim=0, keepdim=True)
                self.running_mean_cam[self.cams_map[cam]] = self.running_mean_cam[self.cams_map[cam]].mul_(self.momentum)
                self.running_mean_cam[self.cams_map[cam]] = self.running_mean_cam[self.cams_map[cam]].add_((1 - self.momentum) * cam_center[0].data)
                x[self.cams_map[cam] + B] = cam_center
        return x, adj


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        setattr(self.weight, 'gcn_weight', True)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            setattr(self.bias, 'gcn_weight', True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight.float())
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN_3ADJ(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN_3ADJ, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj_0, adj_1, adj_2):
        # x = self.gc1(x, adj_0)
        # x = self.relu(x)
        # x = self.gc2(x, adj_1)
        # x = self.relu(x)
        x = self.gc3(x, adj_2)
        x = self.relu(x)
        return x


class GCN_Layer(nn.Module):
    def __init__(self, in_features, out_features, modal_nodes=2, rgb_cam=1, ir_cam=2):
        super(GCN_Layer, self).__init__()
        self.gcn = GCN_3ADJ(in_features, in_features, in_features)
        # self.adj_0 = ADJ_ZeroLayer(modal_nodes=modal_nodes)
        # self.adj_1 = ADJ_FirstLayer(modal_nodes=modal_nodes)
        self.adj_2 = ADJ_SecondLayer(modal_nodes=modal_nodes)
        self.gcn_bn = BatchNorm1d(in_features)
        self.gcn_classifier = Incremental_Classifier(in_features, out_features)

    def increase_step(self, out_features, rgb_cams, ir_cams):
        self.gcn_classifier.increase_step(self.gcn_classifier.fc, out_features)
        # self.adj_0.increase_step()
        # self.adj_1.increase_step(rgb_cams, ir_cams)
        self.adj_2.increase_step(rgb_cams, ir_cams)

    def adj_increase_step(self, rgb_cams, ir_cams):
        # self.adj_0.increase_step()
        # self.adj_1.increase_step(rgb_cams, ir_cams)
        self.adj_2.increase_step(rgb_cams, ir_cams)

    def forward(self, x, modal=0, step=0, cams=0):
        # adj_0 = self.adj_0(x, step)
        # adj_1 = self.adj_1(x, step, modal=modal)
        x_new, adj_2 = self.adj_2(x, step=step, modal=modal, cams=cams)
        # adj_0 = adj_0.detach()
        # adj_1 = adj_1.detach()
        adj_2 = adj_2.detach()
        # gcn_feat = self.gcn(x_new, adj_0, adj_1, adj_2)
        gcn_feat = self.gcn(x_new, 0, adj_2, adj_2)
        x_gcn = x + gcn_feat[:x.size()[0]]
        x_gcn_bn = self.gcn_bn(x_gcn)

        if self.training:
            return self.gcn_classifier(x_gcn_bn)
        else:
            return x_gcn_bn


class BiasCorrectionLayer(nn.Module):
    def __init__(self):
        super(BiasCorrectionLayer, self).__init__()
        self.linear = nn.Linear(1, 1, bias=True)
        self.linear.weight.data.fill_(1.0)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        correction = self.linear(x.unsqueeze(dim=2))
        correction = correction.squeeze(dim=2)
        return correction

    def frozen_all(self):
        for p in self.parameters():
            p.requires_grad = False


class KMeansPlusPlus(torch.nn.Module):
    def __init__(self, n_clusters: int = 256, max_iter: int = 300):
        super(KMeansPlusPlus, self).__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def init_centroids(self, data) -> None:
        first_center = data[np.random.randint(len(data))]
        self.centroids = first_center.unsqueeze(0).cuda()

        for _ in range(self.n_clusters - 1):
            distances = torch.cdist(data, self.centroids)
            probs = (distances ** 2).sum(dim=1) / distances.sum()
            probs = probs.cpu().numpy()
            probs /= probs.sum()
            new_center_index = np.random.choice(np.arange(len(data)), p=probs)
            new_center = data[new_center_index].unsqueeze(0)
            self.centroids = torch.cat((self.centroids, new_center.cuda()))

    def forward(self, data):
        data = data.cuda()

        if self.centroids is None:
            self.init_centroids(data)

        for _ in range(self.max_iter):
            distances = torch.cdist(data, self.centroids)
            last_centroids = self.centroids.clone()
            assignments = distances.argmin(dim=1)

            for i in range(self.n_clusters):
                cluster_points = data[assignments == i]
                if cluster_points.numel() > 0:
                    self.centroids[i] = cluster_points.mean(dim=0)

            center_changes = torch.norm(self.centroids - last_centroids, dim=1)
            if center_changes.abs().max() < 0.1:
                break

        return assignments


def get_pseudo_features(data_specific_batch_norm, training_phase, x, domain, unchange=False):
    fake_feat_list = []
    if unchange is False:
        for i in range(training_phase):
            if int(domain[0]) == i:
                data_specific_batch_norm[i].train()
                fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])
            else:
                data_specific_batch_norm[i].eval()
                fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])
                data_specific_batch_norm[i].train()
    else:
        for i in range(training_phase):
            data_specific_batch_norm[i].eval()
            fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])

    return fake_feat_list


class GeneralizedMeanPooling(nn.Module):
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.p) + ', ' + 'output_size=' + str(self.output_size) + ')'


# AKA
def Truncated_initializer(m):
    # sample u1:
    size = m.size()
    u1 = torch.rand(size) * (1 - np.exp(-2)) + np.exp(-2)
    # sample u2:
    u2 = torch.rand(size)
    # sample the truncated gaussian ~TN(0,1,[-2,2]):
    z = torch.sqrt(-2 * torch.log(u1)) * torch.cos(2 * np.pi * u2)
    m.data = z


class Graph_Convolution(nn.Module):
    def __init__(self, hidden_dim, sparse_inputs=False, act=nn.Tanh(), bias=True, dropout=0.6):
        super(Graph_Convolution, self).__init__()
        self.active_function = act
        self.dropout_rate = dropout
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.sparse_inputs = sparse_inputs
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.W = nn.Parameter(torch.zeros(size=(hidden_dim, hidden_dim)))
        Truncated_initializer(self.W)
        if self.bias:
            self.b = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.b = None
        self.device = torch.device('cuda')

    def forward(self, inputs, adj):
        x = inputs
        x = self.dropout(x)
        node_size = adj.size(0)
        I = torch.eye(node_size, requires_grad=False).to(self.device)
        adj = adj + I
        D = torch.diag(torch.sum(adj, dim=1, keepdim=False))
        adj = torch.matmul(torch.inverse(D), adj)
        pre_sup = torch.matmul(x, self.W)
        output = torch.matmul(adj, pre_sup)

        if self.bias:
            output += self.b
        if self.active_function is not None:
            return self.active_function(output)
        else:
            return output


class MetaGraph_fd(nn.Module):
    def __init__(self, hidden_dim, input_dim, sigma=2.0, proto_graph_vertex_num=16, meta_graph_vertex_num=128):
        super(MetaGraph_fd, self).__init__()
        self.hidden_dim, self.input_dim, self.sigma = hidden_dim, input_dim, sigma
        adj_mlp = nn.Linear(hidden_dim, 1)
        Truncated_initializer(adj_mlp.weight)
        nn.init.constant_(adj_mlp.bias, 0.1)

        gate_mlp = nn.Linear(hidden_dim, 1)
        Truncated_initializer(gate_mlp.weight)
        nn.init.constant_(gate_mlp.bias, 0.1)

        self.softmax = nn.Softmax(dim=0)
        self.meta_graph_vertex_num = meta_graph_vertex_num
        self.proto_graph_vertex_num = proto_graph_vertex_num
        self.meta_graph_vertex = nn.Parameter(torch.rand(meta_graph_vertex_num, input_dim))
        self.distance = nn.Sequential(adj_mlp, nn.Sigmoid())
        self.gate = nn.Sequential(gate_mlp, nn.Sigmoid())
        self.device = torch.device('cuda')
        self.meta_GCN = Graph_Convolution(self.hidden_dim).to(self.device)
        self.MSE = nn.MSELoss(reduce='mean')
        self.register_buffer('meta_graph_vertex_buffer', torch.rand(self.meta_graph_vertex.size(), requires_grad=False))

    def StabilityLoss(self, old_vertex, new_vertex):
        old_vertex = F.normalize(old_vertex)
        new_vertex = F.normalize(new_vertex)

        # return torch.mean(torch.log(1 + torch.exp(torch.sqrt(torch.sum((old_vertex-new_vertex).pow(2), dim=1, keepdim=False)))))
        return torch.mean(torch.sum((old_vertex - new_vertex).pow(2), dim=1, keepdim=False))

    def forward(self, inputs):
        correlation_meta = self._correlation(self.meta_graph_vertex_buffer, self.meta_graph_vertex.detach())

        self.meta_graph_vertex_buffer = self.meta_graph_vertex.detach()

        batch_size = inputs.size(0)
        protos = inputs
        meta_graph = self._construct_graph(self.meta_graph_vertex, self.meta_graph_vertex).to(self.device)
        proto_graph = self._construct_graph(protos, protos).to(self.device)
        m, n = protos.size(0), self.meta_graph_vertex.size(0)
        xx = torch.pow(protos, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(self.meta_graph_vertex, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(mat1=protos, mat2=self.meta_graph_vertex.t(), beta=1, alpha=-2)
        dist_square = dist.clamp(min=1e-6)
        cross_graph = self.softmax((- dist_square / (2.0 * self.sigma))).to(self.device)
        super_garph = torch.cat((torch.cat((proto_graph, cross_graph), dim=1), torch.cat((cross_graph.t(), meta_graph), dim=1)), dim=0)
        feature = torch.cat((protos, self.meta_graph_vertex), dim=0).to(self.device)
        representation = self.meta_GCN(feature, super_garph)

        # most_similar_index = torch.argmax(cross_graph, dim=1)
        correlation_transfer_meta = self._correlation(representation[batch_size:].detach(), self.meta_graph_vertex.detach())

        correlation_protos = self._correlation(representation[0:batch_size].detach(), protos.detach())

        return representation[0:batch_size].to(self.device), [correlation_meta, correlation_transfer_meta, correlation_protos]

    def _construct_graph(self, A, B):
        m = A.size(0)
        n = B.size(0)
        I = torch.eye(n, requires_grad=False).to(self.device)
        index_aabb = torch.arange(0, m, requires_grad=False).repeat_interleave(n, dim=0).long()
        index_abab = torch.arange(0, n, requires_grad=False).repeat(m).long()
        diff = A[index_aabb] - B[index_abab]
        graph = self.distance(diff).view(m, n)
        graph = graph.to(self.device) * (1 - I) + I
        return graph

    def _correlation(self, A, B):
        similarity = F.cosine_similarity(A, B)
        similarity = torch.mean(similarity)
        return similarity


class FixedMetaGraph(nn.Module):
    def __init__(self, hidden_dim, input_dim, sigma=2.0, proto_graph_vertex_num=16, meta_graph_vertex_num=128):
        super(FixedMetaGraph, self).__init__()
        self.hidden_dim, self.input_dim, self.sigma = hidden_dim, input_dim, sigma
        adj_mlp = nn.Linear(hidden_dim, 1)
        Truncated_initializer(adj_mlp.weight)
        nn.init.constant_(adj_mlp.bias, 0.1)

        gate_mlp = nn.Linear(hidden_dim, 1)
        Truncated_initializer(gate_mlp.weight)
        nn.init.constant_(gate_mlp.bias, 0.1)

        self.softmax = nn.Softmax(dim=0)
        self.meta_graph_vertex_num = meta_graph_vertex_num
        self.proto_graph_vertex_num = proto_graph_vertex_num
        self.meta_graph_vertex = nn.Parameter(torch.rand(meta_graph_vertex_num, input_dim), requires_grad=False)
        self.distance = nn.Sequential(adj_mlp, nn.Sigmoid())
        self.gate = nn.Sequential(gate_mlp, nn.Sigmoid())
        self.device = torch.device('cuda')
        self.meta_GCN = Graph_Convolution(self.hidden_dim).to(self.device)
        self.MSE = nn.MSELoss(reduce='mean')

    def forward(self, inputs):
        batch_size = inputs.size(0)
        protos = inputs
        meta_graph = self._construct_graph(self.meta_graph_vertex, self.meta_graph_vertex).to(self.device)
        proto_graph = self._construct_graph(protos, protos).to(self.device)
        m, n = protos.size(0), self.meta_graph_vertex.size(0)
        xx = torch.pow(protos, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(self.meta_graph_vertex, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(mat1=protos, mat2=self.meta_graph_vertex.t(), beta=1, alpha=-2)
        dist_square = dist.clamp(min=1e-6)
        cross_graph = self.softmax((- dist_square / (2.0 * self.sigma))).to(self.device)
        super_garph = torch.cat((torch.cat((proto_graph, cross_graph), dim=1), torch.cat((cross_graph.t(), meta_graph), dim=1)), dim=0)
        feature = torch.cat((protos, self.meta_graph_vertex), dim=0).to(self.device)
        representation = self.meta_GCN(feature, super_garph)

        # most_similar_index = torch.argmax(cross_graph, dim=1)
        normalized_transfered_meta = F.normalize(representation[batch_size:])
        normalized_meta = F.normalize(self.meta_graph_vertex)
        ccT = torch.mm(normalized_transfered_meta, normalized_transfered_meta.t())
        mmT = torch.mm(normalized_meta, normalized_meta.t())
        # I = torch.eye(self.meta_graph_vertex_num, requires_grad=False).to(self.device)
        correlation = self.MSE(ccT, mmT)

        return representation[0:batch_size].to(self.device), correlation

    def _construct_graph(self, A, B):
        m = A.size(0)
        n = B.size(0)
        I = torch.eye(n, requires_grad=False).to(self.device)
        index_aabb = torch.arange(0, m, requires_grad=False).repeat_interleave(n, dim=0).long()
        index_abab = torch.arange(0, n, requires_grad=False).repeat(m).long()
        diff = A[index_aabb] - B[index_abab]
        graph = self.distance(diff).view(m, n)
        graph = graph.to(self.device) * (1 - I) + I
        return graph
