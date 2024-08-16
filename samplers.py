import copy
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_ir_label: labels of two modalities
            color_pos, ir_pos: positions of each identity
            batch_size: batch size
    """

    def __init__(self, trainset, rgb_pos, ir_pos, num_pos, batch_size, pid_decrease):
        train_rgb_label = trainset.train_rgb_label
        train_ir_label = trainset.train_ir_label
        uni_label = np.unique(train_rgb_label)
        self.n_classes = len(uni_label)

        N = np.maximum(len(train_rgb_label), len(train_ir_label))
        for j in range(int(N / (batch_size * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label, batch_size, replace=False)
            for i in range(batch_size):
                sample_rgb = np.random.choice(rgb_pos[batch_idx[i] - pid_decrease], num_pos)
                sample_ir = np.random.choice(ir_pos[batch_idx[i] - pid_decrease], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_rgb
                    index2 = sample_ir
                else:
                    index1 = np.hstack((index1, sample_rgb))
                    index2 = np.hstack((index2, sample_ir))

        self.index1 = index1
        trainset.rIndex = index1
        trainset.iIndex = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


class BiCIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances, batch_size):
        self.data_source = data_source
        self.num_instances = num_instances
        self.batch_size = batch_size

        self.pid2index_rgb = defaultdict(list)
        self.pid2index_ir = defaultdict(list)
        for index, (pid1, pid2) in enumerate(zip(data_source.train_rgb_label, data_source.train_ir_label)):
            self.pid2index_rgb[pid1].append(index)
            self.pid2index_ir[pid2].append(index)

        self.pids = list(self.pid2index_rgb.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_instances * self.num_samples

    def __iter__(self):
        iterator_list = []

        for pid in self.pids:
            idx_rgb = copy.deepcopy(self.pid2index_rgb[pid])
            idx_ir = copy.deepcopy(self.pid2index_ir[pid])

            idx_rgb = np.random.choice(idx_rgb, size=self.num_instances, replace=False)
            idx_ir = np.random.choice(idx_ir, size=self.num_instances, replace=False)

            iterator_list.append((idx_rgb[0], idx_ir[0]))
            iterator_list.append((idx_rgb[1], idx_ir[1]))
        return iter(iterator_list)


class ReplyIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances, batch_size, reply_type='default', length=0):
        self.data_source = data_source
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.reply_type = reply_type

        self.domains2pids = defaultdict(list)
        self.pid2index_rgb = defaultdict(list)
        self.pid2index_ir = defaultdict(list)
        for index, (pid, domain) in enumerate(zip(data_source.reply_rgb_label, data_source.rgb_domain_flag)):
            if pid not in self.domains2pids[domain]:
                self.domains2pids[domain].append(pid)
            self.pid2index_rgb[pid].append(index)

        for index, (pid, domain) in enumerate(zip(data_source.reply_ir_label, data_source.ir_domain_flag)):
            self.pid2index_ir[pid].append(index)

        self.pids = list(self.pid2index_rgb.keys())
        self.domains = list(sorted(self.domains2pids.keys()))
        self.num_samples = len(self.pids)
        if length > 0:
            self.num_instances = length * batch_size // self.num_samples
            # print(length, batch_size, self.num_samples, self.num_instances)

    def __len__(self):
        return self.num_instances * self.num_samples

    def __iter__(self):
        iterator_list = []
        domain2pids = copy.deepcopy(self.domains2pids)
        for domain_index in range(len(self.domains)):
            pids = np.random.choice(domain2pids[self.domains[domain_index]], size=len(domain2pids[self.domains[domain_index]]), replace=False)
            for pid in pids:
                if self.reply_type == 'pcb':
                    idx_rgb = copy.deepcopy(self.pid2index_rgb[pid])[0]
                    idx_ir = copy.deepcopy(self.pid2index_ir[pid])[0]
                    for _ in range(self.num_instances):
                        iterator_list.append((idx_rgb, idx_ir))
                else:
                    idx_rgb = copy.deepcopy(self.pid2index_rgb[pid])
                    idx_ir = copy.deepcopy(self.pid2index_ir[pid])
                    idx_rgb = np.random.choice(idx_rgb, size=self.num_instances, replace=True)
                    idx_ir = np.random.choice(idx_ir, size=self.num_instances, replace=True)
                    for i in range(self.num_instances):
                        iterator_list.append((idx_rgb[i], idx_ir[i]))
        # 随机打乱
        np.random.shuffle(iterator_list)
        return iter(iterator_list)