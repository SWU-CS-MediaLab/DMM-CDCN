import copy
import errno
import math
import os
import random
from collections import defaultdict
import matplotlib.colors as mcolors
import numpy as np
import sys
import os.path as osp
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt, rcParams
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from torch.distributions.uniform import Uniform

from model_utils import KMeansPlusPlus


def cosine_distance(input1, input2):
    """Computes cosine distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


def cosine_similarity(input1, input2):
    """Computes cosine distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = torch.mm(input1_normed, input2_normed.t())
    return distmat


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=-2, alpha=1)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def _batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if (indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


def _batch_mid_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
    hard_p = sorted_mat_distance[:, 1]
    hard_p_indice = positive_indices[:, 1]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if (indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


def process_meta_data(input1, input2, label1, label2, num_pos, k_shots):
    support_index = []
    for id_idx in range(int(input1.shape[0] / num_pos)):
        support_index.extend(id_idx * num_pos + (np.random.choice(num_pos, k_shots, False)))
    query_index = np.delete(np.arange(input1.shape[0]), support_index)
    support_rgb = input1[support_index].cuda()
    support_ir = input2[support_index].cuda()
    support_labels = torch.cat((label1[support_index], label2[support_index]), 0).cuda()
    query_rgb = input1[query_index].cuda()
    query_ir = input2[query_index].cuda()
    query_labels = torch.cat((label1[query_index], label2[query_index]), 0).cuda()
    return support_rgb, support_ir, support_labels, query_rgb, query_ir, query_labels


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of color image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
    return file_image, file_label


def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos


def GenCamIdx(gall_img, gall_label, mode):
    if mode == 'indoor':
        camIdx = [1, 2]
    else:
        camIdx = [1, 2, 4, 5]
    gall_cam = []
    for i in range(len(gall_img)):
        gall_cam.append(int(gall_img[i][-10]))

    sample_pos = []
    unique_label = np.unique(gall_label)
    for i in range(len(unique_label)):
        for j in range(len(camIdx)):
            id_pos = [k for k, v in enumerate(gall_label) if v == unique_label[i] and gall_cam[k] == camIdx[j]]
            if id_pos:
                sample_pos.append(id_pos)
    return sample_pos


def ExtractCam(gall_img):
    gall_cam = []
    for i in range(len(gall_img)):
        cam_id = int(gall_img[i][-10])
        # if cam_id ==3:
        # cam_id = 2
        gall_cam.append(cam_id)

    return np.array(gall_cam)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
            Parameters:
                nets (network list)   -- a list of networks
                requires_grad (bool)  -- whether the networks require gradients or not
            """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def save_state(state_dict, out_features, next_epoch, next_step, dataset, save_path):
    state = {'state_dict': state_dict, 'out_features': out_features, 'next_epoch': next_epoch, 'next_step': next_step, 'dataset': dataset}
    torch.save(state, save_path)


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = torch.nonzero(backward_k_neigh_index == i)[:, 0]
    return forward_k_neigh_index[fi]


def compute_jaccard_dist(target_features, k1=20, k2=6):
    N = target_features.size(0)
    target_features = target_features.cuda()

    original_dist = torch.pow(target_features, 2).sum(dim=1, keepdim=True) * 2
    original_dist = original_dist.expand(N, N) - 2 * torch.mm(target_features, target_features.t())
    original_dist /= original_dist.max(0)[0]
    original_dist = original_dist.t()
    initial_rank = torch.argsort(original_dist, dim=-1)

    initial_rank = initial_rank.cpu()

    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    V = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        # k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)  ## element-wise unique
        # weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])
        # V[i, k_reciprocal_expansion_index] = weight / torch.sum(weight)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2 - 2 * torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        # k2_rank = initial_rank[:, :k2].clone().view(-1)
        # V_qe = V[k2_rank]
        # V_qe = V_qe.view(initial_rank.size(0), k2, -1).sum(1)
        # V_qe /= k2
        # V = V_qe
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(N):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=N

    jaccard_dist = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=np.float32)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])  # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)  # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0

    return jaccard_dist


def select_samples(rgb_pids2imgs, ir_pids2imgs, rgb_pids2feats, ir_pids2feats, domain_flag, reply_type='fmh', select_nums=256, num_instance=2, pid_increase=0, pid_incorrect=None):
    rgb_samples = []
    ir_samples = []
    rgb_labels = []
    ir_labels = []
    rgb_cams = []
    ir_cams = []
    rgb_domain_flag = []
    ir_domain_flag = []

    if reply_type == 'fmh':
        selected_pids = []
        selected_features = []
        dataset_center = []
        pids2feats = copy.deepcopy(rgb_pids2feats)
        pids2feats.update(ir_pids2feats)
        for pid in pids2feats.keys():
            dataset_center.extend(pids2feats[pid])
        dataset_center = F.normalize(torch.stack(dataset_center), dim=1, p=2).mean(0)
        pids_center = {pid: F.normalize(torch.stack(pids2feats[pid]), dim=1, p=2).mean(0) for pid in pids2feats.keys()}
        similarity = torch.mm(F.normalize(torch.stack(list(pids_center.values())), dim=1, p=2), dataset_center.unsqueeze(0).t())
        min_similarity_idx = torch.argmin(similarity)
        # 添加数据集中心与第一个选中的类中心的相似度
        selected_pids.append(list(pids2feats.keys())[min_similarity_idx])
        selected_features.append(dataset_center)
        selected_features.append(pids_center[selected_pids[-1]])

        # 选select_nums-1个类
        for i in range(select_nums - 1):
            # 每次计算当前类与已选择类相似度之和最小
            select_class = (-1, 500000, 0)
            for pid in pids2feats.keys():
                if pid not in selected_pids:
                    similarity_sum = torch.mm(torch.stack(selected_features), pids_center[pid].unsqueeze(0).t()).sum()
                    if similarity_sum < select_class[1]:
                        select_class = (pid, similarity_sum, pids_center[pid])
            if select_class[0] != -1:
                selected_pids.append(select_class[0])
                selected_features.append(pids_center[select_class[0]])

        # 每个类选2张rgb和2张ir
        for pid in selected_pids:
            # rgb
            rgb_center = F.normalize(torch.stack(rgb_pids2feats[pid]), dim=1, p=2).mean(0)
            rgb_feats = F.normalize(torch.stack(rgb_pids2feats[pid]), dim=1, p=2)
            similarity = torch.mm(rgb_feats, rgb_center.unsqueeze(0).t())
            min_similarity_idx = torch.argmin(similarity)
            rgb_samples.append(rgb_pids2imgs[pid][min_similarity_idx])
            rgb_labels.append(pid)
            rgb_cams.append(0)
            rgb_domain_flag.append(domain_flag)
            rgb_selected_features = [rgb_center, rgb_feats[min_similarity_idx]]
            min_similarity_idx = [min_similarity_idx]
            for i in range(num_instance - 1):
                select_img = (-1, 50000, 0)
                for idx in range(len(rgb_pids2feats[pid])):
                    if idx not in min_similarity_idx:
                        similarity_sum = torch.mm(torch.stack(rgb_selected_features), rgb_feats[idx].unsqueeze(0).t(), ).sum()
                        if similarity_sum < select_img[1]:
                            select_img = (idx, similarity_sum, rgb_feats[idx])
                if select_img[0] != -1:
                    min_similarity_idx.append(select_img[0])
                    rgb_selected_features.append(select_img[2])
                    rgb_samples.append(rgb_pids2imgs[pid][select_img[0]])
                    rgb_cams.append(0)
                    rgb_labels.append(pid)
                    rgb_domain_flag.append(domain_flag)
                else:
                    print('error_rgb_reply')

            # ir
            ir_center = F.normalize(torch.stack(ir_pids2feats[pid]), dim=1, p=2).mean(0)
            ir_feats = F.normalize(torch.stack(ir_pids2feats[pid]), dim=1, p=2)
            similarity = torch.mm(ir_feats, ir_center.unsqueeze(0).t())
            min_similarity_idx = torch.argmin(similarity)
            ir_samples.append(ir_pids2imgs[pid][min_similarity_idx])
            ir_labels.append(pid)
            ir_cams.append(1)
            ir_domain_flag.append(domain_flag)
            ir_selected_features = [ir_center, ir_feats[min_similarity_idx]]
            min_similarity_idx = [min_similarity_idx]
            for i in range(num_instance - 1):
                select_img = (-1, 50000, 0)
                for idx in range(len(ir_pids2feats[pid])):
                    if idx not in min_similarity_idx:
                        similarity_sum = torch.mm(torch.stack(ir_selected_features), ir_feats[idx].unsqueeze(0).t()).sum()
                        if similarity_sum < select_img[1]:
                            select_img = (idx, similarity_sum, ir_feats[idx])
                if select_img[0] != -1:
                    min_similarity_idx.append(select_img[0])
                    ir_selected_features.append(select_img[2])
                    ir_samples.append(ir_pids2imgs[pid][select_img[0]])
                    ir_cams.append(1)
                    ir_labels.append(pid)
                    ir_domain_flag.append(domain_flag)
                else:
                    print('error_ir_reply')

    elif reply_type == 'icarl':
        selected_pids = random.sample(list(rgb_pids2imgs.keys()), min(len(rgb_pids2imgs.keys()), select_nums))
        for pid in selected_pids:
            # rgb
            select_features = []
            rgb_features = F.normalize(torch.stack(rgb_pids2feats[pid]), dim=1, p=2)
            pid_center = rgb_features.mean(0)
            for i in range(num_instance):
                # 求select的和
                features_sum = torch.stack(select_features).sum() if len(select_features) else 0
                mu_p = 1.0 / (i + 1) * (features_sum + rgb_features)
                mu_p = mu_p / torch.norm(mu_p, dim=1, keepdim=True)
                dist = torch.sqrt(torch.sum((pid_center - mu_p) ** 2, dim=1))
                idx = torch.argmin(dist)
                rgb_samples.append(rgb_pids2imgs[pid][idx])
                rgb_labels.append(pid)
                rgb_cams.append(0)
                rgb_domain_flag.append(domain_flag)
                select_features.append(rgb_features[idx])

                del rgb_pids2imgs[pid][idx]
                rgb_features = torch.cat([rgb_features[:idx], rgb_features[idx + 1:]], 0)

            # ir
            select_features = []
            ir_features = F.normalize(torch.stack(ir_pids2feats[pid]), dim=1, p=2)
            pid_center = ir_features.mean(0)
            for i in range(num_instance):
                # 求select的和
                features_sum = torch.stack(select_features).sum() if len(select_features) else 0
                mu_p = 1.0 / (i + 1) * (features_sum + ir_features)
                mu_p = mu_p / torch.norm(mu_p, dim=1, keepdim=True)
                dist = torch.sqrt(torch.sum((pid_center - mu_p) ** 2, dim=1))
                idx = torch.argmin(dist)
                ir_samples.append(ir_pids2imgs[pid][idx])
                ir_labels.append(pid)
                ir_cams.append(1)
                ir_domain_flag.append(domain_flag)
                select_features.append(ir_features[idx])

                del ir_pids2imgs[pid][idx]
                ir_features = torch.cat([ir_features[:idx], ir_features[idx + 1:]], 0)

    elif reply_type == 'hcr':
        # compute the center of each class
        selected_pids = []
        pids2feats = copy.deepcopy(rgb_pids2feats)
        pids2feats.update(ir_pids2feats)
        pids_center = {pid: F.normalize(torch.stack(pids2feats[pid]), dim=1).mean(0) for pid in sorted(pids2feats.keys())}
        pids_center_list = list(pids_center.values())
        rerank_dist = compute_jaccard_dist(F.normalize(torch.stack(pids_center_list), dim=1))
        cluster_pid = DBSCAN(eps=0.2, min_samples=1, metric='precomputed', n_jobs=-1)
        cluster_labels = cluster_pid.fit_predict(rerank_dist)
        # compute nearset pid of each cluster
        for i in range(len(np.unique(cluster_labels))):
            cluster_pids = np.where(cluster_labels == i)[0]
            # print('cluster_pids', len(cluster_pids))
            cluster_center = F.normalize(torch.stack([pids_center[pid] for pid in cluster_pids]), dim=1).mean(0)
            cluster_dist_for_all_pids_in_cluster = torch.stack([torch.sqrt((pids_center[pid] - cluster_center) ** 2).sum() for pid in cluster_pids])

            # print('min', cluster_dist_for_all_pids_in_cluster)
            nearest_pid = cluster_pids[torch.argmin(cluster_dist_for_all_pids_in_cluster)]
            selected_pids.append(nearest_pid)
        print('total selected pids', len(selected_pids))

        # compute the center of imgs in pid
        cluster_img = DBSCAN(eps=0.0025, min_samples=1, metric='precomputed', n_jobs=-1)
        for pid in selected_pids:
            feats = F.normalize(torch.stack(rgb_pids2feats[pid]), dim=1)
            cluster_labels = cluster_img.fit_predict(compute_jaccard_dist(feats))
            # print('rgb_cluster_nums', len(np.unique(cluster_labels)))
            for i in range(len(np.unique(cluster_labels))):
                cluster_imgs = np.where(cluster_labels == i)[0]
                cluster_center = F.normalize(torch.stack([feats[j] for j in cluster_imgs]), dim=1).mean(0)
                cluster_dist_for_all_imgs_in_pid = torch.stack([torch.sqrt((feats[j] - cluster_center) ** 2).sum() for j in cluster_imgs])
                nearest_idx = cluster_imgs[torch.argmin(cluster_dist_for_all_imgs_in_pid)]
                rgb_samples.append(rgb_pids2imgs[nearest_idx])
                rgb_labels.append(pid)
                rgb_cams.append(0)
                rgb_domain_flag.append(domain_flag)

        for pid in selected_pids:
            feats = F.normalize(torch.stack(ir_pids2feats[pid]), dim=1)
            cluster_labels = cluster_img.fit_predict(compute_jaccard_dist(feats))
            # print('ir_cluster_nums', len(np.unique(cluster_labels)))
            for i in range(len(np.unique(cluster_labels))):
                cluster_imgs = np.where(cluster_labels == i)[0]
                cluster_center = F.normalize(torch.stack([feats[j] for j in cluster_imgs]), dim=1).mean(0)
                cluster_dist_for_all_imgs_in_pid = torch.stack([torch.sqrt((feats[j] - cluster_center) ** 2).sum() for j in cluster_imgs])
                nearest_idx = cluster_imgs[torch.argmin(cluster_dist_for_all_imgs_in_pid)]
                ir_samples.append(ir_pids2imgs[nearest_idx])
                ir_labels.append(pid)
                ir_cams.append(1)
                ir_domain_flag.append(domain_flag)

    elif reply_type == 'nmh':
        selected_pids = []
        selected_features = []
        dataset_center = []
        pids2feats = copy.deepcopy(rgb_pids2feats)
        pids2feats.update(ir_pids2feats)
        for pid in pids2feats.keys():
            dataset_center.extend(pids2feats[pid])
        dataset_center = F.normalize(torch.stack(dataset_center), dim=1, p=2).mean(0)
        pids_center = {pid: F.normalize(torch.stack(pids2feats[pid]), dim=1, p=2).mean(0) for pid in pids2feats.keys()}
        similarity = torch.mm(F.normalize(torch.stack(list(pids_center.values())), dim=1, p=2), dataset_center.unsqueeze(0).t())
        min_similarity_idx = torch.argmin(similarity)
        # 添加数据集中心与第一个选中的类中心的相似度
        selected_pids.append(list(pids2feats.keys())[min_similarity_idx])
        selected_features.append(dataset_center)
        selected_features.append(pids_center[selected_pids[-1]])

        # 选select_nums-1个类
        for i in range(select_nums - 1):
            # 每次计算当前类与已选择类相似度之和最小
            select_class = (-1, -1, 0)
            for pid in pids2feats.keys():
                if pid not in selected_pids:
                    similarity_sum = torch.mm(torch.stack(selected_features), pids_center[pid].unsqueeze(0).t()).sum()
                    if similarity_sum > select_class[1]:
                        select_class = (pid, similarity_sum, pids_center[pid])
            if select_class[0] != -1:
                selected_pids.append(select_class[0])
                selected_features.append(pids_center[select_class[0]])

        # 每个类选2张rgb和2张ir
        for pid in selected_pids:
            # rgb
            rgb_center = F.normalize(torch.stack(rgb_pids2feats[pid]), dim=1, p=2).mean(0)
            rgb_feats = F.normalize(torch.stack(rgb_pids2feats[pid]), dim=1, p=2)
            similarity = torch.mm(rgb_feats, rgb_center.unsqueeze(0).t())
            min_similarity_idx = torch.argmax(similarity)
            rgb_samples.append(rgb_pids2imgs[pid][min_similarity_idx])
            rgb_labels.append(pid)
            rgb_cams.append(0)
            rgb_domain_flag.append(domain_flag)
            rgb_selected_features = [rgb_center, rgb_feats[min_similarity_idx]]
            min_similarity_idx = [min_similarity_idx]
            for i in range(num_instance - 1):
                select_img = (-1, -1, 0)
                for idx in range(len(rgb_pids2feats[pid])):
                    if idx not in min_similarity_idx:
                        similarity_sum = torch.mm(torch.stack(rgb_selected_features), rgb_feats[idx].unsqueeze(0).t(), ).sum()
                        if similarity_sum > select_img[1]:
                            select_img = (idx, similarity_sum, rgb_feats[idx])
                if select_img[0] != -1:
                    min_similarity_idx.append(select_img[0])
                    rgb_selected_features.append(select_img[2])
                    rgb_samples.append(rgb_pids2imgs[pid][select_img[0]])
                    rgb_cams.append(0)
                    rgb_labels.append(pid)
                    rgb_domain_flag.append(domain_flag)
                else:
                    print('error_rgb_reply')

            # ir
            ir_center = F.normalize(torch.stack(ir_pids2feats[pid]), dim=1, p=2).mean(0)
            ir_feats = F.normalize(torch.stack(ir_pids2feats[pid]), dim=1, p=2)
            similarity = torch.mm(ir_feats, ir_center.unsqueeze(0).t())
            min_similarity_idx = torch.argmax(similarity)
            ir_samples.append(ir_pids2imgs[pid][min_similarity_idx])
            ir_labels.append(pid)
            ir_cams.append(1)
            ir_domain_flag.append(domain_flag)
            ir_selected_features = [ir_center, ir_feats[min_similarity_idx]]
            min_similarity_idx = [min_similarity_idx]
            for i in range(num_instance - 1):
                select_img = (-1, -1, 0)
                for idx in range(len(ir_pids2feats[pid])):
                    if idx not in min_similarity_idx:
                        similarity_sum = torch.mm(torch.stack(ir_selected_features), ir_feats[idx].unsqueeze(0).t()).sum()
                        if similarity_sum > select_img[1]:
                            select_img = (idx, similarity_sum, ir_feats[idx])
                if select_img[0] != -1:
                    min_similarity_idx.append(select_img[0])
                    ir_selected_features.append(select_img[2])
                    ir_samples.append(ir_pids2imgs[pid][select_img[0]])
                    ir_cams.append(1)
                    ir_labels.append(pid)
                    ir_domain_flag.append(domain_flag)
                else:
                    print('error_ir_reply')

    elif reply_type == 'rmh':
        selected_pids = []
        selected_features = []
        dataset_center = []
        pids2feats = copy.deepcopy(rgb_pids2feats)
        for pid in pids2feats.keys():
            dataset_center.extend(pids2feats[pid])
        dataset_center = F.normalize(torch.stack(dataset_center), dim=1, p=2).mean(0)
        pids_center = {pid: F.normalize(torch.stack(pids2feats[pid]), dim=1, p=2).mean(0) for pid in pids2feats.keys()}
        similarity = torch.mm(F.normalize(torch.stack(list(pids_center.values())), dim=1, p=2), dataset_center.unsqueeze(0).t())
        min_similarity_idx = torch.argmin(similarity)
        # 添加数据集中心与第一个选中的类中心的相似度
        selected_pids.append(list(pids2feats.keys())[min_similarity_idx])
        selected_features.append(dataset_center)
        selected_features.append(pids_center[selected_pids[-1]])

        # 选select_nums-1个类
        for i in range(select_nums - 1):
            # 每次计算当前类与已选择类相似度之和最小
            select_class = (-1, 500000, 0)
            for pid in pids2feats.keys():
                if pid not in selected_pids:
                    similarity_sum = torch.mm(torch.stack(selected_features), pids_center[pid].unsqueeze(0).t()).sum()
                    if similarity_sum < select_class[1]:
                        select_class = (pid, similarity_sum, pids_center[pid])
            if select_class[0] != -1:
                selected_pids.append(select_class[0])
                selected_features.append(pids_center[select_class[0]])

        # 每个类选2张rgb和2张ir
        for pid in selected_pids:
            # rgb
            rgb_center = F.normalize(torch.stack(rgb_pids2feats[pid]), dim=1, p=2).mean(0)
            rgb_feats = F.normalize(torch.stack(rgb_pids2feats[pid]), dim=1, p=2)
            similarity = torch.mm(rgb_feats, rgb_center.unsqueeze(0).t())
            min_similarity_idx = torch.argmax(similarity)
            rgb_samples.append(rgb_pids2imgs[pid][min_similarity_idx])
            rgb_labels.append(pid)
            rgb_cams.append(0)
            rgb_domain_flag.append(domain_flag)
            rgb_selected_features = [rgb_center, rgb_feats[min_similarity_idx]]
            min_similarity_idx = [min_similarity_idx]
            for i in range(num_instance - 1):
                select_img = (-1, -1, 0)
                for idx in range(len(rgb_pids2feats[pid])):
                    if idx not in min_similarity_idx:
                        similarity_sum = torch.mm(torch.stack(rgb_selected_features), rgb_feats[idx].unsqueeze(0).t(), ).sum()
                        if similarity_sum > select_img[1]:
                            select_img = (idx, similarity_sum, rgb_feats[idx])
                if select_img[0] != -1:
                    min_similarity_idx.append(select_img[0])
                    rgb_selected_features.append(select_img[2])
                    rgb_samples.append(rgb_pids2imgs[pid][select_img[0]])
                    rgb_cams.append(0)
                    rgb_labels.append(pid)
                    rgb_domain_flag.append(domain_flag)
                else:
                    print('error_rgb_reply')

            # ir
            ir_center = F.normalize(torch.stack(ir_pids2feats[pid]), dim=1, p=2).mean(0)
            ir_feats = F.normalize(torch.stack(ir_pids2feats[pid]), dim=1, p=2)
            similarity = torch.mm(ir_feats, ir_center.unsqueeze(0).t())
            min_similarity_idx = torch.argmax(similarity)
            ir_samples.append(ir_pids2imgs[pid][min_similarity_idx])
            ir_labels.append(pid)
            ir_cams.append(1)
            ir_domain_flag.append(domain_flag)
            ir_selected_features = [ir_center, ir_feats[min_similarity_idx]]
            min_similarity_idx = [min_similarity_idx]
            for i in range(num_instance - 1):
                select_img = (-1, -1, 0)
                for idx in range(len(ir_pids2feats[pid])):
                    if idx not in min_similarity_idx:
                        similarity_sum = torch.mm(torch.stack(ir_selected_features), ir_feats[idx].unsqueeze(0).t()).sum()
                        if similarity_sum > select_img[1]:
                            select_img = (idx, similarity_sum, ir_feats[idx])
                if select_img[0] != -1:
                    min_similarity_idx.append(select_img[0])
                    ir_selected_features.append(select_img[2])
                    ir_samples.append(ir_pids2imgs[pid][select_img[0]])
                    ir_cams.append(1)
                    ir_labels.append(pid)
                    ir_domain_flag.append(domain_flag)
                else:
                    print('error_ir_reply')

    elif reply_type == 'kmh':
        selected_pids = []
        if len(rgb_pids2feats.keys()) > 256:
            selected_feats = []
            pids2feats = copy.deepcopy(rgb_pids2feats)
            pids2feats.update(ir_pids2feats)
            pids_center = {pid: F.normalize(torch.stack(pids2feats[pid]), dim=1, p=2).mean(0) for pid in sorted(pids2feats.keys())}
            kmeanspp = KMeansPlusPlus()
            cluster_labels = kmeanspp(torch.stack(list(pids_center.values()))).cpu().numpy()
            print('total cluster nums', len(np.unique(cluster_labels)))
            # print('cluster labels', np.unique(cluster_labels))
            for i in list(np.unique(cluster_labels)):
                cluster_pids = list(np.where(cluster_labels == i)[0] + pid_increase)
                # print(cluster_pids)
                cluster_center = F.normalize(torch.stack([pids_center[pid] for pid in cluster_pids]), dim=1).mean(0)
                cluster_distances = torch.cdist(F.normalize(torch.stack([pids_center[pid] for pid in cluster_pids]), dim=1, p=2), cluster_center.unsqueeze(0))
                selected_pids.append(cluster_pids[cluster_distances.argmin(0)])
                selected_feats.append(pids_center[selected_pids[-1]])
            # 寻找离群样本
            residual_num = select_nums - len(selected_pids)
            if residual_num > 0:
                for _ in range(residual_num):
                    select_class = (-1, -1, 0)
                    for pid in pids2feats.keys():
                        if pid not in selected_pids:
                            # 计算距离
                            pids_distance = torch.cdist(torch.stack(selected_feats), pids_center[pid].unsqueeze(0)).sum()
                            if pids_distance > select_class[1]:
                                select_class = (pid, pids_distance, pids_center[pid])
                    if select_class[0] != -1:
                        selected_pids.append(select_class[0])
                        selected_feats.append(select_class[-1])
                    else:
                        print('error_km_reply')
        else:
            selected_pids = list(rgb_pids2feats.keys())

        for pid in selected_pids:
            rgb_feats = F.normalize(torch.stack(rgb_pids2feats[pid]), dim=1, p=2)
            ir_feats = F.normalize(torch.stack(ir_pids2feats[pid]), dim=1, p=2)
            rgb_ir_distances = torch.cdist(rgb_feats, ir_feats)
            # max_distance, max_indices = rgb_ir_distances.max(dim=1)
            # global_max_distance, global_max_index = max_distance.max(dim=0)
            # first pair
            max_pair = (-1, -1, -1)
            for i in range(rgb_ir_distances.shape[0]):
                for j in range(rgb_ir_distances.shape[1]):
                    if rgb_ir_distances[i][j] > max_pair[-1]:
                        max_pair = (i, j, rgb_ir_distances[i][j])  # feature_mean2 = (rgb_feats[i] + ir_feats[j]) / 2  # 计算mean距离  # dist = torch.sqrt(torch.sum((feature_mean1 - feature_mean2) ** 2))
            if max_pair[0] != -1:
                rgb_samples.append(rgb_pids2imgs[pid][max_pair[0]])
                rgb_labels.append(pid)
                rgb_cams.append(0)
                rgb_domain_flag.append(domain_flag)
                ir_samples.append(ir_pids2imgs[pid][max_pair[1]])
                ir_labels.append(pid)
                ir_cams.append(1)
                ir_domain_flag.append(domain_flag)
                rgb_idx, ir_idx = max_pair[0], max_pair[1]

            # second pair
            max_pair = (-1, -1, -1)
            for i in range(rgb_ir_distances.shape[0]):
                for j in range(rgb_ir_distances.shape[1]):
                    if i != rgb_idx and j != ir_idx:
                        if rgb_ir_distances[i][j] > max_pair[-1]:
                            max_pair = (i, j, rgb_ir_distances[i][j])  # feature_mean2 = (rgb_feats[i] + ir_feats[j]) / 2  # 计算mean距离  # dist = torch.sqrt(torch.sum((feature_mean1 - feature_mean2) ** 2))
            if max_pair[0] != -1:
                rgb_samples.append(rgb_pids2imgs[pid][max_pair[0]])
                rgb_labels.append(pid)
                rgb_cams.append(0)
                rgb_domain_flag.append(domain_flag)
                ir_samples.append(ir_pids2imgs[pid][max_pair[1]])
                ir_labels.append(pid)
                ir_cams.append(1)
                ir_domain_flag.append(domain_flag)

    elif reply_type == 'imh':
        selected_pids = []
        if len(rgb_pids2feats.keys()) > 256:
            selected_feats = []
            pids2feats = copy.deepcopy(rgb_pids2feats)
            pids2feats.update(ir_pids2feats)
            pids_center = {pid: F.normalize(torch.stack(pids2feats[pid]), dim=1, p=2).mean(0) for pid in sorted(pids2feats.keys())}
            kmeanspp = KMeansPlusPlus()
            cluster_labels = kmeanspp(torch.stack(list(pids_center.values()))).cpu().numpy()
            print('total cluster nums', len(np.unique(cluster_labels)))
            for i in list(np.unique(cluster_labels)):
                cluster_pids = list(np.where(cluster_labels == i)[0] + pid_increase)
                cluster_center = F.normalize(torch.stack([pids_center[pid] for pid in cluster_pids]), dim=1).mean(0)
                cluster_distances = torch.cdist(F.normalize(torch.stack([pids_center[pid] for pid in cluster_pids]), dim=1, p=2), cluster_center.unsqueeze(0))
                selected_pids.append(cluster_pids[cluster_distances.argmin(0)])
                selected_feats.append(pids_center[selected_pids[-1]])
            # 寻找离群样本
            residual_num = select_nums - len(selected_pids)
            if residual_num > 0:
                for _ in range(residual_num):
                    select_class = (-1, -1, 0)
                    for pid in pids2feats.keys():
                        if pid not in selected_pids:
                            # 计算距离
                            pids_distance = torch.cdist(torch.stack(selected_feats), pids_center[pid].unsqueeze(0)).sum()
                            if pids_distance > select_class[1]:
                                select_class = (pid, pids_distance, pids_center[pid])
                    if select_class[0] != -1:
                        selected_pids.append(select_class[0])
                        selected_feats.append(select_class[-1])
                    else:
                        print('error_km_reply')
        else:
            selected_pids = list(rgb_pids2feats.keys())
        for pid in selected_pids:
            # rgb
            select_features = []
            rgb_features = F.normalize(torch.stack(rgb_pids2feats[pid]), dim=1, p=2)
            pid_center = rgb_features.mean(0)
            for i in range(num_instance):
                # 求select的和
                features_sum = torch.stack(select_features).sum() if len(select_features) else 0
                mu_p = 1.0 / (i + 1) * (features_sum + rgb_features)
                mu_p = mu_p / torch.norm(mu_p, dim=1, keepdim=True)
                dist = torch.sqrt(torch.sum((pid_center - mu_p) ** 2, dim=1))
                idx = torch.argmin(dist)
                rgb_samples.append(rgb_pids2imgs[pid][idx])
                rgb_labels.append(pid)
                rgb_cams.append(0)
                rgb_domain_flag.append(domain_flag)
                select_features.append(rgb_features[idx])

                del rgb_pids2imgs[pid][idx]
                rgb_features = torch.cat([rgb_features[:idx], rgb_features[idx + 1:]], 0)

            # ir
            select_features = []
            ir_features = F.normalize(torch.stack(ir_pids2feats[pid]), dim=1, p=2)
            pid_center = ir_features.mean(0)
            for i in range(num_instance):
                # 求select的和
                features_sum = torch.stack(select_features).sum() if len(select_features) else 0
                mu_p = 1.0 / (i + 1) * (features_sum + ir_features)
                mu_p = mu_p / torch.norm(mu_p, dim=1, keepdim=True)
                dist = torch.sqrt(torch.sum((pid_center - mu_p) ** 2, dim=1))
                idx = torch.argmin(dist)
                ir_samples.append(ir_pids2imgs[pid][idx])
                ir_labels.append(pid)
                ir_cams.append(1)
                ir_domain_flag.append(domain_flag)
                select_features.append(ir_features[idx])

                del ir_pids2imgs[pid][idx]
                ir_features = torch.cat([ir_features[:idx], ir_features[idx + 1:]], 0)

    elif reply_type == 'herd':
        class_centers_rgb = [torch.stack(rgb_pids2feats[pid]).mean(0) for pid in sorted(rgb_pids2feats.keys())]
        class_centers_ir = [torch.stack(ir_pids2feats[pid]).mean(0) for pid in sorted(ir_pids2feats.keys())]
        class_centers_rgb = F.normalize(torch.stack(class_centers_rgb), dim=1)
        class_centers_ir = F.normalize(torch.stack(class_centers_ir), dim=1)
        pids_all = list(rgb_pids2imgs.keys())
        select_pids = np.random.choice(pids_all, min(select_nums, len(pids_all)), replace=False)
        for pid in select_pids:
            feat2pid_rgb = F.normalize(torch.stack(rgb_pids2feats[pid]), dim=1, p=2)
            feat2pid_ir = F.normalize(torch.stack(ir_pids2feats[pid]), dim=1, p=2)
            center2pid_rgb = class_centers_rgb[pid - pid_increase]
            center2pid_ir = class_centers_ir[pid - pid_increase]
            similarity_rgb = torch.mm(feat2pid_rgb, center2pid_rgb.unsqueeze(0).t())
            similarity_ir = torch.mm(feat2pid_ir, center2pid_ir.unsqueeze(0).t())
            similarity_rgb_sort_inx = torch.sort(similarity_rgb, dim=0)[1][:num_instance]
            similarity_ir_sort_inx = torch.sort(similarity_ir, dim=0)[1][:num_instance]
            for idx in similarity_rgb_sort_inx:
                rgb_samples.append(rgb_pids2imgs[pid][idx])
                rgb_labels.append(pid)
                rgb_cams.append(0)
                rgb_domain_flag.append(domain_flag)
            for idx in similarity_ir_sort_inx:
                ir_samples.append(ir_pids2imgs[pid][idx])
                ir_labels.append(pid)
                ir_cams.append(1)
                ir_domain_flag.append(domain_flag)

    elif reply_type == 'der':
        selected_pids = random.sample(list(rgb_pids2imgs.keys()), min(len(rgb_pids2imgs.keys()), select_nums))
        for pid in selected_pids:
            rgb_samples.extend(random.sample(rgb_pids2imgs[pid], num_instance))
            rgb_labels.extend([pid, pid])
            rgb_cams.extend([0, 0])
            rgb_domain_flag.extend([domain_flag, domain_flag])
            ir_samples.append(random.sample(ir_pids2imgs[pid], num_instance))
            ir_labels.extend([pid, pid])
            ir_cams.extend([1, 1])
            rgb_domain_flag.extend([domain_flag, domain_flag])

    elif reply_type == 'b+der' and pid_incorrect is not None:
        selected_pids = []
        selected_features = []
        incorrect_pid = defaultdict(list)

        pids2feats = copy.deepcopy(rgb_pids2feats)
        pids2feats.update(ir_pids2feats)
        pids_center = {pid: F.normalize(torch.stack(pids2feats[pid]), dim=1, p=2).mean(0) for pid in pids2feats.keys()}
        # pid-inc -> inc-pid
        for k, v in pid_incorrect.items():
            incorrect_pid[v].append(k)

        for k in sorted(incorrect_pid.keys(), reverse=True):
            for pid in incorrect_pid[k]:
                if len(selected_pids) < select_nums and pid in rgb_pids2imgs.keys():
                    selected_pids.append(pid)
                    selected_features.append(pids_center[pid])

        residual_num = min(select_nums, len(rgb_pids2imgs.keys())) - len(selected_pids)
        if residual_num > 0:
            # 选select_nums-1个类
            for i in range(residual_num):
                # 每次计算当前类与已选择类相似度之和最小
                select_class = (-1, 50000, 0)
                for pid in pids2feats.keys():
                    if pid not in selected_pids:
                        similarity_sum = torch.mm(torch.stack(selected_features), pids_center[pid].unsqueeze(0).t()).sum()
                        if similarity_sum < select_class[1]:
                            select_class = (pid, similarity_sum, pids_center[pid])
                if select_class[0] != -1:
                    selected_pids.append(select_class[0])
                    selected_features.append(pids_center[select_class[0]])

        # 每个类选2张rgb和2张ir
        for pid in selected_pids:
            # rgb
            rgb_center = F.normalize(torch.stack(rgb_pids2feats[pid]), dim=1, p=2).mean(0)
            rgb_feats = F.normalize(torch.stack(rgb_pids2feats[pid]), dim=1, p=2)
            similarity = torch.mm(rgb_feats, rgb_center.unsqueeze(0).t())
            min_similarity_idx = torch.argmax(similarity)
            rgb_samples.append(rgb_pids2imgs[pid][min_similarity_idx])
            rgb_labels.append(pid)
            rgb_cams.append(0)
            rgb_domain_flag.append(domain_flag)
            rgb_selected_features = [rgb_center, rgb_feats[min_similarity_idx]]
            min_similarity_idx = [min_similarity_idx]
            for i in range(num_instance - 1):
                select_img = (-1, -1, 0)
                for idx in range(len(rgb_pids2feats[pid])):
                    if idx not in min_similarity_idx:
                        similarity_sum = torch.mm(torch.stack(rgb_selected_features), rgb_feats[idx].unsqueeze(0).t(), ).sum()
                        if similarity_sum > select_img[1]:
                            select_img = (idx, similarity_sum, rgb_feats[idx])
                if select_img[0] != -1:
                    min_similarity_idx.append(select_img[0])
                    rgb_selected_features.append(select_img[2])
                    rgb_samples.append(rgb_pids2imgs[pid][select_img[0]])
                    rgb_cams.append(0)
                    rgb_labels.append(pid)
                    rgb_domain_flag.append(domain_flag)
                else:
                    print('error_rgb_reply')

            # ir
            ir_center = F.normalize(torch.stack(ir_pids2feats[pid]), dim=1, p=2).mean(0)
            ir_feats = F.normalize(torch.stack(ir_pids2feats[pid]), dim=1, p=2)
            similarity = torch.mm(ir_feats, ir_center.unsqueeze(0).t())
            min_similarity_idx = torch.argmax(similarity)
            ir_samples.append(ir_pids2imgs[pid][min_similarity_idx])
            ir_labels.append(pid)
            ir_cams.append(1)
            ir_domain_flag.append(domain_flag)
            ir_selected_features = [ir_center, ir_feats[min_similarity_idx]]
            min_similarity_idx = [min_similarity_idx]
            for i in range(num_instance - 1):
                select_img = (-1, -1, 0)
                for idx in range(len(ir_pids2feats[pid])):
                    if idx not in min_similarity_idx:
                        similarity_sum = torch.mm(torch.stack(ir_selected_features), ir_feats[idx].unsqueeze(0).t()).sum()
                        if similarity_sum > select_img[1]:
                            select_img = (idx, similarity_sum, ir_feats[idx])
                if select_img[0] != -1:
                    min_similarity_idx.append(select_img[0])
                    ir_selected_features.append(select_img[2])
                    ir_samples.append(ir_pids2imgs[pid][select_img[0]])
                    ir_cams.append(1)
                    ir_labels.append(pid)
                    ir_domain_flag.append(domain_flag)
                else:
                    print('error_ir_reply')

    elif reply_type == 'pcb':
        selected_pids = random.sample(list(rgb_pids2imgs.keys()), min(len(rgb_pids2imgs.keys()), select_nums))
        for pid in selected_pids:
            tmp_img_list = []
            select_features = []
            feat = F.normalize(torch.stack(rgb_pids2feats[pid]), dim=1, p=2)
            imgs = copy.deepcopy(rgb_pids2imgs[pid])
            pid_center = feat.mean(0)

            for i in range(num_instance):
                features_sum = torch.stack(select_features).sum() if len(select_features) else 0
                mu_p = 1.0 / (i + 1) * (features_sum + feat)
                mu_p = mu_p / torch.norm(mu_p, dim=1, keepdim=True)
                dist = torch.sqrt(torch.sum((pid_center - mu_p) ** 2, dim=1))
                idx = torch.argmin(dist)
                tmp_img_list.append(imgs[idx])
                select_features.append(feat[idx])
                del imgs[idx]
                feat = torch.cat([feat[:idx], feat[idx + 1:]], 0)
            rgb_samples.append(tmp_img_list)
            rgb_labels.append(pid)
            rgb_cams.append(0)
            rgb_domain_flag.append(domain_flag)

            # ir
            feat = F.normalize(torch.stack(ir_pids2feats[pid]), dim=1, p=2)
            imgs = copy.deepcopy(ir_pids2imgs[pid])
            tmp_img_list = []
            select_features = []
            pid_center = feat.mean(0)
            for i in range(num_instance):
                features_sum = torch.stack(select_features).sum() if len(select_features) else 0
                mu_p = 1.0 / (i + 1) * (features_sum + feat)
                mu_p = mu_p / torch.norm(mu_p, dim=1, keepdim=True)
                dist = torch.sqrt(torch.sum((pid_center - mu_p) ** 2, dim=1))
                idx = torch.argmin(dist)
                tmp_img_list.append(imgs[idx])
                select_features.append(feat[idx])
                del imgs[idx]
                feat = torch.cat([feat[:idx], feat[idx + 1:]], 0)
            ir_samples.append(tmp_img_list)
            ir_labels.append(pid)
            ir_cams.append(1)
            ir_domain_flag.append(domain_flag)
    else:
        selected_pids = random.sample(list(rgb_pids2imgs.keys()), min(len(rgb_pids2imgs.keys()), select_nums))
        for pid in selected_pids:
            rgb_samples.extend(random.sample(rgb_pids2imgs[pid], num_instance))
            rgb_labels.extend([pid, pid])
            rgb_cams.extend([0, 0])
            rgb_domain_flag.extend([domain_flag, domain_flag])
            ir_samples.append(random.sample(ir_pids2imgs[pid], num_instance))
            ir_labels.extend([pid, pid])
            ir_cams.extend([1, 1])
            rgb_domain_flag.extend([domain_flag, domain_flag])

    return rgb_samples, rgb_labels, rgb_cams, rgb_domain_flag, ir_samples, ir_labels, ir_cams, ir_domain_flag


def tensor_cosine_dist(x, y):
    '''
    compute cosine distance between two matrix x and y
    with size (n1, d) and (n2, d) and type torch.tensor
    return a matrix (n1, n2)
    '''

    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return torch.matmul(x, y.transpose(0, 1))


def tensor_euclidean_dist(x, y):
    """
    compute euclidean distance between two matrix x and y
    with size (n1, d) and (n2, d) and type torch.tensor
    return a matrix (n1, n2)
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def T_SNE(feat, label, dataset_name):
    print(feat.shape)
    # start_color = np.array([255, 0.0, 0.0])  # 红色 (R=1, G=0, B=0)
    # end_color = np.array([0.0, 0.0, 255.0])  # 蓝色 (R=0, G=0, B=1)

    # 生成颜色列表，这里我们将颜色范围分为500份
    # steps = np.unique(label).shape[0]
    # # steps = 1000
    # print("labels", steps)
    # colors = [start_color + (end_color - start_color) * i / steps for i in range(steps)]
    # # 将颜色归一化至0-1之间，因为cmap需要这样的值
    # normalized_colors = [(color[0] / 255, color[1] / 255, color[2] / 255) for color in colors]
    # # 创建一个LinearSegmentedColormap
    # cmap_name = 'my_custom_cmap'
    # custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, normalized_colors, N=steps)
    # # 注册这个新的颜色映射以便在matplotlib中使用
    # plt.register_cmap(cmap=custom_cmap)
    rcParams['font.family'] = 'Liberation Serif'
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca', random_state=42)
    embedded_features = tsne.fit_transform(feat)
    plt.figure(figsize=(6, 10), dpi=600)
    plt.scatter(embedded_features[:, 0], embedded_features[:, 1], c=label, cmap='plasma', s=2)  # y是你对应的类别标签，如果没有标签则可以不设置c参数
    # figures_save_path = f'./tsn_imgs/'
    # mkdir_if_missing(figures_save_path)
    figures_save_path = f'./a_{dataset_name}_tsne.pdf'
    plt.xticks(np.arange(-100,101,50))  # 移除x轴刻度
    plt.yticks(np.arange(-100,101,25))  # 移除y轴刻度
    # plt.axis('off')  # 关闭整个坐标轴（包括轴线）
    plt.savefig(figures_save_path, dpi=600, bbox_inches='tight', format='pdf')
    # plt.show()
