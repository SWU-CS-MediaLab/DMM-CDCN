import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import cosine_distance, euclidean_dist, _batch_mid_hard, _batch_hard, tensor_cosine_dist, tensor_euclidean_dist


# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def PT_KD(self, fake_feat_list_old, fake_feat_list_new):
    loss_cross = []
    for i in range(len(fake_feat_list_old)):
        for j in range(i, len(fake_feat_list_old)):
            loss_cross.append(self.loss_kd_L1(fake_feat_list_old[i], fake_feat_list_new[j]))
    loss_cross = torch.mean(torch.stack(loss_cross))
    return loss_cross


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    # dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.addmm_(mat1=emb1, mat2=emb2.t(), beta=1, alpha=-2)
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis=1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis=1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx


class CrossEntropyLabelSmooth_weighted(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth_weighted, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets, weights):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = torch.sum(torch.sum((- targets * log_probs), dim=1).view(-1) * weights)
        return loss


class DomainConsistencyLoss(nn.Module):
    def __init__(self):
        super(DomainConsistencyLoss, self).__init__()

    @staticmethod
    def forward(feats, feats_list, pids):
        loss = []
        pids = copy.deepcopy(pids).detach().cpu().numpy()
        uniq_pid = np.unique(pids)
        for pid in uniq_pid:
            pid_index = np.where(pid == pids)[0]
            pid_index = torch.from_numpy(pid_index).cuda()
            # print(pid_index)
            global_bn_feat_single = torch.index_select(feats, 0, pid_index)
            for feat in feats_list:
                specific_bn_feat_single = torch.index_select(feat, 0, pid_index)
                distance_matrix = -torch.mm(F.normalize(global_bn_feat_single, p=2, dim=1), F.normalize(specific_bn_feat_single, p=2, dim=1).t().detach())
                loss.append(torch.mean(distance_matrix))
        loss = torch.mean(torch.stack(loss))
        return loss


class DistillationLoss(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, T: float, distype='kl'):
        super(DistillationLoss, self).__init__()
        self.T = 1
        self.distype = distype

    def forward(self, new_features=None, old_features=None, new_logits=None, old_logits=None):

        if self.distype == 'kl':
            old_logits = old_logits.detach()
            new_logits = new_logits[:, :old_logits.size(1)]
            log_softmax = nn.LogSoftmax(dim=1).cuda()
            distillation_loss = (-F.softmax(old_logits, dim=1) * log_softmax(new_logits)).mean(0).sum()  # --> cross entropy loss
            return distillation_loss

        elif self.distype == 'l1':
            old_logits = old_logits.detach()
            new_logits = new_logits[:, :old_logits.size(1)]
            log_softmax = nn.LogSoftmax(dim=1).cuda()
            loss_ke_ce = (- F.softmax(old_logits, dim=1) * log_softmax(new_logits)).mean(0).sum()
            if new_features is not None and old_features is not None:
                L1 = torch.nn.L1Loss()
                old_simi_matrix = cosine_distance(old_features, old_features)
                new_simi_matrix = cosine_distance(new_features, new_features)
                simi_loss = L1(old_simi_matrix, new_simi_matrix)
                loss_ke_ce += simi_loss

            return loss_ke_ce

        elif self.distype == 'icarl':
            old_logits = old_logits.detach()
            new_logits = new_logits[:, :old_logits.size(1)]
            log_softmax = nn.LogSoftmax(dim=1).cuda()
            loss_ke_ce = (- F.softmax(old_logits / self.T, dim=1).detach() * log_softmax(new_logits / self.T)).mean(0).sum()
            return loss_ke_ce * self.T * self.T

        elif self.distype == 'l2':
            old_logits = old_logits.detach()
            new_logits = new_logits[:, :old_logits.size(1)]
            L2 = torch.nn.MSELoss()
            loss_ke_ce = L2(old_logits, new_logits)
            return loss_ke_ce

        elif self.distype == 'js':
            old_logits = old_logits.detach()
            p_s = F.log_softmax((new_logits + old_logits) / (2 * self.T), dim=1)
            p_t = F.softmax(old_logits / self.T, dim=1)
            p_t2 = F.softmax(new_logits / self.T, dim=1)
            loss_js = 0.5 * F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2) + 0.5 * F.kl_div(p_s, p_t2, reduction='batchmean') * (self.T ** 2)
            return loss_js

        elif self.distype == 'aka':
            """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

                  Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
                  'Hyperparameter': temperature"""
            log_scores_norm = F.log_softmax(new_logits / self.T, dim=1)
            targets_norm = F.softmax(old_logits/ self.T, dim=1)

            # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
            n = new_logits.size(1)
            if n > old_logits.size(1):
                n_batch = new_logits.size(0)
                zeros_to_add = torch.zeros(n_batch, n - old_logits.size(1))
                zeros_to_add = zeros_to_add.cuda()
                targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)
            # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
            KD_loss_unnorm = -(targets_norm * log_scores_norm)
            KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)  # --> sum over classes
            KD_loss_unnorm = KD_loss_unnorm.mean()  # --> average over batch
            # normalize
            KD_loss = KD_loss_unnorm * self.T ** 2
            return KD_loss


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.batch_size = batch_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class WRTTripletLoss(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(WRTTripletLoss, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct


class HardTripletLoss(nn.Module):
    '''
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    '''

    def __init__(self, margin=0.3, normalize_feature=False, mid_hard=False):
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()
        self.mid_hard = mid_hard

    def forward(self, emb, label, emb_=None):
        if emb_ is None:
            mat_dist = euclidean_dist(emb, emb)
            # mat_dist = cosine_dist(emb, emb)
            assert mat_dist.size(0) == mat_dist.size(1)
            N = mat_dist.size(0)
            mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
            if self.mid_hard:
                dist_ap, dist_an = _batch_mid_hard(mat_dist, mat_sim)
            else:
                dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
            assert dist_an.size(0) == dist_ap.size(0)
            y = torch.ones_like(dist_ap)
            loss = self.margin_loss(dist_an, dist_ap, y)
            prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
            return loss, prec
        else:
            mat_dist = euclidean_dist(emb, emb_)
            N = mat_dist.size(0)
            mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
            dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
            y = torch.ones_like(dist_ap)
            loss = self.margin_loss(dist_an, dist_ap, y)
            return loss


class SoftTripletLoss(nn.Module):
    def __init__(self, margin=0.3, normalize_feature=False, mid_hard=False):
        super(SoftTripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.mid_hard = mid_hard

    def forward(self, emb1, emb2, label):
        if self.normalize_feature:
            # equal to cosine similarity
            emb1 = F.normalize(emb1)
            emb2 = F.normalize(emb2)

        mat_dist = euclidean_dist(emb1, emb1)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
        if self.mid_hard:
            dist_ap, dist_an, ap_idx, an_idx = _batch_mid_hard(mat_dist, mat_sim, indice=True)
        else:
            dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
        assert dist_an.size(0) == dist_ap.size(0)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = F.log_softmax(triple_dist, dim=1)
        if (self.margin is not None):
            loss = (- self.margin * triple_dist[:, 0] - (1 - self.margin) * triple_dist[:, 1]).mean()
            return loss

        mat_dist_ref = euclidean_dist(emb2, emb2)
        dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N, 1).expand(N, N))[:, 0]
        dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N, 1).expand(N, N))[:, 0]
        triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
        triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

        loss = (- triple_dist_ref * triple_dist).mean(0).sum()
        return loss


class SoftTripletLoss_weighted(nn.Module):
    def __init__(self, margin=0.3, normalize_feature=False, mid_hard=False):
        super(SoftTripletLoss_weighted, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.mid_hard = mid_hard

    def forward(self, emb1, emb2, label, weights):
        if self.normalize_feature:
            # equal to cosine similarity
            emb1 = F.normalize(emb1)
            emb2 = F.normalize(emb2)

        mat_dist = euclidean_dist(emb1, emb1)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
        if self.mid_hard:
            dist_ap, dist_an, ap_idx, an_idx = _batch_mid_hard(mat_dist, mat_sim, indice=True)
        else:
            dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
        assert dist_an.size(0) == dist_ap.size(0)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = F.log_softmax(triple_dist, dim=1)
        if (self.margin is not None):
            loss = torch.sum((- self.margin * triple_dist[:, 0] - (1 - self.margin) * triple_dist[:, 1]) * weights)
            return loss

        mat_dist_ref = euclidean_dist(emb2, emb2)
        dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N, 1).expand(N, N))[:, 0]
        dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N, 1).expand(N, N))[:, 0]
        triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
        triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

        loss = torch.sum((- torch.sum((triple_dist_ref * triple_dist), dim=1)).view(-1) * weights)
        return loss


class CrossEntropyLabelSmooth_weighted(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth_weighted, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets, weights):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = torch.sum(torch.sum((- targets * log_probs), dim=1).view(-1) * weights)
        return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class RankingLoss:
	def __init__(self):
		pass

	def _label2similarity(sekf, label1, label2):
		'''
		compute similarity matrix of label1 and label2
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [n]
		:return: torch.Tensor, [m, n], {0, 1}
		'''
		m, n = len(label1), len(label2)
		l1 = label1.view(m, 1).expand([m, n])
		l2 = label2.view(n, 1).expand([n, m]).t()
		similarity = l1 == l2
		return similarity

	def _batch_hard(self, mat_distance, mat_similarity, more_similar):

		if more_similar is 'smaller':
			sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,descending=True)
			hard_p = sorted_mat_distance[:, 0]
			sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
			hard_n = sorted_mat_distance[:, 0]
			return hard_p, hard_n

		elif more_similar is 'larger':
			sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1, descending=False)
			hard_p = sorted_mat_distance[:, 0]
			sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
			hard_n = sorted_mat_distance[:, 0]
			return hard_p, hard_n


class PlasticityLoss(RankingLoss):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, margin, metric, if_l2='euclidean'):
		'''
		:param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
		:param bh: batch hard
		:param metric: l2 distance or cosine distance
		'''
		self.margin = margin
		self.margin_loss = nn.MarginRankingLoss(margin=margin)
		self.metric = metric
		self.if_l2 = if_l2

	def __call__(self, emb1, emb2, emb3, label1, label2, label3):
		'''

		:param emb1: torch.Tensor, [m, dim]
		:param emb2: torch.Tensor, [n, dim]
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [b]
		:return:
		'''

		if self.metric == 'cosine':
			mat_dist = tensor_cosine_dist(emb1, emb2)
			mat_dist = torch.log(1 + torch.exp(mat_dist))
			mat_sim = self._label2similarity(label1, label2)
			hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

			mat_dist = tensor_cosine_dist(emb1, emb3)
			mat_dist = torch.log(1 + torch.exp(mat_dist))
			mat_sim = self._label2similarity(label1, label3)
			_, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

			margin_label = -torch.ones_like(hard_p)

		elif self.metric == 'euclidean':
			if self.if_l2:
				emb1 = F.normalize(emb1)
				emb2 = F.normalize(emb2)
			mat_dist = tensor_euclidean_dist(emb1, emb2)
			mat_dist = torch.log(1 + torch.exp(mat_dist))
			mat_sim = self._label2similarity(label1, label2)
			hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

			mat_dist = tensor_euclidean_dist(emb1, emb3)
			mat_dist = torch.log(1 + torch.exp(mat_dist))
			mat_sim = self._label2similarity(label1, label3)
			_, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

			margin_label = torch.ones_like(hard_p)

		return self.margin_loss(hard_n, hard_p, margin_label)