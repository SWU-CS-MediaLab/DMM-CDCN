import torch
import time

from eval_metrics import *
from tqdm import tqdm
from utils import save_state, mkdir_if_missing


class Tester:
    def __init__(self, datasets, total_epoch, save_path, state, whether_save=True, net=None):
        net = net
        self.state = state
        self.best_acc = -1
        self.best_epoch = -1
        self.datasets = datasets
        self.total_epoch = total_epoch
        self.save_path = save_path
        self.whether_save = whether_save

    def ori_test(self, net, epoch, step, total_step, total_epoch):
        net.eval()
        start = time.time()
        ngall = self.datasets.ngall[step]
        nquery = self.datasets.nquery[step]
        dataset = self.datasets.names[step]
        gall_cam = self.datasets.gall_cam[step]
        gall_label = self.datasets.gall_label[step]
        query_cam = self.datasets.query_cam[step]
        test_mode = self.datasets.test_mode[step]
        gall_loader = self.datasets.gall_loader[step]
        query_label = self.datasets.query_label[step]
        query_loader = self.datasets.query_loader[step]

        ptr = 0
        gall_feat = np.zeros((ngall, 2048))
        gall_feat_att = np.zeros((ngall, 2048))
        gall_loader = tqdm(gall_loader, total=len(gall_loader), leave=True, position=0)
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = input.cuda()
                feat, feat_att = net(input, input, modal=test_mode[1])
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
                ptr = ptr + batch_num
                gall_loader.set_description('Extracting Gallery Feature')
        ptr = 0
        query_feat = np.zeros((nquery, 2048))
        query_feat_att = np.zeros((nquery, 2048))
        query_loader = tqdm(query_loader, total=len(query_loader), leave=True, position=0)

        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(query_loader):
                batch_num = input.size(0)
                input = input.cuda()
                feat, feat_att = net(input, input, modal=test_mode[0])
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
                ptr = ptr + batch_num
                query_loader.set_description('Extracting Query Feature')

        # compute the similarity
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

        # evaluation
        if dataset == 'regdb':
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
            cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, query_label, gall_label)
        elif dataset == 'sysu':
            cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
            cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
        elif dataset == 'llcm':
            cmc, mAP, mINP = eval_llcm(-distmat, query_label, gall_label, query_cam, gall_cam)
            cmc_att, mAP_att, mINP_att = eval_llcm(-distmat_att, query_label, gall_label, query_cam, gall_cam)
        elif dataset == 'vcm':
            cmc, mAP, mINP = eval_vcm(-distmat, query_label, gall_label, query_cam, gall_cam)
            cmc_att, mAP_att, mINP_att = eval_vcm(-distmat_att, query_label, gall_label, query_cam, gall_cam)

        print(f'Evaluation Time:\t {time.time() - start:.3f}')
        print(f'Step[{total_step}] Model [{self.state}], the result of dataset[{self.datasets.names[step]}]:')
        print('POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('BN: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))

        if step == total_step:
            if cmc_att[0] > self.best_acc or cmc[0] > self.best_acc:
                self.best_acc = max(cmc_att[0], cmc[0])
                self.best_epoch = epoch
                if self.whether_save:
                    model_save_path_best = self.save_path + '_' + self.datasets.names[total_step] + '_' + self.state + '_best.pth'
                    save_state(net.state_dict(), net.out_features, (epoch + 1) % total_epoch, total_step + (epoch + 1 == total_epoch), self.datasets.names, model_save_path_best)
                print('Best Epoch [{}] Pool[{:.4f}] BN[{:.4f}]'.format(self.best_epoch, cmc[0], cmc_att[0]))
            if epoch == total_epoch - 1:
                if self.whether_save:
                    model_save_path_final = self.save_path + '_' + self.datasets.names[total_step] + '_' + self.state + '_final.pth'
                    save_state(net.state_dict(), net.out_features, (epoch + 1) % total_epoch, total_step + (epoch + 1 == total_epoch), self.datasets.names, model_save_path_final)
            print(f'Step[{total_step}] Model [{self.state}]---Best Epoch[{self.best_epoch}] Best R1[{self.best_acc:.4f}]\n')
        if step == 0:
            model_save_path_final = self.save_path + '_' + self.datasets.names[total_step] + '_' + self.state + f'_epoch[{epoch + 1}].pth'
            save_state(net.state_dict(), net.out_features, (epoch + 1) % total_epoch, total_step + (epoch + 1 == total_epoch), self.datasets.names, model_save_path_final)
        return cmc[0], mAP, mINP, cmc_att[0], mAP_att, mINP_att

    def gcn_test(self, net, epoch, step, total_step, total_epoch):
        net.eval()
        start = time.time()
        ngall = self.datasets.ngall[step]
        nquery = self.datasets.nquery[step]
        dataset = self.datasets.names[step]
        gall_cam = self.datasets.gall_cam[step]
        gall_label = self.datasets.gall_label[step]
        query_cam = self.datasets.query_cam[step]
        test_mode = self.datasets.test_mode[step]
        gall_loader = self.datasets.gall_loader[step]
        query_label = self.datasets.query_label[step]
        query_loader = self.datasets.query_loader[step]

        ptr = 0
        gall_feat = np.zeros((ngall, 2048))
        gall_feat_att = np.zeros((ngall, 2048))
        gall_feat_gcn = np.zeros((ngall, 2048))
        gall_loader = tqdm(gall_loader, total=len(gall_loader), leave=True, position=0)
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(gall_loader):
                batch_num = input.size(0)
                input = input.cuda()
                feat, feat_att, feat_gcn = net(input, input, modal=test_mode[1], step=step, cams=gall_cam)
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
                gall_feat_gcn[ptr:ptr + batch_num, :] = feat_gcn.detach().cpu().numpy()
                ptr = ptr + batch_num
                gall_loader.set_description('Extracting Gallery Feature')

        ptr = 0
        query_feat = np.zeros((nquery, 2048))
        query_feat_att = np.zeros((nquery, 2048))
        query_feat_gcn = np.zeros((nquery, 2048))
        query_loader = tqdm(query_loader, total=len(query_loader), leave=True, position=0)
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(query_loader):
                batch_num = input.size(0)
                input = input.cuda()
                feat, feat_att, feat_gcn = net(input, input, modal=test_mode[0], step=step, cams=query_cam)
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
                query_feat_gcn[ptr:ptr + batch_num, :] = feat_gcn.detach().cpu().numpy()
                ptr = ptr + batch_num
                query_loader.set_description('Extracting Query Feature')

        # compute the similarity
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))
        distmat_gcn = np.matmul(query_feat_gcn, np.transpose(gall_feat_gcn))

        # evaluation
        if dataset == 'regdb':
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
            cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, query_label, gall_label)
            cmc_gcn, mAP_gcn, mINP_gcn = eval_regdb(-distmat_gcn, query_label, gall_label)
        elif dataset == 'sysu':
            cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
            cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
            cmc_gcn, mAP_gcn, mINP_gcn = eval_sysu(-distmat_gcn, query_label, gall_label, query_cam, gall_cam)
        elif dataset == 'llcm':
            cmc, mAP, mINP = eval_llcm(-distmat, query_label, gall_label, query_cam, gall_cam)
            cmc_att, mAP_att, mINP_att = eval_llcm(-distmat_att, query_label, gall_label, query_cam, gall_cam)
            cmc_gcn, mAP_gcn, mINP_gcn = eval_llcm(-distmat_gcn, query_label, gall_label, query_cam, gall_cam)
        elif dataset == 'vcm':
            cmc, mAP, mINP = eval_vcm(-distmat, query_label, gall_label, query_cam, gall_cam)
            cmc_att, mAP_att, mINP_att = eval_vcm(-distmat_att, query_label, gall_label, query_cam, gall_cam)
            cmc_gcn, mAP_gcn, mINP_gcn = eval_vcm(-distmat_gcn, query_label, gall_label, query_cam, gall_cam)

        print(f'Evaluation Time:\t {time.time() - start:.3f}')
        print(f'Step[{total_step}] Model [{self.state}], the result of dataset[{self.datasets.names[step]}]:')
        print('POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('BN: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('GCN:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(cmc_gcn[0], cmc_gcn[4], cmc_gcn[9], cmc_gcn[19], mAP_gcn, mINP_gcn))

        if step == total_step:
            if cmc_att[0] > self.best_acc or cmc[0] > self.best_acc or cmc_gcn[0] > self.best_acc:
                self.best_acc = max(cmc_att[0], cmc[0], cmc_gcn[0])
                self.best_epoch = epoch
                if self.whether_save:
                    model_save_path_best = self.save_path + '_' + self.datasets.names[total_step] + '_' + self.state + '_best.pth'
                    save_state(net.state_dict(), net.out_features, (epoch + 1) % total_epoch, total_step + (epoch + 1 == total_epoch), self.datasets.names, model_save_path_best)
                print('Best Epoch [{}] Pool[{:.4f}] BN[{:.4f}]'.format(self.best_epoch, cmc[0], cmc_att[0]))
            if epoch == total_epoch - 1:
                if self.whether_save:
                    model_save_path_final = self.save_path + '_' + self.datasets.names[total_step] + '_' + self.state + '_final.pth'
                    save_state(net.state_dict(), net.out_features, (epoch + 1) % total_epoch, total_step + (epoch + 1 == total_epoch), self.datasets.names, model_save_path_final)
            print(f'Step[{total_step}] Model [{self.state}]---Best Epoch[{self.best_epoch}] Best R1[{self.best_acc:.4f}]\n')
        if step == 0:
            model_save_path_final = self.save_path + '_' + self.datasets.names[total_step] + '_' + self.state + f'_epoch[{epoch + 1}].pth'
            save_state(net.state_dict(), net.out_features, (epoch + 1) % total_epoch, total_step + (epoch + 1 == total_epoch), self.datasets.names, model_save_path_final)

        return cmc[0], mAP, mINP, cmc_att[0], mAP_att, mINP_att, cmc_gcn[0], mAP_gcn, mINP_gcn

    def acc_reset(self):
        self.best_acc = -1
        self.best_epoch = -1
