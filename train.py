import time

import torch
import torch.optim as optim

from Modality_Mix import Modal_Mix, Reply_Modal_Mix
from loss import *
from tqdm import tqdm

from utils import AverageMeter, cosine_similarity


class Trainer:
    def __init__(self, args, learner_net, pid_decrease=0, fp16='f', old_learner_net=None, bic_train='f', num_classes=0):
        self.lr = args.lr
        self.args = args
        self.fp16 = True if fp16 == 't' else False
        self.learner_net = learner_net
        self.pid_decrease = pid_decrease
        self.old_learner_net = old_learner_net
        if self.old_learner_net is not None:
            old_learner_net.train()
        self.train_loader = None
        self.reply_loader = None
        self.reply_batch_idx = 0
        self.criterion_id = nn.CrossEntropyLoss().cuda()
        self.criterion_tri = OriTripletLoss(batch_size=args.batch_size * args.num_pos, margin=args.margin).cuda()
        self.criterion_kd = DistillationLoss(T=2.0, distype=args.kd_loss).cuda()

        if bic_train == 't':
            self.optimizer = optim.SGD(learner_net.bias_correct_list[-1].parameters(), lr=self.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
        elif self.args.method == 'ptkp':
            self.criterion_dcl = DomainConsistencyLoss().cuda()
            self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
            self.criterion_ce_weight = CrossEntropyLabelSmooth_weighted(num_classes).cuda()
            self.criterion_triple = SoftTripletLoss().cuda()
            self.criterion_triple_hard = HardTripletLoss().cuda()
            self.criterion_triple_weight = SoftTripletLoss_weighted().cuda()
            ignored_params = list(map(id, param) for param in self.learner_net.parameters() if param.requires_grad is False)
            base_params = filter(lambda p: id(p) not in ignored_params, self.learner_net.parameters())
            self.optimizer = optim.SGD([{'params': base_params, 'lr': 0.1 * self.lr}], weight_decay=5e-4, momentum=0.9, nesterov=True)
        elif self.args.method == 'krkc':
            self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
            self.criterion_triple = SoftTripletLoss().cuda()
            self.criterion_triple_hard = HardTripletLoss().cuda()
            ignored_params = list(map(id, param) for param in self.learner_net.parameters() if param.requires_grad is False)
            base_params = filter(lambda p: id(p) not in ignored_params, self.learner_net.parameters())
            self.optimizer = optim.SGD([{'params': base_params, 'lr': 0.1 * self.lr}], weight_decay=5e-4, momentum=0.9, nesterov=True)
            if old_learner_net is not None:
                ignored_params = list(map(id, param) for param in self.old_learner_net.parameters() if param.requires_grad is False)
                base_params = filter(lambda p: id(p) not in ignored_params, self.old_learner_net.parameters())
                self.old_optimizer = optim.SGD([{'params': base_params, 'lr': 0.001 * self.lr}], weight_decay=5e-4, momentum=0.9, nesterov=True)
        elif self.args.gcn == 't':
            ignored_params_gcn_besides = list(map(id, self.learner_net.gcn_layer.gcn_bn.parameters())) + list(map(id, self.learner_net.gcn_layer.gcn_classifier.parameters()))
            ignored_params_gcn = list(map(id, param) for param in self.learner_net.gcn_layer.parameters() if param.requires_grad is False) + ignored_params_gcn_besides
            base_params_gcn = filter(lambda p: id(p) not in ignored_params_gcn, self.learner_net.gcn_layer.parameters())
            other_params_gcn = filter(lambda p: id(p) in ignored_params_gcn_besides, self.learner_net.gcn_layer.parameters())
            self.gcn_optimizer = optim.SGD([{'params': base_params_gcn, 'lr': 0.1 * self.lr}, {'params': other_params_gcn, 'lr': self.lr}], weight_decay=5e-4, momentum=0.9, nesterov=True)
            ignored_params_besides = list(map(id, self.learner_net.bottleneck.parameters())) + list(map(id, self.learner_net.classifier.parameters())) + list(map(id, self.learner_net.gcn_layer.parameters()))
            ignored_params = list(map(id, param) for param in self.learner_net.parameters() if param.requires_grad is False) + ignored_params_besides + ignored_params_gcn
            base_params = filter(lambda p: id(p) not in ignored_params, self.learner_net.parameters())
            other_params = filter(lambda p: id(p) in ignored_params_besides, self.learner_net.parameters())
            self.optimizer = optim.SGD([{'params': base_params, 'lr': 0.1 * self.lr}, {'params': other_params, 'lr': self.lr}], weight_decay=5e-4, momentum=0.9, nesterov=True)
        elif self.args.method == 'aka':
            self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
            self.criterion_triple = PlasticityLoss(0.3, 'euclidean', False)
            self.reconstruction_criterion = torch.nn.L1Loss()
            ignored_params_gcn = list(map(id, param) for param in self.learner_net.gcn_layer.parameters() if param.requires_grad is False)
            base_params_gcn = filter(lambda p: id(p) not in ignored_params_gcn, self.learner_net.gcn_layer.parameters())
            self.gcn_optimizer = optim.SGD([{'params': base_params_gcn, 'lr': 0.1 * self.lr}], weight_decay=5e-4, momentum=0.9, nesterov=True)
            ignored_params_besides = list(map(id, self.learner_net.bottleneck.parameters())) + list(map(id, self.learner_net.classifier.parameters())) + list(map(id, self.learner_net.gcn_layer.parameters()))
            ignored_params = list(map(id, param) for param in self.learner_net.parameters() if param.requires_grad is False) + ignored_params_besides + ignored_params_gcn
            base_params = filter(lambda p: id(p) not in ignored_params, self.learner_net.parameters())
            other_params = filter(lambda p: id(p) in ignored_params_besides, self.learner_net.parameters())
            self.optimizer = optim.SGD([{'params': base_params, 'lr': 0.1 * self.lr}, {'params': other_params, 'lr': self.lr}], weight_decay=5e-4, momentum=0.9, nesterov=True)
        else:
            ignored_params_besides = list(map(id, self.learner_net.bottleneck.parameters())) + list(map(id, self.learner_net.classifier.parameters()))
            ignored_params = list(map(id, param) for param in self.learner_net.parameters() if param.requires_grad is False) + ignored_params_besides
            base_params = filter(lambda p: id(p) not in ignored_params, self.learner_net.parameters())
            other_params = filter(lambda p: id(p) in ignored_params_besides, self.learner_net.parameters())
            self.optimizer = optim.SGD([{'params': base_params, 'lr': 0.1 * self.lr}, {'params': other_params, 'lr': self.lr}], weight_decay=5e-4, momentum=0.9, nesterov=True)
        self.incorrect_count_dict = {} if args.reply_type == 'b+der' else None

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if epoch < 10:
            lr = self.lr * (epoch + 1) / 10
        elif 10 <= epoch < 20:
            lr = self.lr
        elif 20 <= epoch < 40:
            lr = self.lr * 0.1
        elif epoch >= 40:
            lr = self.lr * 0.01
        self.optimizer.param_groups[0]['lr'] = 0.1 * lr
        for i in range(len(self.optimizer.param_groups) - 1):
            self.optimizer.param_groups[i + 1]['lr'] = lr

        if self.args.gcn == 't' and self.args.method == 'pcb':
            self.gcn_optimizer.param_groups[0]['lr'] = 0.1 * lr
            for i in range(len(self.gcn_optimizer.param_groups) - 1):
                self.gcn_optimizer.param_groups[i + 1]['lr'] = lr

        return lr

    def caj_train(self, epoch, current_step):
        total = 0  # total number of images
        correct = 0
        end = time.time()
        self.learner_net.train()
        id_loss = AverageMeter()  # identification loss
        tri_loss = AverageMeter()  # triplet loss
        kd_loss = AverageMeter()  # knowledge distillation loss
        train_loss = AverageMeter()  # loss
        data_time = AverageMeter()  # data loading time
        batch_time = AverageMeter()  # batch processing time
        current_lr = self.adjust_learning_rate(epoch)  # adjust learning rate
        self.train_loader = tqdm(self.train_loader, total=len(self.train_loader), leave=True, position=0)

        for batch_idx, (input10, input1x, input2, label1, label2, cam1, cam2) in enumerate(self.train_loader):
            input1 = input10.cuda()
            # input10 = input10.cuda()
            input2 = input2.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()
            labels = torch.cat((label1, label2), 0).long()
            # input1 = Modal_Mix(input10, input2, label1, label2).cuda()
            # input2 = Modal_Mix(input2, input10, label2, label1).cuda()
            # input1 = torch.cat((input10, input11), 0).cuda()
            # labels = torch.cat((label1, label1, label2), 0).cuda()

            data_time.update(time.time() - end)
            if self.args.single_modal_test == 't':
                if self.args.test_mode == 'vtv':
                    labels = label1
                    # print('vtv modal')
                    feat, new_logit = self.learner_net(input1, input1, modal=1)
                elif self.args.test_mode == 'iti':
                    labels = label2
                    # print('iti modal')
                    feat, new_logit = self.learner_net(input2, input2, modal=2)
            else:
                # labels = torch.cat((label1, label2), 0).cuda()
                feat, new_logit = self.learner_net(input1, input2)
            # print('shape:', feat.shape, new_logit.shape, labels.shape)
            loss_id = self.criterion_id(new_logit, labels)
            loss_tri = self.criterion_tri(feat, labels)[0]
            loss = loss_id + loss_tri

            correct += new_logit.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
            acc = 100. * correct / total

            if self.old_learner_net is not None and self.reply_loader is not None and self.args.sample_reply == 't':
                reply_input10, reply_input11, reply_input2, reply_label1, reply_label2, reply_cam1, reply_cam2 = self.reply_loader[self.reply_batch_idx]
                self.reply_batch_idx = (self.reply_batch_idx + 1) % len(self.reply_loader)
                reply_input1 = reply_input10.cuda()
                reply_input2 = reply_input2.cuda()
                # reply_input1 = Reply_Modal_Mix(reply_input10, reply_input2, reply_label1, reply_label2).cuda()
                # reply_input2 = Reply_Modal_Mix(reply_input2, reply_input10, reply_label2, reply_label1).cuda()
                # reply_input1 = torch.cat((reply_input10, reply_input11), 0).cuda()
                reply_feat, reply_logit = self.learner_net(reply_input1, reply_input2)
                with torch.no_grad():
                    reply_old_feat, reply_old_logit = self.old_learner_net(reply_input1, reply_input2)
                loss_kd = self.criterion_kd(reply_feat, reply_old_feat, reply_logit, reply_old_logit)
                kd_loss.update(loss_kd.item(), 2 * reply_input2.size(0))
                loss += loss_kd

            elif self.old_learner_net is not None and self.args.use_kd == 't':
                with torch.no_grad():
                    old_feat, old_logit = self.old_learner_net(input1, input2)
                loss_kd = self.criterion_kd(feat, old_feat, new_logit, old_logit)
                kd_loss.update(loss_kd.item(), 2 * input2.size(0))
                loss += loss_kd

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            id_loss.update(loss_id.item(), 2 * input2.size(0))
            tri_loss.update(loss_tri.item(), 2 * input2.size(0))
            train_loss.update(loss.item(), 2 * input2.size(0))

            self.train_loader.set_description(
                f'Step[{current_step}]/Epoch[{epoch}] ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'TLoss: {tri_loss.val:.3f} ({tri_loss.avg:.3f}) ' f'KDLoss: {kd_loss.val:.3f} ({kd_loss.avg:.3f}) ' f'Accu: {acc :.2f} ' f'lr: {current_lr:.3f} ' 'Loading')
        batch_time.update(time.time() - end)
        print(
            f'Step[{current_step}]/Epoch[{epoch}] ' f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) ' f'lr:{current_lr:.3f} ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'TLoss: {tri_loss.val:.3f} ({tri_loss.avg:.3f}) ' f'KDLoss: {kd_loss.val:.3f} ({kd_loss.avg:.3f}) ' f'Accu: {acc:.2f}\n')

    def caj_gcn_train(self, epoch, current_step):
        total = 0  # total number of images
        correct = 0
        end = time.time()
        self.learner_net.train()
        id_loss = AverageMeter()  # identification loss
        tri_loss = AverageMeter()  # triplet loss
        kd_loss = AverageMeter()  # knowledge distillation loss
        train_loss = AverageMeter()  # loss
        data_time = AverageMeter()  # data loading time
        batch_time = AverageMeter()  # batch processing time
        current_lr = self.adjust_learning_rate(epoch)  # adjust learning rate
        self.train_loader = tqdm(self.train_loader, total=len(self.train_loader), leave=True, position=0)

        for batch_idx, (input10, input11, input2, label1, label2, cam1, cam2) in enumerate(self.train_loader):
            input1 = input10.cuda()
            # input10 = input10.cuda()
            input2 = input2.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()
            labels = torch.cat((label1, label2), 0).long()
            # labels = torch.cat((label1, label2), 0).cuda().long() - self.pid_decrease
            # labels = torch.cat((label1, label2), 0).long()
            # input1 = Modal_Mix(input10, input2, label1, label2).cuda()
            # input2 = Modal_Mix(input2, input10, label2, label1).cuda()
            # input1 = torch.cat((input10, input11), 0).cuda()
            # labels = torch.cat((label1, label1, label2), 0).cuda()

            cams = torch.cat((cam1, cam2), 0)
            data_time.update(time.time() - end)

            feat, new_logit, gcn_new_logit = self.learner_net(input1, input2, cams=cams, step=current_step)
            loss_id = self.criterion_id(new_logit, labels) + self.criterion_id(gcn_new_logit, labels)
            loss_tri = self.criterion_tri(feat, labels)[0]
            loss = loss_id + loss_tri

            correct += new_logit.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
            acc = 100. * correct / total

            if self.old_learner_net is not None and self.reply_loader is not None and self.args.sample_reply == 't':
                reply_input10, reply_input11, reply_input2, reply_label1, reply_label2, reply_cam1, reply_cam2 = self.reply_loader[self.reply_batch_idx]
                self.reply_batch_idx = (self.reply_batch_idx + 1) % len(self.reply_loader)
                # reply_input1 = reply_input10.cuda()
                reply_input1 = reply_input10.cuda()
                reply_input2 = reply_input2.cuda()
                reply_feat, reply_logit, _ = self.learner_net(reply_input1, reply_input2, step=current_step, reply='t')

                with torch.no_grad():
                    reply_old_feat, reply_old_logit, _ = self.old_learner_net(reply_input1, reply_input2, step=current_step - 1, reply='t')
                loss_kd = self.criterion_kd(reply_feat, reply_old_feat, reply_logit, reply_old_logit)
                kd_loss.update(loss_kd.item(), 3 * reply_input2.size(0))
                loss += loss_kd

            self.optimizer.zero_grad()
            self.gcn_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.gcn_optimizer.step()

            id_loss.update(loss_id.item(), 2 * input2.size(0))
            tri_loss.update(loss_tri.item(), 2 * input2.size(0))
            train_loss.update(loss.item(), 2 * input2.size(0))

            self.train_loader.set_description(
                f'Step[{current_step}]/Epoch[{epoch}] ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'TLoss: {tri_loss.val:.3f} ({tri_loss.avg:.3f}) ' f'KDLoss: {kd_loss.val:.3f} ({kd_loss.avg:.3f}) ' f'Accu: {acc :.2f} ' f'lr: {current_lr:.3f} ' 'Loading')
        batch_time.update(time.time() - end)
        print(
            f'Step[{current_step}]/Epoch[{epoch}] ' f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) ' f'lr:{current_lr:.3f} ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'TLoss: {tri_loss.val:.3f} ({tri_loss.avg:.3f}) ' f'KDLoss: {kd_loss.val:.3f} ({kd_loss.avg:.3f}) ' f'Accu: {acc:.2f}\n')

    def caj_bic_bias_train(self, epoch, current_step):
        total = 0  # total number of images
        correct = 0
        end = time.time()
        self.learner_net.train()
        id_loss = AverageMeter()  # identification loss
        train_loss = AverageMeter()  # loss
        data_time = AverageMeter()  # data loading time
        batch_time = AverageMeter()  # batch processing time
        current_lr = self.adjust_learning_rate(epoch)  # adjust learning rate
        self.train_loader = tqdm(self.train_loader, total=len(self.train_loader), leave=True, position=0)

        for batch_idx, (input10, input11, input2, label1, label2) in enumerate(self.train_loader):
            # print("input10", input10)
            input1 = input10.cuda()
            input2 = input2.cuda()
            labels = torch.cat((label1, label2), 0).cuda().long()
            # print("labels", labels)
            data_time.update(time.time() - end)

            _, new_logit = self.learner_net(input1, input2)
            new_logit = self.learner_net.bias_forward(new_logit, current_step)
            loss_id = self.criterion_id(new_logit, labels)
            loss = loss_id

            correct += new_logit.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
            acc = 100. * correct / total

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            id_loss.update(loss_id.item(), 2 * input2.size(0))
            train_loss.update(loss.item(), 2 * input2.size(0))

            self.train_loader.set_description(
                f'Step[{current_step}]/Epoch[{epoch}] ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'Accu: {acc :.2f} ' f'lr: {current_lr:.3f} ' 'Loading')
        batch_time.update(time.time() - end)
        print(
            f'Step[{current_step}]/Epoch[{epoch}] ' f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) ' f'lr:{current_lr:.3f} ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'Accu: {acc:.2f}\n')

    def caj_bic_train(self, epoch, current_step):
        total = 0  # total number of images
        correct = 0
        end = time.time()
        self.learner_net.train()
        id_loss = AverageMeter()  # identification loss
        tri_loss = AverageMeter()  # triplet loss
        kd_loss = AverageMeter()  # knowledge distillation loss
        train_loss = AverageMeter()  # loss
        data_time = AverageMeter()  # data loading time
        batch_time = AverageMeter()  # batch processing time
        current_lr = self.adjust_learning_rate(epoch)  # adjust learning rate
        self.train_loader = tqdm(self.train_loader, total=len(self.train_loader), leave=True, position=0)

        for batch_idx, (input10, input11, input2, label1, label2, cam1, cam2) in enumerate(self.train_loader):
            input1 = input10.cuda()
            input2 = input2.cuda()
            labels = torch.cat((label1, label2), 0).cuda().long() - self.pid_decrease
            data_time.update(time.time() - end)

            feat, new_logit = self.learner_net(input1, input2)
            if current_step > 0:
                new_logit = self.learner_net.bias_forward(new_logit, current_step - 1)
            loss_id = self.criterion_id(new_logit, labels)
            loss_tri = self.criterion_tri(feat, labels)[0]
            loss = loss_id + loss_tri

            correct += new_logit.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
            acc = 100. * correct / total

            if self.old_learner_net is not None:
                with torch.no_grad():
                    old_feat, old_logit = self.old_learner_net(input1, input2)
                loss_kd = self.criterion_kd(feat, old_feat, new_logit, old_logit)
                kd_loss.update(loss_kd.item(), 2 * input2.size(0))
                loss += loss_kd

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            id_loss.update(loss_id.item(), 2 * input2.size(0))
            tri_loss.update(loss_tri.item(), 2 * input2.size(0))
            train_loss.update(loss.item(), 2 * input2.size(0))

            self.train_loader.set_description(
                f'Step[{current_step}]/Epoch[{epoch}] ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'TLoss: {tri_loss.val:.3f} ({tri_loss.avg:.3f}) ' f'KDLoss: {kd_loss.val:.3f} ({kd_loss.avg:.3f}) ' f'Accu: {acc :.2f} ' f'lr: {current_lr:.3f} ' 'Loading')
        batch_time.update(time.time() - end)
        print(
            f'Step[{current_step}]/Epoch[{epoch}] ' f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) ' f'lr:{current_lr:.3f} ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'TLoss: {tri_loss.val:.3f} ({tri_loss.avg:.3f}) ' f'KDLoss: {kd_loss.val:.3f} ({kd_loss.avg:.3f}) ' f'Accu: {acc:.2f}\n')

    def caj_ptkp_train(self, epoch, current_step):
        total = 0  # total number of images
        correct = 0
        end = time.time()
        self.learner_net.train()
        id_loss = AverageMeter()  # identification loss
        tri_loss = AverageMeter()  # triplet loss
        kd_loss = AverageMeter()  # knowledge distillation loss
        train_loss = AverageMeter()  # loss
        data_time = AverageMeter()  # data loading time
        batch_time = AverageMeter()  # batch processing time
        current_lr = self.adjust_learning_rate(epoch)  # adjust learning rate
        self.train_loader = tqdm(self.train_loader, total=len(self.train_loader), leave=True, position=0)

        for batch_idx, (input10, input11, input2, label1, label2, cam1, cam2) in enumerate(self.train_loader):
            input1 = input10.cuda()
            input2 = input2.cuda()
            labels = torch.cat((label1, label2), 0).cuda().long() - self.pid_decrease
            data_time.update(time.time() - end)

            feat, new_logit, bn_feat, bn_feat_list = self.learner_net(input1, input2)

            if current_step == 0:
                loss_id = self.criterion_id(new_logit, labels)
                loss_tri = self.criterion_tri(feat, labels)[0]
            else:
                weight_list = []
                for j in range(current_step):
                    statistics_mean = self.old_learner_net.domain_bns.bns[j].running_mean.data.clone().unsqueeze(0)
                    weight_list.append(cosine_similarity(feat, statistics_mean).view(-1))
                temp = torch.mean(torch.stack(weight_list, dim=0), dim=0)
                weights = F.softmax(temp * 2, dim=0)
                loss_id = self.criterion_ce_weight(new_logit, labels, weights)
                loss_tri = self.criterion_triple_weight(feat, feat, labels, weights)

            loss = loss_id + loss_tri

            correct += new_logit.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
            acc = 100. * correct / total

            if self.old_learner_net is not None and self.reply_loader is not None and self.args.sample_reply == 't':
                reply_input10, reply_input11, reply_input2, reply_label1, reply_label2, reply_cam1, reply_cam2 = self.reply_loader[self.reply_batch_idx]
                self.reply_batch_idx = (self.reply_batch_idx + 1) % len(self.reply_loader)
                reply_input1 = reply_input10.cuda()
                reply_input2 = reply_input2.cuda()
                reply_labels = torch.cat((reply_label1, reply_label2), 0).cuda().long()

                reply_feat, reply_logit, reply_bn_feat, reply_bn_feat_list = self.learner_net(reply_input1, reply_input2, reply=True)

                # kd_tri
                loss += self.criterion_triple_hard(reply_feat, reply_labels)[0]

                with torch.no_grad():
                    reply_old_feat, reply_old_logit, reply_old_bn_feat, reply_old_bn_feat_list = self.old_learner_net(reply_input1, reply_input2, reply=True)

                # sce
                loss_kd = self.criterion_kd(reply_bn_feat, reply_old_bn_feat, reply_logit, reply_old_logit)
                kd_loss.update(loss_kd.item(), 3 * reply_input2.size(0))
                loss += loss_kd

                # PT-KD
                loss += self.PT_KD(reply_old_bn_feat_list[:current_step], reply_bn_feat_list[:current_step])

                # PT-ID
                loss += self.PT_ID(bn_feat_list, bn_feat, labels)

            # DCL Loss
            if epoch >= 10:
                loss += self.criterion_dcl(bn_feat, bn_feat_list, labels)

            id_loss.update(loss_id.item(), 2 * input2.size(0))
            tri_loss.update(loss_tri.item(), 2 * input2.size(0))
            train_loss.update(loss.item(), 2 * input2.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.learner_net.domain_bns.weight_clone(self.learner_net.bottleneck)

            self.train_loader.set_description(
                f'Step[{current_step}]/Epoch[{epoch}] ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'TLoss: {tri_loss.val:.3f} ({tri_loss.avg:.3f}) ' f'KDLoss: {kd_loss.val:.3f} ({kd_loss.avg:.3f}) ' f'Accu: {acc :.2f} ' f'lr: {current_lr:.3f} ' 'Loading')
        batch_time.update(time.time() - end)
        print(
            f'Step[{current_step}]/Epoch[{epoch}] ' f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) ' f'lr:{current_lr:.3f} ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'TLoss: {tri_loss.val:.3f} ({tri_loss.avg:.3f}) ' f'KDLoss: {kd_loss.val:.3f} ({kd_loss.avg:.3f}) ' f'Accu: {acc:.2f}\n')

    def caj_krkc_train(self, epoch, current_step):
        total = 0  # total number of images
        correct = 0
        end = time.time()
        self.learner_net.train()
        id_loss = AverageMeter()  # identification loss
        tri_loss = AverageMeter()  # triplet loss
        kd_loss = AverageMeter()  # knowledge distillation loss
        train_loss = AverageMeter()  # loss
        data_time = AverageMeter()  # data loading time
        batch_time = AverageMeter()  # batch processing time
        current_lr = self.adjust_learning_rate(epoch)  # adjust learning rate
        self.train_loader = tqdm(self.train_loader, total=len(self.train_loader), leave=True, position=0)

        for batch_idx, (input10, input11, input2, label1, label2, cam1, cam2) in enumerate(self.train_loader):
            input1 = input10.cuda()
            input2 = input2.cuda()
            labels = torch.cat((label1, label2), 0).cuda().long() - self.pid_decrease
            data_time.update(time.time() - end)

            feat, new_logit = self.learner_net(input1, input2)
            loss_id = self.criterion_id(new_logit, labels)
            loss_tri = self.criterion_tri(feat, labels)[0]
            # loss_tri = self.criterion_triple(feat, feat, labels)
            loss_rehearsal = loss_id + loss_tri

            correct += new_logit.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
            acc = 100. * correct / total

            if self.old_learner_net is not None and self.reply_loader is not None and self.args.sample_reply == 't':
                reply_input10, reply_input11, reply_input2, reply_label1, reply_label2, reply_cam1, reply_cam2 = self.reply_loader[self.reply_batch_idx]
                self.reply_batch_idx = (self.reply_batch_idx + 1) % len(self.reply_loader)
                reply_input1 = reply_input10.cuda()
                reply_input2 = reply_input2.cuda()
                reply_label = torch.cat((reply_label1, reply_label2), 0).cuda().long() - self.pid_decrease
                reply_feat, reply_logit = self.learner_net(reply_input1, reply_input2)

                loss_rehearsal += self.criterion_triple_hard(reply_feat, reply_label)[0]

                old_feat, old_logit = self.old_learner_net(input1, input2)
                reply_old_feat, reply_old_logit = self.old_learner_net(reply_input1, reply_input2)
                loss_refresh = self.criterion_id(old_logit, labels) + self.criterion_tri(old_feat, labels)[0] + self.criterion_triple_hard(reply_old_feat, reply_label)[0] + self.criterion_kd(None, None, old_logit, new_logit)
                kd_loss.update(loss_refresh.item(), 3 * reply_input2.size(0))
                self.old_optimizer.zero_grad()
                loss_refresh.backward()
                self.old_optimizer.step()

                loss_rehearsal += self.criterion_kd(None, None, new_logit, old_logit)

            self.optimizer.zero_grad()
            loss_rehearsal.backward()
            self.optimizer.step()

            id_loss.update(loss_id.item(), 2 * input2.size(0))
            tri_loss.update(loss_tri.item(), 2 * input2.size(0))
            train_loss.update(loss_rehearsal.item(), 2 * input2.size(0))

            self.train_loader.set_description(f'Step[{current_step}]/Epoch[{epoch}] ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'TLoss: {tri_loss.val:.3f} ({tri_loss.avg:.3f}) ' f'KDLoss: {kd_loss.val:.3f} ({kd_loss.avg:.3f}) ' f'Accu: {acc :.2f} ' f'lr: {current_lr:.3f} ' 'Loading')
        batch_time.update(time.time() - end)
        print(f'Step[{current_step}]/Epoch[{epoch}] ' f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) ' f'lr:{current_lr:.3f} ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'TLoss: {tri_loss.val:.3f} ({tri_loss.avg:.3f}) ' f'KDLoss: {kd_loss.val:.3f} ({kd_loss.avg:.3f}) ' f'Accu: {acc:.2f}\n')

    def caj_der_train(self, epoch, current_step):
        total = 0  # total number of images
        correct = 0
        end = time.time()
        self.learner_net.train()
        id_loss = AverageMeter()  # identification loss
        tri_loss = AverageMeter()  # triplet loss
        kd_loss = AverageMeter()  # knowledge distillation loss
        train_loss = AverageMeter()  # loss
        data_time = AverageMeter()  # data loading time
        batch_time = AverageMeter()  # batch processing time
        current_lr = self.adjust_learning_rate(epoch)  # adjust learning rate
        self.train_loader = tqdm(self.train_loader, total=len(self.train_loader), leave=True, position=0)

        for batch_idx, (input10, input11, input2, label1, label2, cam1, cam2) in enumerate(self.train_loader):
            input1 = input10.cuda()
            input2 = input2.cuda()
            labels = torch.cat((label1, label2), 0).cuda().long() - self.pid_decrease
            data_time.update(time.time() - end)

            feat, new_logit = self.learner_net(input1, input2)
            loss_id = self.criterion_id(new_logit, labels)
            loss_tri = self.criterion_tri(feat, labels)[0]
            loss = loss_id + loss_tri

            correct += new_logit.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
            acc = 100. * correct / total

            if self.reply_loader is not None and self.args.sample_reply == 't':
                reply_input1, reply_input2, reply_label, reply_old_logit = self.reply_loader.meta_data[self.reply_batch_idx]
                self.reply_batch_idx = (self.reply_batch_idx + 1) % len(self.reply_loader)
                reply_input1 = reply_input1.cuda()
                reply_input2 = reply_input2.cuda()
                reply_label = reply_label.cuda().long()
                reply_old_logit = reply_old_logit.cuda()

                _, reply_logit = self.learner_net(reply_input1, reply_input2)

                loss_kd = self.criterion_kd(None, None, reply_logit, reply_old_logit) + self.criterion_id(reply_logit, reply_label)
                kd_loss.update(loss_kd.item(), 3 * reply_input2.size(0))
                loss += loss_kd

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            id_loss.update(loss_id.item(), 2 * input2.size(0))
            tri_loss.update(loss_tri.item(), 2 * input2.size(0))
            train_loss.update(loss.item(), 2 * input2.size(0))

            self.train_loader.set_description(
                f'Step[{current_step}]/Epoch[{epoch}] ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'TLoss: {tri_loss.val:.3f} ({tri_loss.avg:.3f}) ' f'KDLoss: {kd_loss.val:.3f} ({kd_loss.avg:.3f}) ' f'Accu: {acc :.2f} ' f'lr: {current_lr:.3f} ' 'Loading')
        batch_time.update(time.time() - end)
        print(
            f'Step[{current_step}]/Epoch[{epoch}] ' f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) ' f'lr:{current_lr:.3f} ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'TLoss: {tri_loss.val:.3f} ({tri_loss.avg:.3f}) ' f'KDLoss: {kd_loss.val:.3f} ({kd_loss.avg:.3f}) ' f'Accu: {acc:.2f}\n')

    def caj_aka_train(self, epoch, current_step):
        total = 0  # total number of images
        correct = 0
        end = time.time()
        self.learner_net.train()
        id_loss = AverageMeter()  # identification loss
        tri_loss = AverageMeter()  # triplet loss
        kd_loss = AverageMeter()  # knowledge distillation loss
        train_loss = AverageMeter()  # loss
        data_time = AverageMeter()  # data loading time
        batch_time = AverageMeter()  # batch processing time
        current_lr = self.adjust_learning_rate(epoch)  # adjust learning rate
        self.train_loader = tqdm(self.train_loader, total=len(self.train_loader), leave=True, position=0)

        for batch_idx, (input10, input11, input2, label1, label2, cam1, cam2) in enumerate(self.train_loader):
            input1 = input10.cuda()
            input2 = input2.cuda()
            labels = torch.cat((label1, label2), 0).cuda().long() - self.pid_decrease
            data_time.update(time.time() - end)

            feat, new_logit, feat_gcn, corr = self.learner_net(input1, input2)
            loss_id = self.criterion_id(new_logit, labels)
            loss_tri = self.criterion_tri(feat, labels)[0]
            loss_tri2 = self.criterion_triple(feat_gcn, feat_gcn, feat_gcn, labels, labels, labels)
            # loss_tri = self.criterion_tri(feat, labels)[0]
            loss = loss_id + loss_tri + loss_tri2

            correct += new_logit.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
            acc = 100. * correct / total

            if self.old_learner_net is not None:
                with torch.no_grad():
                    reply_feat, reply_logit, _, _ = self.old_learner_net(input1, input2, reply='t')
                    reply_vertex = self.old_learner_net.gcn_layer.meta_graph_vertex
                loss_kd = self.criterion_kd(_, _, new_logit, reply_logit)
                kd_loss.update(loss_kd.item(), 2 * input2.size(0))
                loss += loss_kd
                loss += self.learner_net.gcn_layer.StabilityLoss(reply_vertex, self.learner_net.gcn_layer.meta_graph_vertex) * 0.0005

            self.optimizer.zero_grad()
            self.gcn_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.gcn_optimizer.step()

            id_loss.update(loss_id.item(), 2 * input2.size(0))
            tri_loss.update(loss_tri.item(), 2 * input2.size(0))
            train_loss.update(loss.item(), 2 * input2.size(0))

            self.train_loader.set_description(
                f'Step[{current_step}]/Epoch[{epoch}] ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'TLoss: {tri_loss.val:.3f} ({tri_loss.avg:.3f}) ' f'KDLoss: {kd_loss.val:.3f} ({kd_loss.avg:.3f}) ' f'Accu: {acc :.2f} ' f'lr: {current_lr:.3f} ' 'Loading')
        batch_time.update(time.time() - end)
        print(
            f'Step[{current_step}]/Epoch[{epoch}] ' f'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) ' f'lr:{current_lr:.3f} ' f'Loss: {train_loss.val:.3f} ({train_loss.avg:.3f}) ' f'iLoss: {id_loss.val:.3f} ({id_loss.avg:.3f}) ' f'TLoss: {tri_loss.val:.3f} ({tri_loss.avg:.3f}) ' f'KDLoss: {kd_loss.val:.3f} ({kd_loss.avg:.3f}) ' f'Accu: {acc:.2f}\n')

    def PT_KD(self, fake_feat_list_old, fake_feat_list_new):
        loss_cross = []
        L1 = torch.nn.L1Loss()
        for i in range(len(fake_feat_list_old)):
            for j in range(i, len(fake_feat_list_old)):
                old_simi_matrix = cosine_distance(fake_feat_list_old[i], fake_feat_list_old[i])
                new_simi_matrix = cosine_distance(fake_feat_list_new[j], fake_feat_list_new[j])
                simi_loss = L1(old_simi_matrix, new_simi_matrix)
                loss_cross.append(simi_loss)
        loss_cross = torch.mean(torch.stack(loss_cross))
        return loss_cross

    def PT_ID(self, feature_list_bn, bn_features, pids):

        loss = []
        for features in feature_list_bn:
            loss.append(self.criterion_tri(features, pids)[0])
        loss.append(self.criterion_tri(bn_features, pids)[0])
        loss = torch.mean(torch.stack(loss))

        loss_cross = []
        for i in range(len(feature_list_bn)):
            for j in range(i + 1, len(feature_list_bn)):
                loss_cross.append(self.criterion_triple_hard(feature_list_bn[i], pids, feature_list_bn[j]))
        loss_cross = torch.mean(torch.stack(loss_cross))
        loss = 0.5 * (loss + loss_cross)

        return loss




