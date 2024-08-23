from __future__ import print_function, absolute_import

import datetime
import torch.backends.cudnn as cudnn
from incremental_data_loader import *
from loss import *
from resnet50 import CAJ_ResNet50, CAJ_GCN_ResNet50, PTKP_ResNet50, AKA_ResNet50
from samplers import IdentitySampler, ReplyIdentitySampler, BiCIdentitySampler
from test import Tester
from train import Trainer

# ——————————————————————————————————————————————————————————
# basic settings
args = get_config()
device = torch.device(f'cuda:{args.gpu}')
torch.cuda.set_device(device)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)
cudnn.benchmark = True
set_seed(args.seed)
start_epoch = 0
start_step = 0
if args.debug == 'f':
    date = args.ex_name + '_' + datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    log_path = os.path.join(args.log_path, '{}/'.format(date))
    total_epoch = args.total_epoch
    model_save_path = args.model_path
    vis_log_path = args.vis_log_path
else:
    log_path = './debug/'
    total_epoch = 1
    model_save_path = './debug/'
    vis_log_path = './debug/'
suffix = 'p{}_n{}_lr{}_seed{}_trial{}'.format(args.num_pos, args.batch_size, args.lr, args.seed, args.trial)
sys.stdout = Logger(log_path + suffix + '_os.txt')
mkdir_if_missing(log_path)
mkdir_if_missing(model_save_path)
mkdir_if_missing(vis_log_path)

print("====================\nArgs:{}\n====================".format(args))
# load model
if args.method == 'der':
    replyset = CAJ_Replay_DER_Dataset(args.method)
elif args.method == 'pcb':
    replyset = CAJ_Replay_PCB_Dataset(args.method, args=args)
else:
    replyset = CAJ_Replay_Dataset(args.method)

if args.resume != '':
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['next_epoch']
    start_step = checkpoint['next_step']

    if args.next_step == 't':
        start_step += 1
        start_epoch = 0
    if args.joint_learning == 't' or args.tsne == 't':
        start_step = 4
    print('Start Step:', start_step)
    test_dataset = []
    # tt = ['regdb', 'sysu', 'llcm', 'vcm', 'regdb']
    # for i in range(4):
    #     test_dataset.append(tt[i])
    datasets = Incremental_Datasets(args.train_datasets) if args.debug == 'f' and args.test_noly == 'f' else Incremental_Datasets(args.test_datasets)
    # datasets = Incremental_Datasets(test_dataset) if args.debug == 'f' else Incremental_Datasets(test_dataset)
    if args.method == 'ptkp':
        learner_net = PTKP_ResNet50(checkpoint['out_features'], device).cuda()
    elif args.gcn == 't':
        learner_net = CAJ_GCN_ResNet50(checkpoint['out_features'], device, rgb_cams=datasets.rgb_cams[0], ir_cams=datasets.ir_cams[0]).cuda()
    elif args.method == 'aka':
        learner_net = AKA_ResNet50(checkpoint['out_features'], device).cuda()
    else:
        learner_net = CAJ_ResNet50(checkpoint['out_features'], device, args.method).cuda()
    # learner_net = CAJ_ResNet50(checkpoint['out_features'], device).cuda() if args.gcn == 'f' else CAJ_GCN_ResNet50(checkpoint['out_features'], device, datasets.rgb_cams[0], datasets.ir_cams[0]).cuda()

    if args.method == 'bic':
        for i in range(start_step):
            learner_net.bic_increase_step()
            learner_net.increase_step(datasets.nclass[i + 1])
    if args.gcn == 't':
        increase_step = start_step
        for i in range(increase_step - 1):
            learner_net.gcn_layer.adj_increase_step(rgb_cams=datasets.rgb_cams[i + 1], ir_cams=datasets.ir_cams[i + 1])
    if args.method == 'ptkp':
        for i in range(start_step - 1):
            learner_net.domain_bns.increase_step()
    learner_net.load_state_dict(checkpoint['state_dict'])
    if start_step > 0:
        old_learner_net = copy.deepcopy(learner_net).cuda()
        old_learner_net.frozen_all()

    if args.tmp_net_path != '':
        tmp_net_path = args.tmp_net_path
        for step in range(len(tmp_net_path)):
            tmp_checkpoint = torch.load(tmp_net_path[step])
            # print('out_features', tmp_checkpoint['out_features'])
            if args.gcn != 't':
                tmp_net = CAJ_ResNet50(tmp_checkpoint['out_features'], device, args.method).cuda()
            else:
                tmp_net = CAJ_GCN_ResNet50(tmp_checkpoint['out_features'], device, rgb_cams=datasets.rgb_cams[0], ir_cams=datasets.ir_cams[0]).cuda()
                for i in range(step):
                    tmp_net.gcn_layer.adj_increase_step(rgb_cams=datasets.rgb_cams[i + 1], ir_cams=datasets.ir_cams[i + 1])
            # tmp_net = CAJ_ResNet50(datasets.nclass[0], device, args.method).cuda()
                    # tmp_net.gcn_layer.adj_increase_step()
            if args.method == 'bic':
                for i in range(step):
                    tmp_net.bic_increase_step()
                    tmp_net.increase_step(datasets.nclass[i + 1])
            tmp_net.load_state_dict(tmp_checkpoint['state_dict'])
            replyset.increase_reply_data(tmp_net, datasets.trainset[step], step, args.gcn, args.reply_type, pid_increase=datasets.pid_increment[step])
            datasets.trainset[step] = None
            datasets.rgb_pos[step] = None
            datasets.infrared_pos[step] = None
            del tmp_net
    print(f'Resume the learner model from [{args.resume}]')
else:
    datasets = Incremental_Datasets(args.train_datasets) if args.debug == 'f' or args.memory_report == 't'  else Incremental_Datasets(['regdb', 'llcm', 'llcm'])
    if args.method == 'ptkp':
        learner_net = PTKP_ResNet50(datasets.nclass[0], device).cuda()
    elif args.gcn == 't':
        learner_net = CAJ_GCN_ResNet50(datasets.nclass[0], device, rgb_cams=datasets.rgb_cams[0], ir_cams=datasets.ir_cams[0]).cuda()
    elif args.method == 'aka':
        learner_net = AKA_ResNet50(datasets.nclass[0], device).cuda()
    else:
        learner_net = CAJ_ResNet50(datasets.nclass[0], device, args.method).cuda()

if args.tsne == 't':
    sne_feature = []
    sne_label = []
    sne_cam = []
    sne_domian = []
    learner_net.eval()
    for idx in range(len(datasets.names)):
        snedataset = SNEDatasets(datasets.trainset[idx], 'rgb', datasets.transform_test)
        sne_loader = DataLoader(snedataset, batch_size=128, shuffle=False, num_workers=args.workers, drop_last=False)
        with torch.no_grad():
            for inputs, targets, cams in sne_loader:
                inputs = inputs.to(device)
                if args.gcn == 't':
                    _, feat, _ = learner_net(inputs, inputs, modal=1, reply='t')
                else:
                    feat, _ = learner_net(inputs, inputs, modal=1, reply='t')
                sne_feature.extend(feat)
                sne_label.extend(targets)
                sne_cam.extend(cams)
                domain = [idx for _ in range(len(targets))]
                sne_domian.extend(domain)
        snedataset = SNEDatasets(datasets.trainset[idx], 'ir', datasets.transform_test)
        sne_loader = DataLoader(snedataset, batch_size=128, shuffle=False, num_workers=args.workers, drop_last=False)
        with torch.no_grad():
            for (inputs, targets, cams) in sne_loader:
                inputs = inputs.to(device)
                if args.gcn == 't':
                    _, feat, _ = learner_net(inputs, inputs, modal=2, reply='t')
                else:
                    feat, _ = learner_net(inputs, inputs, modal=2, reply='t')
                sne_feature.extend(feat)
                sne_label.extend(targets)
                sne_cam.extend(cams)
                domain = [idx  for _ in range(len(targets))]
                sne_domian.extend(domain)
    sne_feature = torch.stack(sne_feature).cpu().numpy()
    sne_domian = np.array(sne_domian)
    T_SNE(sne_feature, sne_domian, datasets.names[idx])
    exit(0)

learner_tester = Tester(datasets, total_epoch, log_path + suffix, 'Guider_GPU' + str(args.gpu)) if args.gcn == 'f' else Tester(datasets, total_epoch, log_path + suffix, 'Guider_GCN_GPU' + str(args.gpu))
if args.resume != '':
    if args.domain_gap_test == 'f' and args.test_mode == 'vti':
        for step in range(start_step):
            if args.gcn == 'f' and args.method != 'aka':
                learner_tester.ori_test(learner_net, 0, step, start_step - 1, total_epoch)
            else:
                learner_tester.gcn_test(learner_net, 0, step, start_step - 1, total_epoch)
    else:
        for step in range(len(datasets.names)):
            if args.gcn == 'f' and args.method != 'aka':
                learner_tester.ori_test(learner_net, 0, step, start_step - 1, total_epoch)
            else:
                learner_tester.gcn_test(learner_net, 0, step, start_step - 1, total_epoch)
if args.test_only == 't':
    exit(0)

# training
if args.memory_report == 't':
    usage_memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f'Berfore Before Before train------Memory Usage:{usage_memory:.2f} MB')
    print()
print('==> Start Training...')
for current_step in range(start_step, len(datasets.names)):
    if args.memory_report == 't':
        usage_memory = psutil.Process().memory_info().rss / 1024 / 1024
        print(f'Before Before train Step[{current_step}]------Memory Usage:{usage_memory:.2f} MB')
        print()
    if args.joint_learning == 't' and current_step > 0: break

    datasets.dataset_info(current_step)
    learner_tester.acc_reset()

    print(f'Train the learner model in [{current_step}]')
    if current_step > 0:
        if args.gcn == 'f':
            learner_net.increase_step(datasets.nclass[current_step])
        else:
            learner_net.increase_step(datasets.nclass[current_step], rgb_cams=datasets.rgb_cams[current_step], ir_cams=datasets.ir_cams[current_step])
        learner_net.cuda()
        if args.method == 'krkc':
            old_learner_net = copy.deepcopy(learner_net).cuda()
            old_learner_net.train()
        if args.weight_align == 't':
            learner_net.classifier.weight_align(learner_net.out_features - old_learner_net.out_features)
        learner_trainer = Trainer(args, learner_net, fp16=args.fp_16, old_learner_net=old_learner_net, num_classes=learner_net.out_features)
    else:
        learner_trainer = Trainer(args, learner_net, fp16=args.fp_16, num_classes=learner_net.out_features)

    # train

    for epoch in range(start_epoch, total_epoch):
        if args.memory_report == 't':
            usage_memory = psutil.Process().memory_info().rss / 1024 / 1024
            print(f'Before train Step[{current_step}]------Memory Usage:{usage_memory:.2f} MB')
            print()
        learner_trainer.train_loader = DataLoader(datasets.trainset[current_step], batch_size=args.batch_size * args.num_pos, sampler=IdentitySampler(datasets.trainset[current_step], datasets.rgb_pos[current_step], datasets.infrared_pos[current_step], args.num_pos, args.batch_size, datasets.pid_increment[current_step]), num_workers=args.workers, drop_last=True, pin_memory=False)
        if current_step > 0:
            if args.sample_reply == 't':
                if args.method != 'der':
                    learner_trainer.reply_loader = list(DataLoader(replyset, batch_size=args.batch_size * 2, sampler=ReplyIdentitySampler(replyset, 2, args.batch_size * 2, reply_type=args.reply_type, length=len(learner_trainer.train_loader)), num_workers=args.workers, drop_last=True))
                    # print(len(learner_trainer.reply_loader))
                else:
                    learner_trainer.reply_loader = replyset
                learner_trainer.reply_batch_idx = 0
        if args.gcn == 't':
            learner_trainer.caj_gcn_train(epoch, current_step)
        elif args.method == 'bic':
            learner_trainer.caj_bic_train(epoch, current_step)
        elif args.method == 'ptkp':
            learner_trainer.caj_ptkp_train(epoch, current_step)
        elif args.method == 'krkc':
            learner_trainer.caj_krkc_train(epoch, current_step)
        elif args.method == 'der':
            learner_trainer.caj_der_train(epoch, current_step)
        elif args.method == 'aka':
            learner_trainer.caj_aka_train(epoch, current_step)
        else:
            learner_trainer.caj_train(epoch, current_step)
        if args.memory_report == 't':
            usage_memory = psutil.Process().memory_info().rss / 1024 / 1024
            print(f'After train Step[{current_step}]------Memory Usage:{usage_memory:.2f} MB')
            print()

        # test
        if args.test_frequency > 0 and epoch > 28 and (((epoch + 1) % args.test_frequency == 0) or ((epoch + 1) == total_epoch)):
            test_result = []
            if args.joint_learning == 't' or args.domain_gap_test == 't':
                for step in range(len(datasets.names)):
                    if args.gcn == 'f' and args.method != 'aka':
                        test_result.append(learner_tester.ori_test(learner_net, epoch, step, len(datasets.names) - 1, total_epoch))
                    else:
                        test_result.append(learner_tester.gcn_test(learner_net, epoch, step, len(datasets.names) - 1, total_epoch))
            else:
                for step in range(current_step + 1):
                    if args.gcn == 'f' and args.method != 'aka':
                        test_result.append(learner_tester.ori_test(learner_net, epoch, step, current_step, total_epoch))
                    else:
                        test_result.append(learner_tester.gcn_test(learner_net, epoch, step, current_step, total_epoch))
    # sample reply and bic sample
    if args.sample_reply == 't' and current_step < len(datasets.names) - 1 and args.joint_learning == 'f':
        replyset.increase_reply_data(learner_net, datasets.trainset[current_step], current_step, args.gcn, args.reply_type, pid_increase=datasets.pid_increment[current_step])

    if args.method == 'bic':
        learner_net.bic_increase_step()
        learner_net.cuda()
        if current_step > 0:
            bic_trainer = Trainer(args, learner_net, bic_train='t')
            bic_dataset = CAJBiCTrainData(replyset.val_rgb_img, replyset.val_ir_img, replyset.val_rgb_label, replyset.val_ir_label, [replyset.transform_rgb1, replyset.transform_rgb2, replyset.transform_ir])
            bic_trainer.train_loader = DataLoader(bic_dataset, batch_size=args.batch_size * 2, sampler=BiCIdentitySampler(bic_dataset, 2, args.batch_size), num_workers=args.workers, drop_last=True)
            for epoch in range(total_epoch):
                bic_trainer.caj_bic_bias_train(epoch, current_step)
        learner_net.bias_correct_list[-1].frozen_all()

    # old net
    start_epoch = 0
    if args.method != 'krkc':
        old_learner_net = copy.deepcopy(learner_net).cuda()
        old_learner_net.frozen_all()
        old_learner_net.train()
    if args.fp_16 == 't':
        learner_net = learner_net.float()
    if args.domain_gap_test == 't':
        break
    # datasets.trainset[current_step] = None
    torch.cuda.empty_cache()
    if args.memory_report == 't':
        usage_memory = psutil.Process().memory_info().rss / 1024 / 1024
        print(f'After After train Step[{current_step}]------Memory Usage:{usage_memory:.2f} MB')
        print()


