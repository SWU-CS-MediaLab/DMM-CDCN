from __future__ import print_function
import psutil
from torch.utils.data import DataLoader
from Apreprocess.ChannelAug import *
from train_data_loader import *
from test_data_loader import *
from train_data_loader_caj import *
from utils import *
from config import get_config


class Incremental_Datasets(object):
    def __init__(self, datasets):
        args = get_config()
        print('==> Loading data..')

        # define transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if args.caj == 't':
            self.transform_train = [
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((args.img_h, args.img_w), interpolation=InterpolationMode.LANCZOS),
                    transforms.Pad(10),
                    transforms.RandomCrop((args.img_h, args.img_w)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
                ),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((args.img_h, args.img_w), interpolation=InterpolationMode.LANCZOS),
                    transforms.ToTensor(),
                    normalize,
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((args.img_h, args.img_w), interpolation=InterpolationMode.LANCZOS),
                    transforms.Pad(10),
                    transforms.RandomCrop((args.img_h, args.img_w)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
                )]
        else:
            self.transform_train = transforms.Compose([  # 训练集的数据增强
                transforms.ToPILImage(),
                transforms.Resize((args.img_h, args.img_w), interpolation=InterpolationMode.LANCZOS),
                transforms.Pad(10),
                transforms.RandomCrop((args.img_h, args.img_w)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        self.transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_h, args.img_w)),
            transforms.ToTensor(),
            normalize,
        ])

        # Traverse each data set
        self.ngall = []
        self.rgb_pos = []
        self.infrared_pos = []
        self.nclass = []
        self.nquery = []
        self.trainset = []
        self.query_label = []
        self.gall_loader = []
        self.query_loader = []
        self.gall_label = []
        self.test_mode = []
        self.query_cam = []
        self.gall_cam = []
        self.pid_increment = []
        self.camid_increment = []
        self.gallset = []
        self.queryset = []
        self.names = datasets
        self.rgb_cams = []
        self.ir_cams = []
        self.max_pid = 0
        self.max_camid = 0

        print("==> All train datasets:{}".format(self.names))
        if args.test_mode == 'vti':
            test_mode = [1, 2]  # [1, 2]: VIS to IR;
        elif args.test_mode == 'itv':
            test_mode = [2, 1]  # [2, 1]: IR to VIS
        elif args.test_mode == 'vtv':
            test_mode = [1, 1]
        elif args.test_mode == 'iti':
            test_mode = [2, 2]

        for step, dataset in enumerate(self.names):
            if args.memory_report == 't':
                before_memory = psutil.Process().memory_info().rss / 1024 / 1024

            if dataset == 'sysu':
                data_path = args.datasets_path + 'SYSU-MM01/'
            elif dataset == 'regdb':
                data_path = args.datasets_path + 'RegDB/'
            elif dataset == 'llcm':
                data_path = args.datasets_path + 'LLCM/'
            elif dataset == 'vcm':
                data_path = args.datasets_path + 'VCM/'

            if dataset == 'sysu':
                # training set
                if args.caj == 't':
                    trainset = CAJSYSUData(data_path, transform=self.transform_train, max_pid=self.max_pid)
                else:
                    trainset = SYSUData(data_path, transform=self.transform_train, max_pid=self.max_pid)
                rgb_pos, infrared_pos = GenIdx(trainset.train_rgb_label, trainset.train_ir_label)
                query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode, modal=test_mode[0], max_pid=self.max_pid, max_camid=self.max_camid)
                gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, seed=0, modal=test_mode[1], max_pid=self.max_pid, max_camid=self.max_camid)

            elif dataset == 'regdb':
                # training set
                if args.caj == 't':
                    trainset = CAJRegDBData(data_path, args.trial, transform=self.transform_train, max_pid=self.max_pid)
                else:
                    trainset = RegDBData(data_path, args.trial, transform=self.transform_train, max_pid=self.max_pid)
                rgb_pos, infrared_pos = GenIdx(trainset.train_rgb_label, trainset.train_ir_label)
                query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal=test_mode[0], max_pid=self.max_pid, max_camid=self.max_camid)
                gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal=test_mode[1], max_pid=self.max_pid, max_camid=self.max_camid)
                if test_mode[0] == 1:
                    query_cam = [1]
                    gall_cam = [2]
                elif test_mode[0] == 2:
                    query_cam = [2]
                    gall_cam = [1]
                query_cam = np.array(query_cam) + self.max_camid
                gall_cam = np.array(gall_cam) + self.max_camid

            elif dataset == 'llcm':
                # training set
                if args.caj == 't':
                    trainset = CAJLLCMData(data_path, transform=self.transform_train, max_pid=self.max_pid)
                else:
                    trainset = LLCMData(data_path, transform=self.transform_train, max_pid=self.max_pid)
                rgb_pos, infrared_pos = GenIdx(trainset.train_rgb_label, trainset.train_ir_label)
                query_img, query_label, query_cam = process_query_llcm(data_path, modal=test_mode[0], max_pid=self.max_pid, max_camid=self.max_camid)
                gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, modal=test_mode[1], trial=0, max_pid=self.max_pid, max_camid=self.max_camid)
                if args.memory_report == 't':
                    query_img, query_label, query_cam = 0, 0, 0
                    gall_img, gall_label, gall_cam = 0, 0, 0

            elif dataset == 'vcm':
                if args.caj == 't':
                    trainset = CAJVCMData(data_path, transform=self.transform_train, max_pid=self.max_pid)
                else:
                    trainset = VCMData(data_path, transform=self.transform_train, max_pid=self.max_pid)
                rgb_pos, infrared_pos = GenIdx(trainset.train_rgb_label, trainset.train_ir_label)
                query_img, query_label, query_cam = process_query_vcm(data_path, modal=test_mode[0], max_pid=self.max_pid, max_camid=self.max_camid)
                gall_img, gall_label, gall_cam = process_gallery_vcm(data_path, modal=test_mode[1], seed=args.seed, max_pid=self.max_pid, max_camid=self.max_camid)

            if args.memory_report == 'f':
                gallset = TestData(gall_img, gall_label, transform=self.transform_test)
                queryset = TestData(query_img, query_label, transform=self.transform_test)
                gall_loader = DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=False)
                query_loader = DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=False)
                nclass = len(np.unique(trainset.train_rgb_label))
                ncam = len(np.unique(np.concatenate((query_cam, gall_cam))))
                nquery = len(query_label)
                ngall = len(gall_label)
            else:
                gallset = 0
                queryset = 0
                gall_loader = 0
                query_loader = 0
                nclass = len(np.unique(trainset.train_rgb_label))
                ncam = 0
                nquery = 0
                ngall = 0

            self.ngall.append(ngall)
            self.nclass.append(nclass)
            self.nquery.append(nquery)
            self.rgb_pos.append(rgb_pos)
            self.infrared_pos.append(infrared_pos)
            self.trainset.append(trainset)
            self.query_label.append(query_label)
            self.gall_loader.append(gall_loader)
            self.query_loader.append(query_loader)
            self.gall_label.append(gall_label)
            self.test_mode.append(test_mode)
            self.query_cam.append(query_cam)
            self.gall_cam.append(gall_cam)
            self.pid_increment.append(self.max_pid)
            self.camid_increment.append(self.max_camid)
            self.gallset.append(gallset)
            self.queryset.append(queryset)
            if test_mode[0] == 1:
                self.rgb_cams.append(query_cam)
                self.ir_cams.append(gall_cam)
            else:
                self.rgb_cams.append(gall_cam)
                self.ir_cams.append(query_cam)
            self.max_pid += nclass
            self.max_camid += ncam
            if step == len(self.names) - 1:
                self.pid_increment.append(self.max_pid)
                self.camid_increment.append(self.max_camid)
            print("Done dataset[{}]".format(dataset))
            if args.memory_report == 't':
                usage_memory = psutil.Process().memory_info().rss / 1024 / 1024 - before_memory
                print(f'Data Step[{step}]------Memory Usage:{usage_memory:.2f} MB')

        if args.joint_learning == 't' and args.tsne != 't':
            self.nclass = [sum(self.nclass)]
            if args.test_mode == 'vti':
                self.trainset = [JointData(self.trainset)] if args.caj != 't' else [CAJJointData(self.trainset)]
                rgb_pos, infrared_pos = GenIdx(self.trainset[0].train_rgb_label, self.trainset[0].train_ir_label)
            else:
                self.trainset = 0
                rgb_pos = 0
                infrared_pos = 0
            self.rgb_pos = [rgb_pos]
            self.infrared_pos = [infrared_pos]
            print("Done joint learning data loading")
        else:
            del self.gallset, self.queryset
        print("Done data loading")
        # 清空所有中间变量
        del trainset, gallset, queryset, gall_loader, query_loader, nclass, ncam, nquery, ngall, rgb_pos, infrared_pos, query_label, gall_label, test_mode, query_cam, gall_cam

    def dataset_info(self, step):
        print('==> Step[{}] Preparing Data Loader...'.format(step))
        print('Dataset {} statistics:'.format(self.names[step]))
        print('  ------------------------------')
        print('  subset     | # ids | # images')
        print('  ------------------------------')
        print('  visible    | {:5d} | {:8d}'.format(self.nclass[step], len(self.trainset[step].train_rgb_label)))
        print('  infrared   | {:5d} | {:8d}'.format(self.nclass[step], len(self.trainset[step].train_ir_label)))
        print('  ------------------------------')
        print('  query      | {:5d} | {:8d}'.format(len(np.unique(self.query_label[step])), self.nquery[step]))
        print('  gallery    | {:5d} | {:8d}'.format(len(np.unique(self.gall_label[step])), self.ngall[step]))
        print('  ------------------------------')


class CAJ_Replay_Dataset(Dataset):
    def __init__(self, method=''):
        args = get_config()
        self.num_instances = 2  # select 2 imgs for each modal every person which is away from the center
        self.select_num = 256
        self.method = method
        if method == 'bic':
            self.val_rgb_img = []
            self.val_ir_img = []
            self.val_rgb_label = []
            self.val_ir_label = []
        self.reply_rgb_img = []
        self.reply_rgb_label = []
        self.reply_rgb_cam = []
        self.reply_ir_img = []
        self.reply_ir_label = []
        self.reply_ir_cam = []
        self.rgb_domain_flag = []
        self.ir_domain_flag = []
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_rgb1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_h, args.img_w), interpolation=InterpolationMode.LANCZOS),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            # ChannelRandomErasing(probability=0.5)
        ])
        self.transform_rgb2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_h, args.img_w), interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            normalize,
            # ChannelRandomErasing(probability=0.5),
            # ChannelExchange(gray=2)
        ])
        self.transform_ir = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_h, args.img_w), interpolation=InterpolationMode.LANCZOS),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            # ChannelRandomErasing(probability=0.5),
            # ChannelAdapGray(probability=0.5)
        ])
        self.transform_extra = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_h, args.img_w)),
            transforms.ToTensor(),
            normalize,
        ])

    @staticmethod
    def extra_features(net, dataloader, modal, gcn, step):
        pids2imgs = defaultdict(list)
        pids2feats = defaultdict(list)
        net.eval()
        with torch.no_grad():
            for inputs, labels, imgs, cmid in dataloader:
                inputs = inputs.cuda()
                if gcn == 't':
                    feats, _, _ = net(inputs, inputs, modal=modal, step=step)  # rgb feats
                else:
                    feats, _ = net(inputs, inputs, modal=modal)  # rgb feats
                for feat, img, pid in zip(feats, imgs, labels):
                    pid = pid.item()
                    pids2imgs[pid].append(img.numpy())
                    pids2feats[pid].append(feat)
        return pids2imgs, pids2feats

    def __len__(self):
        return len(self.reply_rgb_label)

    def increase_reply_data(self, net, dataset, domain_flag, whether_gcn='t', reply_type='fmh', pid_increase=0, step=0, incorrect_pid=None):
        rgb_trainloader = DataLoader(ReplyTmpData(dataset.train_rgb_image, dataset.train_rgb_label, dataset.train_rgb_cam, transform=self.transform_extra), batch_size=64, num_workers=8, shuffle=False, pin_memory=False)
        ir_trainloader = DataLoader(ReplyTmpData(dataset.train_ir_image, dataset.train_ir_label, dataset.train_ir_cam, transform=self.transform_extra), batch_size=64, num_workers=8, shuffle=False, pin_memory=False)
        rgb_pids2imgs, rgb_pids2feats = self.extra_features(net, rgb_trainloader, 1, whether_gcn, step)
        ir_pids2imgs, ir_pids2feats = self.extra_features(net, ir_trainloader, 2, whether_gcn, step)

        reply_meta_samples = select_samples(rgb_pids2imgs, ir_pids2imgs, rgb_pids2feats, ir_pids2feats, domain_flag, reply_type, self.select_num, self.num_instances, pid_increase, incorrect_pid)
        self.reply_rgb_img = self.reply_rgb_img + reply_meta_samples[0]
        self.reply_rgb_label = self.reply_rgb_label + reply_meta_samples[1]
        self.reply_rgb_cam = self.reply_rgb_cam + reply_meta_samples[2]
        self.rgb_domain_flag = self.rgb_domain_flag + reply_meta_samples[3]
        self.reply_ir_img = self.reply_ir_img + reply_meta_samples[4]
        self.reply_ir_label = self.reply_ir_label + reply_meta_samples[5]
        self.reply_ir_cam = self.reply_ir_cam + reply_meta_samples[6]
        self.ir_domain_flag = self.ir_domain_flag + reply_meta_samples[7]

        if self.method == 'bic':
            for pid in rgb_pids2imgs.keys():
                self.val_rgb_img.extend(random.sample(rgb_pids2imgs[pid], 2))
                self.val_rgb_label.extend([pid, pid])
                self.val_ir_img.extend(random.sample(ir_pids2imgs[pid], 2))
                self.val_ir_label.extend([pid, pid])

        print(f'==> Total replay data: rgb-{len(self.reply_rgb_label)} ir-{len(self.reply_ir_label)}')

    def __getitem__(self, index):
        try:
            idx_rgb, idx_ir = index
            img1, target1, domain1 = self.reply_rgb_img[idx_rgb], self.reply_rgb_label[idx_rgb], self.rgb_domain_flag[idx_rgb]
            img2, target2, doamin2 = self.reply_ir_img[idx_ir], self.reply_ir_label[idx_ir], self.ir_domain_flag[idx_ir]
            img_rgb1 = self.transform_rgb1(img1)
            # img_rgb2 = self.transform_rgb2(img1)
            img_rgb2 = 0
            img_ir = self.transform_ir(img2)
        except:
            print('index:', index)
        return img_rgb1, img_rgb2, img_ir, target1, target2, domain1, doamin2


class CAJ_Replay_DER_Dataset(object):
    def __init__(self, device=None, method=''):
        args = get_config()
        self.num_instances = 2
        self.select_num = 256
        self.method = method
        self.reply_rgb_img = []
        self.reply_rgb_label = []
        self.reply_rgb_logit = []
        self.domain_flag = defaultdict(list)
        self.reply_ir_img = []
        self.reply_ir_label = []
        self.reply_ir_logit = []
        self.meta_data = []
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_rgb1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_h, args.img_w), interpolation=InterpolationMode.LANCZOS),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.transform_rgb2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize,
        ])
        self.transform_ir = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_h, args.img_w), interpolation=InterpolationMode.LANCZOS),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.transform_extra = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_h, args.img_w)),
            transforms.ToTensor(),
            normalize,
        ])

    @staticmethod
    def extra_features(net, dataloader, modal, gcn):
        pids2imgs = defaultdict(list)
        pids2feats = defaultdict(list)
        net.eval()
        with torch.no_grad():
            for inputs, labels, imgs, cmid in dataloader:
                inputs = inputs.cuda()
                if gcn == 't':
                    feats, _, _ = net(inputs, inputs, modal=modal)  # rgb feats
                else:
                    feats, _ = net(inputs, inputs, modal=modal)  # rgb feats
                for feat, img, pid in zip(feats, imgs, labels):
                    pid = pid.item()
                    pids2imgs[pid].append(img.numpy())
                    pids2feats[pid].append(feat)
        return pids2imgs, pids2feats

    def increase_reply_data(self, net, dataset, domain_flag, whether_gcn='t', reply_type='fmh', pid_increase=0, incorrect_pid=None):
        rgb_trainloader = DataLoader(ReplyTmpData(dataset.train_rgb_image, dataset.train_rgb_label, dataset.train_rgb_cam, transform=self.transform_extra), batch_size=64, num_workers=8, shuffle=False, pin_memory=False)
        ir_trainloader = DataLoader(ReplyTmpData(dataset.train_ir_image, dataset.train_ir_label, dataset.train_ir_cam, transform=self.transform_extra), batch_size=64, num_workers=8, shuffle=False, pin_memory=False)
        rgb_pids2imgs, rgb_pids2feats = self.extra_features(net, rgb_trainloader, 1, whether_gcn)
        ir_pids2imgs, ir_pids2feats = self.extra_features(net, ir_trainloader, 2, whether_gcn)

        reply_meta_samples = select_samples(rgb_pids2imgs, ir_pids2imgs, rgb_pids2feats, ir_pids2feats, domain_flag, reply_type, self.select_num, self.num_instances, pid_increase, incorrect_pid)
        reply_dataloader = DataLoader(CAJReplyTmpData2(reply_meta_samples[0], reply_meta_samples[4], reply_meta_samples[1], reply_meta_samples[5], reply_meta_samples[3], reply_meta_samples[7],
                                                       transform_list=[self.transform_rgb1, self.transform_rgb2, self.transform_ir]), batch_size=8 * 2, num_workers=8, shuffle=False, drop_last=False)

        net.train()
        reply_dataloader = tqdm(reply_dataloader)
        with torch.no_grad():
            for img_rgb1, img_rgb2, img_ir, target1, target2, rgb_domain_flag, ir_domain_flag in reply_dataloader:
                img_rgb = img_rgb1.cuda()
                target = torch.cat((target1, target2), dim=0)
                img_ir = img_ir.cuda()
                _, logit_total = net(img_rgb, img_ir)
                img_rgb = img_rgb.cpu()
                img_ir = img_ir.cpu()
                target = target.cpu()
                logit_total = logit_total.cpu()
                self.meta_data.append((img_rgb, img_ir, target, logit_total))

        print(f'==> Replay data: meta_data-{len(self.meta_data)}')

    def __len__(self):
        return len(self.meta_data)


class CAJ_Replay_PCB_Dataset(object):
    def __init__(self, device=None, method='', args=None):
        args = get_config()
        self.num_instances = 2
        self.select_num = args.memory_size
        self.method = method
        self.args = args
        self.reply_rgb_img = []
        self.reply_rgb_label = []
        self.reply_rgb_cam = []
        self.rgb_domain_flag = []
        self.ir_domain_flag = []
        self.reply_ir_img = []
        self.reply_ir_label = []
        self.reply_ir_cam = []
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_rgb1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_h, args.img_w), interpolation=InterpolationMode.LANCZOS),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.transform_rgb2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize,
        ])
        self.transform_ir = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_h, args.img_w), interpolation=InterpolationMode.LANCZOS),
            transforms.Pad(10),
            transforms.RandomCrop((args.img_h, args.img_w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.transform_extra = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.img_h, args.img_w)),
            transforms.ToTensor(),
            normalize,
        ])

    @staticmethod
    def extra_features(net, dataloader, modal, gcn, step):
        pids2imgs = defaultdict(list)
        pids2feats = defaultdict(list)
        net.eval()
        with torch.no_grad():
            for inputs, labels, imgs, cmid in dataloader:
                inputs = inputs.cuda()
                if gcn == 't':
                    feats, _, _ = net(inputs, inputs, modal=modal, step=step)  # rgb feats
                else:
                    feats, _ = net(inputs, inputs, modal=modal)  # rgb feats
                for feat, img, pid in zip(feats, imgs, labels):
                    pid = pid.item()
                    pids2imgs[pid].append(img.numpy())
                    pids2feats[pid].append(feat)
        return pids2imgs, pids2feats

    def increase_reply_data(self, net, dataset, domain_flag, whether_gcn='t', reply_type='fmh', pid_increase=0, incorrect_pid=None):
        rgb_trainloader = DataLoader(ReplyTmpData(dataset.train_rgb_image, dataset.train_rgb_label, dataset.train_rgb_cam, transform=self.transform_extra), batch_size=64, num_workers=8, shuffle=False, pin_memory=False)
        ir_trainloader = DataLoader(ReplyTmpData(dataset.train_ir_image, dataset.train_ir_label, dataset.train_ir_cam, transform=self.transform_extra), batch_size=64, num_workers=8, shuffle=False, pin_memory=False)
        rgb_pids2imgs, rgb_pids2feats = self.extra_features(net, rgb_trainloader, 1, whether_gcn, domain_flag)
        ir_pids2imgs, ir_pids2feats = self.extra_features(net, ir_trainloader, 2, whether_gcn, domain_flag)

        reply_meta_samples = select_samples(rgb_pids2imgs, ir_pids2imgs, rgb_pids2feats, ir_pids2feats, domain_flag, reply_type, self.select_num, self.num_instances, pid_increase, incorrect_pid)
        self.reply_rgb_img = self.reply_rgb_img + reply_meta_samples[0]
        self.reply_rgb_label = self.reply_rgb_label + reply_meta_samples[1]
        self.reply_rgb_cam = self.reply_rgb_cam + reply_meta_samples[2]
        self.rgb_domain_flag = self.rgb_domain_flag + reply_meta_samples[3]
        self.reply_ir_img = self.reply_ir_img + reply_meta_samples[4]
        self.reply_ir_label = self.reply_ir_label + reply_meta_samples[5]
        self.reply_ir_cam = self.reply_ir_cam + reply_meta_samples[6]
        self.ir_domain_flag = self.ir_domain_flag + reply_meta_samples[7]
        print(f'==> Replay data: meta_data-{len(self.reply_rgb_img)}')

    def __len__(self):
        return len(self.reply_rgb_label)

    def __getitem__(self, index):
        cross_domain_p = self.args.cross_domain_p
        cross_modal_p = self.args.cross_modal_p
        idx_rgb, idx_ir = index
        img_tuples = [0, 0]

        img_tuples[0], target1, domain1 = self.reply_rgb_img[idx_rgb], self.reply_rgb_label[idx_rgb], self.rgb_domain_flag[idx_rgb]
        img_tuples[1], target2, domain2 = self.reply_ir_img[idx_ir], self.reply_ir_label[idx_ir], self.ir_domain_flag[idx_ir]
        # print(img_tuples[0][0].shape, img_tuples[1][0].shape)
        img = [np.empty((256, 128, 3)) for _ in range(2)]
        img[0] = img_tuples[0][random.randint(0, 1)]
        img[1] = img_tuples[1][random.randint(0, 1)]
        blocks = 4
        for i in range(len(img)):
            for block_idx in range(blocks):
                left_indices = img[i].shape[0] // blocks * block_idx
                right_indices = img[i].shape[0] // blocks * (block_idx + 1)
                channel_type = random.randint(0, self.args.channel_type)

                if random.uniform(0, 1) > cross_domain_p or len(np.unique(self.rgb_domain_flag)) < 2:
                    img_idx = np.random.choice(range(len(img_tuples[i])), 1)[0]
                    # print(img_idx)
                    if random.uniform(0, 1) > cross_modal_p:
                        if channel_type == 0:
                            channel_select_1 = random.randint(0, 2)
                            img[i][left_indices:right_indices, :, channel_select_1] = img_tuples[i][img_idx][left_indices:right_indices, :, channel_select_1]
                        elif channel_type == 1:
                            channel_select_1 = np.random.choice([0, 1, 2], 2)
                            img[i][left_indices:right_indices, :, channel_select_1] = img_tuples[i][img_idx][left_indices:right_indices, :, channel_select_1]
                        else:
                            img[i][left_indices:right_indices, :, :] = img_tuples[i][img_idx][left_indices:right_indices, :, :]
                    else:
                        img_idx = np.random.choice(range(len(img_tuples[1-i])), 1)[0]
                        if channel_type == 0:
                            channel_select_1 = random.randint(0, 2)
                            img[i][left_indices:right_indices, :, channel_select_1] = img_tuples[1-i][img_idx][left_indices:right_indices, :, channel_select_1]
                        elif channel_type == 1:
                            channel_select_1 = np.random.choice([0, 1, 2], 2)
                            img[i][left_indices:right_indices, :, channel_select_1] = img_tuples[1-i][img_idx][left_indices:right_indices, :, channel_select_1]
                        else:
                            img[i][left_indices:right_indices, :, :] = img_tuples[1-i][img_idx][left_indices:right_indices, :, :]

                elif len(np.unique(self.rgb_domain_flag)) > 1:
                    if i == 0:
                        if np.random.uniform(0, 1) > cross_modal_p:
                            domains_flag = np.unique(self.rgb_domain_flag)
                            domains_flag = np.delete(domains_flag, np.where(domains_flag == domain1))
                            choose_domain = np.random.choice(domains_flag, 1)[0]
                            cross_domain_idx = np.random.choice(np.where(self.rgb_domain_flag == choose_domain)[0], size=1)[0]
                            cross_domain_tuples, _, _ = self.reply_rgb_img[cross_domain_idx], self.reply_rgb_label[cross_domain_idx], self.rgb_domain_flag[cross_domain_idx]
                        else:
                            domains_flag = np.unique(self.ir_domain_flag)
                            domains_flag = np.delete(domains_flag, np.where(domains_flag == domain1))
                            choose_domain = np.random.choice(domains_flag, 1)[0]
                            cross_domain_idx = np.random.choice(np.where(self.ir_domain_flag == choose_domain)[0], size=1)[0]
                            cross_domain_tuples, _, _ = self.reply_ir_img[cross_domain_idx], self.reply_ir_label[cross_domain_idx], self.ir_domain_flag[cross_domain_idx]
                    else:
                        if np.random.uniform(0, 1) > cross_modal_p:
                            domains_flag = np.unique(self.ir_domain_flag)
                            domains_flag = np.delete(domains_flag, np.where(domains_flag == domain2))
                            choose_domain = np.random.choice(domains_flag, 1)[0]
                            cross_domain_idx = np.random.choice(np.where(self.ir_domain_flag == choose_domain)[0], size=1)[0]
                            cross_domain_tuples, _, _ = self.reply_ir_img[cross_domain_idx], self.reply_ir_label[cross_domain_idx], self.ir_domain_flag[cross_domain_idx]
                        else:
                            domains_flag = np.unique(self.rgb_domain_flag)
                            domains_flag = np.delete(domains_flag, np.where(domains_flag == domain2))
                            choose_domain = np.random.choice(domains_flag, 1)[0]
                            cross_domain_idx = np.random.choice(np.where(self.rgb_domain_flag == choose_domain)[0], size=1)[0]
                            cross_domain_tuples, _, _ = self.reply_rgb_img[cross_domain_idx], self.reply_rgb_label[cross_domain_idx], self.rgb_domain_flag[cross_domain_idx]

                    cross_domain_img_idx = np.random.choice(range(len(cross_domain_tuples)), 1)[0]
                    if channel_type == 0:
                        channel_select_1 = random.randint(0, 2)
                        img[i][left_indices:right_indices, :, channel_select_1] = cross_domain_tuples[cross_domain_img_idx][left_indices:right_indices, :, channel_select_1]
                    elif channel_type == 1:
                        channel_select_1 = np.random.choice([0, 1, 2], 2)
                        img[i][left_indices:right_indices, :, channel_select_1] = cross_domain_tuples[cross_domain_img_idx][left_indices:right_indices, :, channel_select_1]
                    else:
                        img[i][left_indices:right_indices, :, :] = cross_domain_tuples[cross_domain_img_idx][left_indices:right_indices, :, :]
                    # ---------------------------------------------------------------------------

        img1 = img[0]
        img2 = img[1]

        img_rgb1 = self.transform_rgb1(img1)
        img_rgb2 = 0
        img_ir = self.transform_ir(img2)

        return img_rgb1, img_rgb2, img_ir, target1, target2, domain1, domain2


class SNEDatasets(Dataset):
    def __init__(self, trainset=None, modal='rgb', transform=None):
        if modal == 'rgb':
            indexs = np.random.choice(range(len(trainset.train_rgb_label)), min(len(trainset.train_ir_label), 1024), replace=False)
            self.train_image = [trainset.train_rgb_image[idx] for idx in indexs]
            self.train_label = [trainset.train_rgb_label[idx] for idx in indexs]
            self.train_cam = [trainset.train_rgb_cam[idx] for idx in indexs]
        else:
            indexs = np.random.choice(range(len(trainset.train_ir_label)), min(len(trainset.train_ir_label), 1024), replace=False)
            self.train_image = [trainset.train_ir_image[idx] for idx in indexs]
            self.train_label = [trainset.train_ir_label[idx] for idx in indexs]
            self.train_cam = [trainset.train_ir_cam[idx] for idx in indexs]

        self.transform = transform

    def __getitem__(self, index):
        img1, target1, cam1 = self.train_image[index], self.train_label[index], self.train_cam[index]
        img1 = self.transform(img1)
        return img1, target1, cam1

    def __len__(self):
        return len(self.train_label)
