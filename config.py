import argparse


def get_config():
    # setting
    parser = argparse.ArgumentParser(description='Life long Cross-Model ReID')

    # train
    parser.add_argument('--total_epoch', default=50, type=int, metavar='N', help='total epochs for single step')
    parser.add_argument('--gpu', default=1, type=int, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--batch_size', default=8, type=int, metavar='B', help='training batch size')
    parser.add_argument('--next_step', default='f', type=str, help='next step')
    parser.add_argument('--guider_train', default='f', type=str, help='whether train guider network')
    parser.add_argument('--fp_16', type=str, default='f', help='whether use fp16')
    parser.add_argument('--memory_size', type=int, default=256, help='the memory size of stored of each modality after each stage')
    parser.add_argument('--test_frequency', type=int, default=3, help='test during trai, i <= 0 means do not test during train')

    # network
    parser.add_argument('--caj', default='t', type=str, help='the original option to use caj network, but we do not use it in this project even if it is set to t, do not change it')
    parser.add_argument('--gcn', default='f', type=str, help='whether use gcn network')
    parser.add_argument('--weight_align', type=str, default='f', help='whether use weight align loss')
    parser.add_argument('--sample_reply', type=str, default='t', help='whether sample reply data')
    parser.add_argument('--method', type=str, default='', help='method')
    parser.add_argument('--reply_type', type=str, default='fmh', help='reply type')

    # test
    parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
    parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
    parser.add_argument('--test_batch', default=64, type=int, metavar='tb', help='testing batch size')
    parser.add_argument('--test_only', default='f', type=str, help='test only')
    parser.add_argument('--mode', default='all', type=str, help='for sysu all or indoor')
    parser.add_argument('--test_mode', default='vti', type=str, help='test mode, whether vis to ir or ir to vis')
    parser.add_argument('--original_datasets', default='f', type=str, help='original datasets')
    parser.add_argument('--domain_gap_test', default='f', type=str, help='test the domain gap')
    parser.add_argument('--single_modal_test', default='f', type=str, help='test the single modal performance')

    # optimize
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.1 for sgd, 0.00035 for adam')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
    parser.add_argument('--wrt_loss', type=str, default='f', help='whether use wrt loss')
    parser.add_argument('--kd_loss', type=str, default='kl', help='the type of kd loss')
    parser.add_argument('--use_kd', type=str, default='t', help='whether use kd loss')
    parser.add_argument('--weight_kd', type=float, default=1, help='weight for kd loss')

    # log
    parser.add_argument('--debug', default='f', type=str, help='debug or not')
    parser.add_argument('--ex_name', default='default_ex', type=str, help='ex_names')
    parser.add_argument('--tsne', default='f', type=str, metavar='N', help='whether use tsne')
    parser.add_argument('--model_path', default='/mnt/sda4/zxy/saved_model', type=str, help='model save path')
    parser.add_argument('--log_path', default='/mnt/sda4/zxy/log/', type=str, help='log save path')
    parser.add_argument('--vis_log_path', default='/mnt/sda4/zxy/log/vis_log/', type=str, help='log save path')
    parser.add_argument('--save_epoch', default=10, type=int, metavar='s', help='save model every 10 epochs')
    parser.add_argument('--memory_report', default='f', type=str, help='whether report memory usage')

    # image
    parser.add_argument('--img_w', default=128, type=int, metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=256, type=int, metavar='imgh', help='img height')
    parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--erasing_p', default=0.3, type=float, help='Random Erasing probability, in [0,1]')

    # dataset
    parser.add_argument('--datasets_path', default='/mnt/sda4/zxy/Datasets/', type=str, help='dataset path')
    parser.add_argument('--tmp_net_path', nargs='+', default='', type=str, help='reply tmp net path')
    parser.add_argument('--train_datasets', nargs='+', default=['regdb', 'sysu', 'llcm', 'vcm'], type=str, help='train datasets')
    parser.add_argument('--test_datasets', nargs='+', default=['regdb', 'sysu', 'llcm', 'vcm'], type=str, help='test datasets')
    parser.add_argument('--joint_learning', default='f', type=str, help='whether to train and test jointly')

    # RSR
    parser.add_argument('--cross_domain_p', type=float, default=0.5, help='the possibility to use cross_domain_mix')
    parser.add_argument('--cross_modal_p', type=float, default=0.05, help='the possibility to use cross_modal_mix')
    parser.add_argument('--channel_type', type=int, default=2, help='0,1,2 the channel type used in mix')

    args, unknown = parser.parse_known_args()
    return args
