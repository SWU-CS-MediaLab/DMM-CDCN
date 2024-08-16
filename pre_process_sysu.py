import numpy as np
from PIL import Image
from config import get_config
import os

rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
ir_cameras = ['cam3', 'cam6']
args = get_config()

data_path = os.path.join(args.datasets_path, 'SYSU-MM01/')
save_path = os.path.join(args.datasets_path, 'SYSU-MM01/')

# load id info
file_path_train = os.path.join(data_path, 'exp/train_id.txt')
file_path_val = os.path.join(data_path, 'exp/val_id.txt')

with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_train = ["%04d" % x for x in ids]

with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]

# combine train and val split   
id_train.extend(id_val)

# 提取红外和可见光图像的路径
files_rgb = []
files_ir = []
for id in sorted(id_train):
    for cam in rgb_cameras:
        img_dir = os.path.join(data_path, cam, id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)

    for cam in ir_cameras:
        img_dir = os.path.join(data_path, cam, id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)

# relabel 重新编号
pid_container = set()
# 提取编号
for img_path in files_ir:
    pid = int(img_path[-13:-9])
    pid_container.add(pid)
pid2label = {pid: label for label, pid in enumerate(pid_container)}
args = get_config()
fix_image_width = args.img_w
fix_image_height = args.img_h


def read_imgs(train_image):
    train_img = []
    train_label = []
    train_cmid = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((args.img_w, args.img_h), Image.LANCZOS)
        pix_array = np.array(img)

        train_img.append(pix_array)

        # label
        pid = int(img_path[-13:-9])
        cmid = int(img_path[-15])
        pid = pid2label[pid]
        train_label.append(pid)
        train_cmid.append(cmid)
    return np.array(train_img), np.array(train_label), np.array(train_cmid)


# rgb imges
train_img, train_label, train_cmid = read_imgs(files_rgb)
np.save(save_path + '/train_rgb_resized_img.npy', train_img)
np.save(save_path + '/train_rgb_resized_label.npy', train_label)
np.save(save_path + '/train_rgb_resized_cam.npy', train_cmid)

# ir imges
train_img, train_label, train_cmid = read_imgs(files_ir)
np.save(save_path + '/train_ir_resized_img.npy', train_img)
np.save(save_path + '/train_ir_resized_label.npy', train_label)
np.save(save_path + '/train_ir_resized_cam.npy', train_cmid)

print('Done!')
