from __future__ import print_function, absolute_import
import os
import pickle
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def process_query_sysu(data_path, max_pid, max_camid, mode='all', modal=2):
    if mode == 'all':
        if modal == 1:
            cameras = ['cam1', 'cam2', 'cam4', 'cam5']  # rgb
        else:
            cameras = ['cam3', 'cam6']  # ir
    elif mode == 'indoor':
        if modal == 1:
            cameras = ['cam1', 'cam2']  # rgb
        else:
            cameras = ['cam3', 'cam6']  # ir

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_list = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_list.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_list:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid + max_pid)
        query_cam.append(camid + max_camid)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_sysu(data_path, max_pid, max_camid, mode='all', seed=0, modal=1):
    if mode == 'all':
        if modal == 1:
            cameras = ['cam1', 'cam2', 'cam4', 'cam5']
        else:
            cameras = ['cam3', 'cam6']
    elif mode == 'indoor':
        if modal == 1:
            cameras = ['cam1', 'cam2']
        else:
            cameras = ['cam3', 'cam6']

    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_list = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    # 每个角色从每个摄像头中选一张
    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_list.append(random.choice(new_files))

    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_list:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid + max_pid)
        gall_cam.append(camid + max_camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_test_regdb(img_dir, max_pid, max_camid, trial=1, modal=1):
    if modal == 1:
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal == 2:
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'
    # if modal == 'visible':
    #     input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    # elif modal == 'thermal':
    #     input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'

    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) + max_pid for s in data_file_list]

    return file_image, np.array(file_label)


def process_query_llcm(data_path, modal, max_pid, max_camid):
    if modal == 1:
        cameras = ['test_vis/cam1', 'test_vis/cam2', 'test_vis/cam3', 'test_vis/cam4', 'test_vis/cam5', 'test_vis/cam6',
                   'test_vis/cam7', 'test_vis/cam8', 'test_vis/cam9']
    elif modal == 2:
        cameras = ['test_nir/cam1', 'test_nir/cam2', 'test_nir/cam4', 'test_nir/cam5', 'test_nir/cam6', 'test_nir/cam7',
                   'test_nir/cam8', 'test_nir/cam9']

    file_path = os.path.join(data_path, 'idx/test_id.txt')
    files_list = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_list.extend(new_files)

    query_img = []
    query_id = []
    query_cam = []

    for img_path in files_list:
        camid, pid = int(img_path.split('cam')[1][0]), int(img_path.split('cam')[1][2:6])
        query_img.append(img_path)
        query_id.append(pid + max_pid)
        query_cam.append(camid + max_camid)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_llcm(data_path, modal, trial, max_pid, max_camid):
    if modal == 1:
        cameras = ['test_vis/cam1', 'test_vis/cam2', 'test_vis/cam3', 'test_vis/cam4', 'test_vis/cam5', 'test_vis/cam6',
                   'test_vis/cam7', 'test_vis/cam8', 'test_vis/cam9']
    elif modal == 2:
        cameras = ['test_nir/cam1', 'test_nir/cam2', 'test_nir/cam4', 'test_nir/cam5', 'test_nir/cam6', 'test_nir/cam7',
                   'test_nir/cam8', 'test_nir/cam9']

    file_path = os.path.join(data_path, 'idx/test_id.txt')
    files_list = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_list.append(random.choice(new_files))

    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_list:
        camid, pid = int(img_path.split('cam')[1][0]), int(img_path.split('cam')[1][2:6])
        gall_img.append(img_path)
        gall_id.append(pid + max_pid)
        gall_cam.append(camid + max_camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_query_vcm(data_path, modal, max_pid, max_camid):
    if modal == 1:  # rgb
        with open(data_path + 'query_vis_path_img.pkl', 'rb') as f:
            gall = pickle.load(f)
    elif modal == 2:  # ir
        with open(data_path + 'query_ir_path_img.pkl', 'rb') as f:
            gall = pickle.load(f)

    gall_img = []
    gall_id = []
    gall_cam = []
    for i in range(len(gall)):
        pid = gall[i][0]
        for j, imgs in enumerate(gall[i][1]):
            camid = imgs[0]
            files_list = sorted(imgs[1])
            for img_path in files_list:
                gall_img.append(img_path)
                gall_id.append(pid + max_pid)
                gall_cam.append(camid + max_camid)

    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_gallery_vcm(data_path, modal, seed, max_pid, max_camid):
    if modal == 1:  # rgb
        with open(data_path + 'gallery_vis_path_img.pkl', 'rb') as f:
            gall = pickle.load(f)
    elif modal == 2:  # ir
        with open(data_path + 'gallery_ir_path_img.pkl', 'rb') as f:
            gall = pickle.load(f)

    gall_img = []
    gall_id = []
    gall_cam = []
    for i in range(len(gall)):
        pid = gall[i][0]
        for j, imgs in enumerate(gall[i][1]):
            camid = imgs[0]
            files_list = sorted(imgs[1])
            img_path = random.choice(files_list)
            gall_img.append(img_path)
            gall_id.append(pid + max_pid)
            gall_cam.append(camid + max_camid)

    return gall_img, np.array(gall_id), np.array(gall_cam)


class TestData(Dataset):
    def __init__(self, test_img_file, test_label, transform=None):
        test_image = []
        for i in tqdm(range(len(test_img_file))):
            img = Image.open(test_img_file[i])
            img = img.resize((128, 256), Image.LANCZOS)
            test_image.append(np.array(img))
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
