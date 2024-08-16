import copy
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class CAJRegDBData(Dataset):  # 64*128
    def __init__(self, data_dir, trial, max_pid, transform=None):
        # Load training images (path) and labels
        train_rgb_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_ir_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        rgb_img_file, train_rgb_label = self.load_data(train_rgb_list, max_pid)
        ir_img_file, train_ir_label = self.load_data(train_ir_list, max_pid)

        train_rgb_image = []
        for i in range(len(rgb_img_file)):
            img = Image.open(data_dir + rgb_img_file[i])
            img = img.resize((128, 256), Image.LANCZOS)
            train_rgb_image.append(np.array(img))
        train_rgb_image = np.array(train_rgb_image)

        train_ir_image = []
        for i in range(len(ir_img_file)):
            img = Image.open(data_dir + ir_img_file[i])
            img = img.resize((128, 256), Image.LANCZOS)
            train_ir_image.append(np.array(img))
        train_ir_image = np.array(train_ir_image)

        self.train_rgb_image = train_rgb_image
        self.train_rgb_label = train_rgb_label

        self.train_ir_image = train_ir_image
        self.train_ir_label = train_ir_label

        self.train_rgb_cam = [0 for i in range(len(train_rgb_image))]
        self.train_ir_cam = [0 for i in range(len(train_ir_image))]

        self.transform_rgb1 = transform[0]
        self.transform_rgb2 = transform[1]
        self.transform_ir = transform[2]
        self.rIndex = None
        self.iIndex = None

    def __getitem__(self, index):
        img1, target1 = self.train_rgb_image[self.rIndex[index]], self.train_rgb_label[self.rIndex[index]]
        img2, target2 = self.train_ir_image[self.iIndex[index]], self.train_ir_label[self.iIndex[index]]
        img_rgb1 = self.transform_rgb1(img1)
        # img_rgb2 = self.transform_rgb2(img1)
        img_rgb2 = 0
        img_ir = self.transform_ir(img2)
        return img_rgb1, img_rgb2, img_ir, target1, target2, 1, 2

    def __len__(self):
        return len(self.train_rgb_label)

    @staticmethod
    def load_data(input_data_path, max_pid):
        with open(input_data_path) as f:
            data_file_list = open(input_data_path, 'rt').read().splitlines()
            file_image = [s.split(' ')[0] for s in data_file_list]
            file_label = [int(s.split(' ')[1]) + max_pid for s in data_file_list]

        return file_image, file_label


class CAJSYSUData(Dataset):
    def __init__(self, data_dir,  max_pid, transform=None):

        # Load training images (path) and labels
        self.train_rgb_label = np.load(data_dir + 'train_rgb_resized_label.npy')
        self.train_rgb_label = self.train_rgb_label + max_pid

        self.train_ir_label = np.load(data_dir + 'train_ir_resized_label.npy')
        self.train_ir_label = self.train_ir_label + max_pid

        self.train_rgb_cam = np.load(data_dir + 'train_rgb_resized_cam.npy')
        self.train_ir_cam = np.load(data_dir + 'train_ir_resized_cam.npy')
        
        # BGR to RGB
        self.train_rgb_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_ir_image = np.load(data_dir + 'train_ir_resized_img.npy')

        self.transform_rgb1 = transform[0]
        self.transform_rgb2 = transform[1]
        self.transform_ir = transform[2]
        self.rIndex = None
        self.iIndex = None

    def __getitem__(self, index):
        img1, target1, cam1 = self.train_rgb_image[self.rIndex[index]], self.train_rgb_label[self.rIndex[index]], self.train_rgb_cam[self.rIndex[index]]
        img2, target2, cam2 = self.train_ir_image[self.iIndex[index]], self.train_ir_label[self.iIndex[index]], self.train_ir_cam[self.iIndex[index]]
        img_rgb1 = self.transform_rgb1(img1)
        # img_rgb2 = self.transform_rgb2(img1)
        img_rgb2 = 0
        img_ir = self.transform_ir(img2)
        return img_rgb1, img_rgb2, img_ir, target1, target2, cam1, cam2

    def __len__(self):
        return len(self.train_rgb_label)


class CAJLLCMData(Dataset):
    def __init__(self, data_dir, max_pid, transform=None):
        # BGR to RGB
        self.train_rgb_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_rgb_label = np.load(data_dir + 'train_rgb_resized_label.npy')
        self.train_rgb_cam = np.load(data_dir + 'train_rgb_resized_cam.npy')
        self.train_rgb_label = self.train_rgb_label + max_pid

        # BGR to RGB
        self.train_ir_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_ir_label = np.load(data_dir + 'train_ir_resized_label.npy')
        self.train_ir_cam = np.load(data_dir + 'train_ir_resized_cam.npy')
        self.train_ir_label = self.train_ir_label + max_pid

        self.transform_rgb1 = transform[0]
        self.transform_rgb2 = transform[1]
        self.transform_ir = transform[2]
        self.rIndex = None
        self.iIndex = None

    def __getitem__(self, index):
        img1, target1, cam1 = self.train_rgb_image[self.rIndex[index]], self.train_rgb_label[self.rIndex[index]], self.train_rgb_cam[self.rIndex[index]]
        img2, target2, cam2 = self.train_ir_image[self.iIndex[index]], self.train_ir_label[self.iIndex[index]], self.train_ir_cam[self.iIndex[index]]
        img_rgb1 = self.transform_rgb1(img1)
        # img_rgb2 = self.transform_rgb2(img1)
        img_rgb2 = 0
        img_ir = self.transform_ir(img2)
        return img_rgb1, img_rgb2, img_ir, target1, target2, cam1, cam2

    def __len__(self):
        return len(self.train_rgb_label)


class CAJVCMData(Dataset):
    def __init__(self, data_dir, max_pid, transform=None):

        self.train_rgb_label = np.load(data_dir + 'train_rgb_path_label.npy')
        self.train_rgb_label = self.train_rgb_label + max_pid
        self.train_ir_label = np.load(data_dir + 'train_ir_path_label.npy')
        self.train_ir_label = self.train_ir_label + max_pid
        self.train_rgb_cam = np.load(data_dir + 'train_rgb_path_cam.npy')
        self.train_ir_cam = np.load(data_dir + 'train_ir_path_cam.npy')
        self.train_rgb_image = np.load(data_dir + 'train_rgb_path_img.npy')
        self.train_ir_image = np.load(data_dir + 'train_ir_path_img.npy')

        self.transform_rgb1 = transform[0]
        self.transform_rgb2 = transform[1]
        self.transform_ir = transform[2]
        self.rIndex = None
        self.iIndex = None

    def __getitem__(self, index):
        img1, target1, cam1 = self.train_rgb_image[self.rIndex[index]], self.train_rgb_label[self.rIndex[index]], self.train_rgb_cam[self.rIndex[index]]
        img2, target2, cam2 = self.train_ir_image[self.iIndex[index]], self.train_ir_label[self.iIndex[index]], self.train_ir_cam[self.iIndex[index]]
        img_rgb1 = self.transform_rgb1(img1)
        # img_rgb2 = self.transform_rgb2(img1)
        img_rgb2 = 0
        img_ir = self.transform_ir(img2)
        return img_rgb1, img_rgb2, img_ir, target1, target2, cam1, cam2

    def __len__(self):
        return len(self.train_rgb_label)


class CAJReplyTmpData(Dataset):
    def __init__(self, img, label, transform):
        self.img = img
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        img, target = self.img[index], self.label[index]
        img_original = copy.deepcopy(img)
        img_original_pil = Image.fromarray(img_original)
        img_original_resized = img_original_pil.resize((256, 128), resample=Image.BILINEAR)
        img_original = np.array(img_original_resized)
        img = self.transform(img)
        print(img.shape)
        print(img_original.shape)
        return img, target, img_original


class CAJReplyTmpData2(Dataset):
    def __init__(self, rgb_img, ir_img, rgb_label, ir_label, rgb_domain_flag, ir_domain_flag, transform_list):
        self.rgb_img = np.array(rgb_img)
        self.ir_img = np.array(ir_img)
        self.rgb_label = np.array(rgb_label)
        self.ir_label = np.array(ir_label)
        self.rgb_domain_flag = np.array(rgb_domain_flag)
        self.ir_domain_flag = np.array(ir_domain_flag)
        self.transform_rgb1 = transform_list[0]
        self.transform_rgb2 = transform_list[1]
        self.transform_ir = transform_list[2]

    def __len__(self):
        return len(self.rgb_label)

    def __getitem__(self, index):
        img1, target1, rgb_domain_flag = self.rgb_img[index], self.rgb_label[index], self.rgb_domain_flag[index]
        img2, target2, ir_domain_flag = self.ir_img[index], self.ir_label[index], self.ir_domain_flag[index]

        img_rgb1 = self.transform_rgb1(img1)
        # img_rgb2 = self.transform_rgb2(img1)
        img_rgb2 = 0
        img_ir = self.transform_ir(img1)
        return img_rgb1, img_rgb2, img_ir, target1, target2, rgb_domain_flag, ir_domain_flag


class CAJBiCTrainData(Dataset):
    def __init__(self, rgb_img, ir_img, rgb_label, ir_label, transform):
        self.train_rgb_label = np.array(rgb_label)
        self.train_ir_label = np.array(ir_label)
        self.train_rgb_img = np.array(rgb_img)
        self.train_ir_img = np.array(ir_img)
        self.transform_rgb1 = transform[0]
        self.transform_rgb2 = transform[1]
        self.transform_ir = transform[2]

    def __len__(self):
        return len(self.train_rgb_label)

    def __getitem__(self, index):
        try:
            idx_rgb, idx_ir = index
            img1, target1 = self.train_rgb_img[idx_rgb], self.train_rgb_label[idx_rgb]
            img2, target2 = self.train_ir_img[idx_ir], self.train_ir_label[idx_ir]
            img_rgb1 = self.transform_rgb1(img1)
            # img_rgb2 = self.transform_rgb2(img1)
            img_rgb2 = 0
            img_ir = self.transform_ir(img2)
        except:
            print('index:', index)
        return img_rgb1, img_rgb2, img_ir, target1, target2


class CAJJointData(Dataset):
    def __init__(self, datasets=[]):
        self.train_rgb_image = datasets[0].train_rgb_image
        self.train_rgb_label = datasets[0].train_rgb_label
        self.train_rgb_cam = datasets[0].train_rgb_cam
        self.train_ir_image = datasets[0].train_ir_image
        self.train_ir_label = datasets[0].train_ir_label
        self.train_ir_cam = datasets[0].train_ir_cam

        for i in range(1, len(datasets)):
            self.train_rgb_image = np.concatenate((self.train_rgb_image, datasets[i].train_rgb_image), axis=0)
            self.train_rgb_label = np.concatenate((self.train_rgb_label, datasets[i].train_rgb_label), axis=0)
            self.train_rgb_cam = np.concatenate((self.train_rgb_cam, datasets[i].train_rgb_cam), axis=0)
            self.train_ir_image = np.concatenate((self.train_ir_image, datasets[i].train_ir_image), axis=0)
            self.train_ir_label = np.concatenate((self.train_ir_label, datasets[i].train_ir_label), axis=0)
            self.train_ir_cam = np.concatenate((self.train_ir_cam, datasets[i].train_ir_cam), axis=0)

        self.transform_rgb1 = datasets[0].transform_rgb1
        self.transform_rgb2 = datasets[0].transform_rgb2
        self.transform_ir = datasets[0].transform_ir
        self.rIndex = None
        self.iIndex = None

    def __getitem__(self, index):
        img1, target1, cam1 = self.train_rgb_image[self.rIndex[index]], self.train_rgb_label[self.rIndex[index]], self.train_rgb_cam[self.rIndex[index]]
        img2, target2, cam2 = self.train_ir_image[self.iIndex[index]], self.train_ir_label[self.iIndex[index]], self.train_ir_cam[self.iIndex[index]]
        img_rgb1 = self.transform_rgb1(img1)
        # img_rgb2 = self.transform_rgb2(img1)
        img_rgb2 = 0
        img_ir = self.transform_ir(img2)
        return img_rgb1, img_rgb2, img_ir, target1, target2, cam1, cam2

    def __len__(self):
        return len(self.train_rgb_label)
