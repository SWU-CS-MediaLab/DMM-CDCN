import copy
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class RegDBData(Dataset):
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

        self.transform = transform
        self.rIndex = None
        self.iIndex = None

    def __getitem__(self, index):
        img1, target1 = self.train_rgb_image[self.rIndex[index]], self.train_rgb_label[self.rIndex[index]]
        img2, target2 = self.train_ir_image[self.iIndex[index]], self.train_ir_label[self.iIndex[index]]
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_rgb_label)

    @staticmethod
    def load_data(input_data_path, max_pid):
        with open(input_data_path) as f:
            data_file_list = open(input_data_path, 'rt').read().splitlines()
            file_image = [s.split(' ')[0] for s in data_file_list]
            file_label = [int(s.split(' ')[1]) + max_pid for s in data_file_list]

        return file_image, file_label


class SYSUData(Dataset):
    def __init__(self, data_dir,  max_pid, transform=None):

        # Load training images (path) and labels
        self.train_rgb_label = np.load(data_dir + 'train_rgb_resized_label.npy')
        self.train_rgb_label = self.train_rgb_label + max_pid

        self.train_ir_label = np.load(data_dir + 'train_ir_resized_label.npy')
        self.train_ir_label = self.train_ir_label + max_pid
        
        # BGR to RGB
        self.train_rgb_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_ir_image = np.load(data_dir + 'train_ir_resized_img.npy')

        self.transform = transform
        self.rIndex = None
        self.iIndex = None

    def __getitem__(self, index):
            img1, target1 = self.train_rgb_image[self.rIndex[index]], self.train_rgb_label[self.rIndex[index]]
            img2, target2 = self.train_ir_image[self.iIndex[index]], self.train_ir_label[self.iIndex[index]]
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_rgb_label)


class LLCMData(Dataset):
    def __init__(self, data_dir, max_pid, transform=None):
        # BGR to RGB
        self.train_rgb_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_rgb_label = np.load(data_dir + 'train_rgb_resized_label.npy')
        self.train_rgb_label = self.train_rgb_label + max_pid

        # BGR to RGB
        self.train_ir_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_ir_label = np.load(data_dir + 'train_ir_resized_label.npy')
        self.train_ir_label = self.train_ir_label + max_pid

        self.transform = transform
        self.rIndex = None
        self.iIndex = None

    def __getitem__(self, index):
        img1, target1 = self.train_rgb_image[self.rIndex[index]], self.train_rgb_label[self.rIndex[index]]
        img2, target2 = self.train_ir_image[self.iIndex[index]], self.train_ir_label[self.iIndex[index]]
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_rgb_label)


class VCMData(Dataset):
    def __init__(self, data_dir, max_pid, transform=None):
        self.train_rgb_path = np.load(data_dir + 'train_rgb_path_img.npy')
        self.train_rgb_label = np.load(data_dir + 'train_rgb_path_label.npy')
        self.train_rgb_label = self.train_rgb_label + max_pid
        self.train_ir_path = np.load(data_dir + 'train_ir_path_img.npy')
        self.train_ir_label = np.load(data_dir + 'train_ir_path_label.npy')
        self.train_ir_label = self.train_ir_label + max_pid
        self.train_rgb_image = []
        self.train_ir_image = []
        print('Loading VCM training images...')
        for i in tqdm(range(len(self.train_rgb_path))):
            img = Image.open(self.train_rgb_path[i])
            img = img.resize((128, 256), Image.LANCZOS)
            self.train_rgb_image.append(np.array(img))
        for i in tqdm(range(len(self.train_ir_path))):
            img = Image.open(self.train_ir_path[i])
            img = img.resize((128, 256), Image.LANCZOS)
            self.train_ir_image.append(np.array(img))
        self.train_rgb_image = np.array(self.train_rgb_image)
        self.train_ir_image = np.array(self.train_ir_image)

        self.transform = transform
        self.rIndex = None
        self.iIndex = None

    def __getitem__(self, index):
            img1, target1 = self.train_rgb_image[self.rIndex[index]], self.train_rgb_label[self.rIndex[index]]
            img2, target2 = self.train_ir_image[self.iIndex[index]], self.train_ir_label[self.iIndex[index]]
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_rgb_label)


class ReplyTmpData(Dataset):
    def __init__(self, img, label, cmid,transform):
        self.img = img
        self.label = label
        self.cmid = cmid
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        img, target, cmid = self.img[index], self.label[index], self.cmid[index]
        img_original = copy.deepcopy(img)
        # print(img_original.shape)
        img_original_pil = Image.fromarray(img_original)
        img_original_resized = img_original_pil.resize((128, 256), resample=Image.BILINEAR)
        img_original = np.array(img_original_resized)
        img = self.transform(img)
        # print(img.shape)
        # print(img_original.shape)
        return img, target, img_original, cmid


class JointData(Dataset):
    def __init__(self, datasets=[]):
        self.train_rgb_image = datasets[0].train_rgb_image
        self.train_rgb_label = datasets[0].train_rgb_label
        self.train_ir_image = datasets[0].train_ir_image
        self.train_ir_label = datasets[0].train_ir_label
        self.transform = datasets[0].transform

        for i in range(1, len(datasets)):
            self.train_rgb_image = np.concatenate((self.train_rgb_image, datasets[i].train_rgb_image), axis=0)
            self.train_rgb_label = np.concatenate((self.train_rgb_label, datasets[i].train_rgb_label), axis=0)
            self.train_ir_image = np.concatenate((self.train_ir_image, datasets[i].train_ir_image), axis=0)
            self.train_ir_label = np.concatenate((self.train_ir_label, datasets[i].train_ir_label), axis=0)

        self.rIndex = None
        self.iIndex = None

    def __getitem__(self, index):
        img1, target1 = self.train_rgb_image[self.rIndex[index]], self.train_rgb_label[self.rIndex[index]]
        img2, target2 = self.train_ir_image[self.iIndex[index]], self.train_ir_label[self.iIndex[index]]
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_rgb_label)
