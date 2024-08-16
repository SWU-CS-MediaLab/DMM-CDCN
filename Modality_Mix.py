import copy

import numpy as np
import torch
import random


def  Modal_Mix(img1, img2, label1, label2, crossp=0.05):
    img1 = copy.deepcopy(img1)
    img2 = copy.deepcopy(img2)
    label1 = copy.deepcopy(label1)
    label2 = copy.deepcopy(label2)

    label1 = label1.cpu().numpy()
    label2 = label2.cpu().numpy()
    blocks = 4
    for label in np.unique(label1):
        indices1 = np.where(label1 == label)[0]
        for img_idx in indices1:
            for block_idx in range(blocks):
                left_indices = img1.shape[2] // blocks * block_idx
                right_indices = img1.shape[2] // blocks * (block_idx + 1)
                channel_type = random.randint(0, 2)
                indices2 = np.where(label2 == label)[0]

                if random.uniform(0, 1) > crossp or len(indices2) == 0:
                    random_img_idx = np.random.choice(indices1, 1)
                    if channel_type == 0:
                        channel_select_1 = random.randint(0, 2)
                        img1[img_idx, channel_select_1, left_indices: right_indices, :] = img1[random_img_idx, channel_select_1, left_indices: right_indices, :]
                    elif channel_type == 1:
                        channel_select_1 = np.random.choice([0, 1, 2], 2)
                        img1[img_idx, channel_select_1, left_indices: right_indices, :] = img1[random_img_idx, channel_select_1, left_indices: right_indices, :]
                    elif channel_type == 2:
                        img1[img_idx, :, left_indices: right_indices, :] = img1[random_img_idx, :, left_indices: right_indices, :]
                else:
                    # continue
                    random_img_idx = np.random.choice(indices2, 1)
                    if channel_type == 0:
                        channel_select_1 = random.randint(0, 2)
                        img1[img_idx, channel_select_1, left_indices: right_indices, :] = img2[random_img_idx, channel_select_1, left_indices: right_indices, :]
                    elif channel_type == 1:
                        channel_select_1 = np.random.choice([0, 1, 2], 2)
                        img1[img_idx, channel_select_1, left_indices: right_indices, :] = img2[random_img_idx, channel_select_1, left_indices: right_indices, :]
                    elif channel_type == 2:
                        img1[img_idx, :, left_indices: right_indices, :] = img2[random_img_idx, :, left_indices: right_indices, :]

    return img1


def  Reply_Modal_Mix(img1, img2, label1, label2, crossd=0.5, crossp=0.05):
    img1 = copy.deepcopy(img1)
    img2 = copy.deepcopy(img2)
    label1 = copy.deepcopy(label1)
    label2 = copy.deepcopy(label2)
    label1 = label1.cpu().numpy()
    label2 = label2.cpu().numpy()

    blocks = 4
    for label in np.unique(label1):
        indices1 = np.where(label1 == label)[0]
        for img_idx in indices1:
            for block_idx in range(blocks):
                left_indices = img1.shape[2] // blocks * block_idx
                right_indices = img1.shape[2] // blocks * (block_idx + 1)
                channel_type = random.randint(0, 2)
                indices4 = np.where(label2 == label)[0]

                if random.uniform(0, 1) > crossp or len(indices4) == 0:
                    indices2 = np.where(label1 == label)[0]
                    indices3 = np.where(label1 != label)[0]
                    if random.uniform(0, 1) > crossd:
                        random_img_idx = np.random.choice(indices2, 1)
                    else:
                        random_img_idx = np.random.choice(indices3, 1)

                    if channel_type < 3:
                        channel_idx_2 = random.randint(0, 2)
                        img1[img_idx, channel_type, left_indices: right_indices, :] = img1[random_img_idx, channel_idx_2, left_indices: right_indices, :]
                    elif channel_type == 3:
                        channel_select_1 = np.random.choice([0, 1, 2], 2)
                        channel_select_2 = np.random.choice([0, 1, 2], 2)
                        img1[img_idx, channel_select_1, left_indices: right_indices, :] = img1[random_img_idx, channel_select_2, left_indices: right_indices, :]
                    elif channel_type == 4:
                        img1[img_idx, :, left_indices: right_indices, :] = img1[random_img_idx, :, left_indices: right_indices, :]
                else:
                    continue
        return img1
