import numpy as np
from PIL import Image
from config import get_config
import os
args = get_config()
data_path = os.path.join(args.datasets_path, 'LLCM/')
save_path = os.path.join(args.datasets_path, 'LLCM/')


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label


photo_width = args.img_w
photo_height = args.img_h


train_color_list = data_path + '/idx/train_vis.txt'  # RGB photos
train_thermal_list = data_path + '/idx/train_nir.txt'  # Thermal photos


color_img_file, train_color_label = load_data(train_color_list)
thermal_img_file, train_thermal_label = load_data(train_thermal_list)

train_color_image = []
train_color_cmid = []
for i in range(len(color_img_file)):
    img = Image.open(data_path + color_img_file[i])
    img = img.resize((photo_width, photo_height), Image.LANCZOS)
    pix_array = np.array(img)
    cmid = int(color_img_file[i].split('/')[-1].split('_')[1][1:])
    train_color_image.append(pix_array)
    train_color_cmid.append(cmid)

train_color_image = np.array(train_color_image)
train_color_label = np.array(train_color_label)
train_color_cmid = np.array(train_color_cmid)

train_thermal_image = []
train_thermal_cmid = []
for i in range(len(thermal_img_file)):
    img = Image.open(data_path + thermal_img_file[i])
    img = img.resize((photo_width, photo_height), Image.LANCZOS)
    pix_array = np.array(img)
    cmid = int(thermal_img_file[i].split('/')[-1].split('_')[1][1:])
    train_thermal_image.append(pix_array)
    train_thermal_cmid.append(cmid)
    # print(pix_array.shape)
train_thermal_image = np.array(train_thermal_image)
train_thermal_label = np.array(train_thermal_label)
train_thermal_cmid = np.array(train_thermal_cmid)

# rgb imges
np.save(save_path + '/train_rgb_resized_img.npy', train_color_image)
np.save(save_path + '/train_rgb_resized_label.npy', train_color_label)
np.save(save_path + '/train_rgb_resized_cam.npy', train_color_cmid)

# ir imges
np.save(save_path + '/train_ir_resized_img.npy', train_thermal_image)
np.save(save_path + '/train_ir_resized_label.npy', train_thermal_label)
np.save(save_path + '/train_ir_resized_cam.npy', train_thermal_cmid)

print('Done!')