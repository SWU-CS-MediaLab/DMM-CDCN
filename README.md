## Lifelong Visible-Infrared Person Re-identification via Replay Samples Domain-Modality-Mix Reconstruction and Cross-domain Cognitive Network


## Requirements and Installation
We recommended the following dependencies.
*  Python 3.7
*  PyTorch 1.13.0
*  Torchvision 0.14.0
*  Tqdm 4.66.2

## Before Training
Download the datasets, the path to get four datasets are below:
```
RegDB: http://dm.dongguk.edu/link.html or https://paperswithcode.com/dataset/regdb
SYSU-MM01: https://github.com/wuancong/SYSU-MM01
LLCM: https://github.com/ZYK100/LLCM
HITSZ-VCM: https://github.com/VCM-project233/HITSZ-VCM-data 
```

Prepare the training and testing data. The folder structure should be:
```
Datasets
└─── RegDB
|	├─── idx
|	├─── Thermal
|	└─── Visible
└─── SYSU-MM01
|	├─── exp
|	├─── cam1
|	├─── cam2
|	└─── ......
└─── LLCM
|	├─── idx
|	├─── vis
|	└─── nir
|	├─── test_vis
|	└─── test_nir
└─── VCM
	├─── info
	├─── Train
	├─── Test
```
The 'Datasets' folder contains four datasets,  while the internal directory of each dataset is not modified.

Then, run preprocess programs to preprocess the images in SYSU-MM01, LLCM and VCM (The short name of HITSZ_VCM).
```
python pre_process_sysu.py 	--datasets_path '/my-tmp/Datasets/' 
python pre_process_llcm.py 	--datasets_path '/my-tmp/Datasets/' 
python pre_process_vcm.py 	--datasets_path '/my-tmp/Datasets/' 
```

## Training

To train LVIReID, run the training script below.
```
python main.py \
	--total_epoch 60 \
	--test_frequency 10 \
	--batch_size 8 \
	--train_datasets  'regdb' 'sysu' 'llcm' 'vcm'  \
	--log_path '/my-tmp/log/' \
	--model_path '/my-tmp/log/best_model/' \
	--vis_log_path '/my-tmp/log/vis_log/' \
	--datasets_path '/my-tmp/Datasets/' \
	--ex_name 'Experment 1' \
	--test_mode 'vti' \
	--lr 0.1 \
	--sample_reply 't' \
	--reply_type 'pcb' \
	--kd_loss 'kl' \
	--method 'pcb' \
	--gcn 't' \
	--gpu 0
```

## Testing

To test LVIReID, run the testing script below. Notably, only the trained dataset can be tested because the model should know the information of cameras in the domain, and the order of testing datasets should be the same as training, the final metricses (except R-1 and mAP) are processed using other tools.
```
python main.py \
	--test_datasets 'regdb' 'sysu' 'llcm' 'vcm' \
	--log_path '/my-tmp/log/' \
	--model_path '/my-tmp/log/best_model/' \
	--vis_log_path '/my-tmp/log/vis_log/' \
	--datasets_path '/my-tmp/Datasets/' \
	--resume '/my-tmp/log/my-net/LVIReID_Net_GPU0.pth' \
	--lr 0.1 \
	--test_only 't' \
	--debug 't' \
	--test_mode 'itv' \
	--reply_type 'pcb' \
	--kd_loss 'kl' \
	--method 'pcb' \
	--gcn 't' \
	--gpu 0 \
```



