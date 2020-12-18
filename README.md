# SRGAN
A PyTorch implementation of SRGAN based on CVPR 2017 paper 
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c pytorch
```
- opencv
```
conda install opencv
```

## Datasets

### Train、Val Dataset
The train and val datasets are sampled from [VOC2012](http://cvlab.postech.ac.kr/~mooyeol/pascal_voc_2012/).
Train dataset has 16700 images and Val dataset has 425 images.
Download the datasets from [here](https://pan.baidu.com/s/1xuFperu2WiYc5-_QXBemlA)(access code:5tzp), and then extract it into `data` directory.

## Usage

### Train
```
python train.py

optional arguments:
--crop_size                   training images crop size [default value is 88]
--num_epochs                  train epoch number [default value is 100]
--root_dir                    root dir with LR and HR folders

For example: python train.py --crop_size 128 --num_epochs 50 --root_dir ../train
```
The output val super resolution images are on `training_results` directory.


### Test Single Image
```
python test_image.py

optional arguments:
--test_mode                   using GPU or CPU [default value is 'GPU'](choices:['GPU', 'CPU'])
--image_name                  test low resolution image name
--model_name                  generator model epoch name [default value is netG_epoch_4_100.pth]
```
The output super resolution image are on the same directory.

### Generate LR datasers
```
python generate_test_images.py 
--dir_name                    datasets with HR images
```