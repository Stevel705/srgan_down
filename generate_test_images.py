import argparse
import time
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import SRDatasetVal

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=2, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_2_7.pth', type=str, help='generator model epoch name')
parser.add_argument('--dir_name', default='../test/HR', type=str, help='test HR resolution image name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name
DATASET_DIR = opt.dir_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

root = os.path.split(DATASET_DIR) 

test_path_lr = root[0] + "/LR/"
if not os.path.exists(test_path_lr):
    os.makedirs(test_path_lr)


val_set  = SRDatasetVal(DATASET_DIR)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

with torch.no_grad():
    val_bar = tqdm(val_loader)
    val_images = []
    for val_hr, image_file  in val_bar:
        hr = val_hr
        if torch.cuda.is_available():
            hr = hr.cuda()

        lr = model(hr)
        out_img = ToPILImage()(lr[0].data.cpu())
        out_img.save(test_path_lr + image_file[0])

# for image_file in os.listdir(DATASET_DIR):
#     print(image_file)
#     image = Image.open(os.path.join(DATASET_DIR, image_file))
#     image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
#     if TEST_MODE:
#         image = image.cuda()

#     start = time.process_time()
#     out = model(image)
#     elapsed = (time.process_time() - start)
#     print('cost ' + str(elapsed) + 's')
#     out_img = ToPILImage()(out[0].data.cpu())
#     out_img.save('../test/LR/' + image_file)