from os import listdir
from os.path import join
import os
import cv2
import torch
import random
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

from typing import Optional, Tuple

class SRDataset(Dataset):
    def __init__(
            self,
            hr_dir: str,
            lr_dir: str,
            crop_size: Optional[int] = None,
            length: Optional[int] = None) -> None:
        self._hr_dir = hr_dir
        self._lr_dir = lr_dir
        self._crop_size = crop_size
        self._length = length

        samples = []
        for name in listdir(lr_dir):
            if not name.endswith(".png"):
                continue
            if not os.path.exists(os.path.join(hr_dir, name)):
                raise RuntimeError(f"File {name} does not exist in {hr_dir}")
            samples.append(name)
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples) if not self._length else self._length

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self._samples[item % len(self._samples)]
        lr_image = cv2.imread(os.path.join(self._lr_dir, name))
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.imread(os.path.join(self._hr_dir, name))
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        if hr_image.shape != (lr_image.shape[0] * 2, lr_image.shape[1] * 2, lr_image.shape[2]):
            raise RuntimeError(f"Shapes of LR and HR images mismatch for sample {name}")

        lr_image = torch.from_numpy(lr_image).permute(2, 0, 1).float() / 255.
        hr_image = torch.from_numpy(hr_image).permute(2, 0, 1).float() / 255.

        if self._crop_size is not None:
            x_start = random.randint(0, lr_image.shape[1] - self._crop_size)
            y_start = random.randint(0, lr_image.shape[2] - self._crop_size)

            lr_image = lr_image[
                       :,
                       x_start:x_start + self._crop_size,
                       y_start:y_start + self._crop_size]
            hr_image = hr_image[
                       :,
                       x_start * 2:x_start * 2 + self._crop_size * 2,
                       y_start * 2:y_start * 2 + self._crop_size * 2]
        # return lr_image, hr_image
        return hr_image, lr_image 

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        # return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
        return  ToTensor()(hr_image), ToTensor()(hr_restore_img),  ToTensor()(lr_image),


    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)



class SRDatasetVal(Dataset):
    def __init__(
            self,
            hr_dir: str,
            crop_size: Optional[int] = None,
            length: Optional[int] = None) -> None:
        self._hr_dir = hr_dir
        self._crop_size = crop_size
        self._length = length

        samples = []
        for name in listdir(hr_dir):
            if not name.endswith(".png"):
                continue
            samples.append(name)
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples) if not self._length else self._length

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self._samples[item % len(self._samples)]
        hr_image = cv2.imread(os.path.join(self._hr_dir, name))
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        hr_image = torch.from_numpy(hr_image).permute(2, 0, 1).float() / 255.

        if self._crop_size is not None:
            hr_image = hr_image[
                       :,
                       x_start * 2:x_start * 2 + self._crop_size * 2,
                       y_start * 2:y_start * 2 + self._crop_size * 2]
        return hr_image, name
