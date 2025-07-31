
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.root = opt.dataroot

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass

def get_params(opt, size):
    width, height = size
    x = width//2
    y = height//2
    if opt.preprocess != 'none':
        new_width = new_height = opt.load_size
        x = random.randint(0, max(0, new_width - opt.crop_size))
        y = random.randint(0, max(0, new_height - opt.crop_size))

    flip = random.random() > 0.5 if not opt.no_flip else False

    return {'crop_pos': (x, y), 'flip': flip}

import torchvision.transforms as transforms
from PIL import Image

def get_transform(opt, params=None, grayscale=False):
    transform_list = []

    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if 'resize' in opt.preprocess:
        transform_list.append(transforms.Resize([opt.load_size, opt.load_size], transforms.InterpolationMode.BICUBIC))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: BaseDataset.crop(img, params['crop_pos'], opt.crop_size)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: BaseDataset.flip(img)))

    transform_list += [transforms.ToTensor()]
    if grayscale:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]


    return transforms.Compose(transform_list)

class BaseDataset:
    @staticmethod
    def crop(img, pos, size):
        ow, oh = img.size
        x1, y1 = pos
        tw = th = size
        if ow >= tw and oh >= th:
            return img.crop((x1, y1, x1 + tw, y1 + th))
        return img

    @staticmethod
    def flip(img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    