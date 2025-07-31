import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_input = os.path.join(opt.dataroot, opt.input_dir)  # e.g., ./datasets/facades/trainA
        self.dir_output = os.path.join(opt.dataroot, opt.output_dir)  # e.g., ./datasets/facades/trainB
        self.input_paths = sorted(make_dataset(self.dir_input, opt.max_dataset_size))  # get input image paths
        self.output_paths = sorted(make_dataset(self.dir_output, opt.max_dataset_size))  # get output image paths
        assert len(self.input_paths) == len(self.output_paths), f"Mismatched dataset sizes: {len(self.input_paths)} images in input, {len(self.output_paths)} images in output"
        assert self.opt.load_size >= self.opt.crop_size, "crop_size should be smaller than the size of loaded image"
        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc


    def __getitem__(self, index):
        input_path = self.input_paths[index]
        output_path = self.output_paths[index]
        input_img = Image.open(input_path).convert('RGB')
        output_img = Image.open(output_path).convert('RGB')

        params = self.get_params(self.opt, input_img.size)
        input_img = self.get_transform(input_img, params)
        output_img = self.get_transform(output_img, params)
        return {'input': input_img, 'output': output_img, 'input_paths': input_path, 'output_paths': output_path}

    def __len__(self):
        return min(len(self.input_paths), len(self.output_paths), self.opt.max_dataset_size)
