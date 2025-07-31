import argparse
from pathlib import Path
import torch
import util

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # basic parameters
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, etc)')
        parser.add_argument('--name', type=str, default='experiment_pix2pix', help='name of the experiment')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: 0,1,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        # model params
        parser.add_argument('--model', type=str, default='pix2pix', help='Only pix2pix is supported')
        parser.add_argument('--input_nc', type=int, default=3, help='# input channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# output channels')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--input_dir', type=str, default='trainA', help='subfolder containing input images (e.g., trainA, testB)')
        parser.add_argument('--output_dir', type=str, default='trainB', help='subfolder containing output images (e.g., trainB, testA)')

        # dataset loading
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | resize | none]')
        parser.add_argument('--no_flip', action='store_true', help='disable flipping for data augmentation')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # extra
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix to save results')
        parser.add_argument('--results_dir', type=str, default='./results', help='where results are saved')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = '------------ Options -------------\n'
        for k, v in sorted(vars(opt).items()):
            default = self.parser.get_default(k)
            comment = f'\t[default: {default}]' if v != default else ''
            message += f'{k:>25}: {v:<30}{comment}\n'
        message += '-------------- End ----------------'
        print(message)

        # Save
        expr_dir = Path(opt.checkpoints_dir) / opt.name
        util.mkdirs(expr_dir)
        file_name = expr_dir / 'opt.txt'
        with open(file_name, 'wt') as f:
            f.write(message + '\n')

    def parse(self):
        opt = self.gather_options()
        if opt.suffix != '':
            opt.name = f"{opt.name}_{opt.suffix.format(**vars(opt))}"
        self.print_options(opt)

        # GPU
        opt.gpu_ids = [int(id) for id in opt.gpu_ids.split(',') if int(id) >= 0]
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
