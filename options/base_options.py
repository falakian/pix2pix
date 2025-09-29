import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

@dataclass
class BaseOptions:
    """Configuration class for managing experiment parameters with type hints and comprehensive documentation."""
    
    def __init__(self) -> None:
        """Initialize the BaseOptions class with default state."""
        self.initialized: bool = False
        self.parser: Optional[argparse.ArgumentParser] = None
        self.opt: Optional[argparse.Namespace] = None

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Configure command-line arguments for the experiment.

        Args:
            parser: ArgumentParser object to add arguments to.

        Returns:
            Configured ArgumentParser object.
        """
        # Basic experiment parameters
        parser.add_argument(
            '--dataroot',
            type=str,
            required=True,
            help='Path to dataset directory containing subfolders (e.g., trainA, trainB)')
        parser.add_argument(
            '--name',
            type=str,
            default='experiment_pix2pix',
            help='Name of the experiment for identification')
        parser.add_argument(
            '--checkpoints_dir',
            type=str,
            default='./checkpoints',
            help='Directory to save model checkpoints')

        # Model architecture parameters
        parser.add_argument(
            '--model',
            type=str,
            default='pix2pix',
            choices=['pix2pix'],
            help='Model type (currently only pix2pix is supported)')
        parser.add_argument(
            '--input_nc',
            type=int,
            default=3,
            help='Number of input channels for the generator')
        parser.add_argument(
            '--output_nc',
            type=int,
            default=3,
            help='Number of output channels for the generator')
        parser.add_argument(
            '--ngf',
            type=int,
            default=64,
            help='Number of filters in the last convolutional layer of the generator')
        parser.add_argument(
            '--ndf',
            type=int,
            default=64,
            help='Number of filters in the first convolutional layer of the discriminator')
        # parser.add_argument(
        #     '--netD',
        #     type=str,
        #     default='basic',
        #     choices=['basic', 'n_layers'],
        #     help='Discriminator architecture (basic: 70x70 PatchGAN')
        # parser.add_argument(
        #     '--netG',
        #     type=str,
        #     default='unet_256',
        #     choices=['unet_256', 'unet_128'],
        #     help='Generator architecture')
        parser.add_argument(
            '--n_layers_D',
            type=int,
            default=3,
            help='Number of layers in discriminator when netD is n_layers')
        parser.add_argument(
            '--num_downs',
            type=int,
            default=8,
            help='Number of layers in generator (at least 5)')
        parser.add_argument(
            '--height_down_layers',
            type=int,
            default=8,
            help='Number of layers in the generator where the feature map height is reduced by half (at least 1)')
        parser.add_argument(
            '--norm',
            type=str,
            default='batch',
            choices=['instance', 'batch', 'none'],
            help='Normalization type for the network')
        parser.add_argument(
            '--init_type',
            type=str,
            default='normal',
            choices=['normal', 'xavier', 'kaiming', 'orthogonal'],
            help='Network weight initialization method')
        parser.add_argument(
            '--init_gain',
            type=float,
            default=0.02,
            help='Scaling factor for weight initialization')
        parser.add_argument(
            '--no_dropout',
            action='store_true',
            help='Disable dropout in the generator')
        parser.add_argument(
            '--input_dir',
            type=str,
            default='input_train',
            help='Subfolder containing input images')
        parser.add_argument(
            '--output_dir',
            type=str,
            default='output_train',
            help='Subfolder containing output images')

        # Dataset loading parameters
        parser.add_argument(
            '--batch_size',
            type=int,
            default=1,
            help='Batch size for training')
        parser.add_argument(
            "--serial_batches",
            action="store_true",
            help="if true, takes images in order to make batches, otherwise takes them randomly")
        parser.add_argument(
            '--load_size_height',
            type=int,
            default=256,
            help='Initial image height resize dimension')
        parser.add_argument(
            '--load_size_width',
            type=int,
            default=256,
            help='Initial image width resize dimension')
        parser.add_argument(
            '--crop_size_height',
            type=int,
            default=256,
            help='Final image height crop dimension')
        parser.add_argument(
            '--crop_size_height',
            type=int,
            default=256,
            help='Final image width crop dimension')
        # parser.add_argument(
        #     '--padding',
        #     type=str,
        #     default='none',
        #     choices=['white', 'random', 'replicate', 'constant', 'none'],
        #     help='Padding for images to equalize size')
        # parser.add_argument(
        #     '--constant_value_padding',
        #     type=int,
        #     default=0,
        #     help='Constant value for constant padding (e.g., 0 for black, 255 for white)')
        parser.add_argument(
            '--preprocess',
            type=str,
            default='none',
            choices=['resize_and_crop', 'crop', 'resize', 'none'],
            help='Preprocessing method for images during loading')
        parser.add_argument(
            '--no_flip',
            action='store_true',
            help='Disable random horizontal flipping for data augmentation')
        parser.add_argument(
            '--num_threads',
            type=int,
            default=4,
            help='Number of threads for data loading')
        parser.add_argument(
            '--max_dataset_size',
            type=int,
            default=float('inf'),
            help='Maximum number of samples to load from dataset')

        # Additional parameters
        parser.add_argument(
            "--epoch",
            type=str,
            default="latest",
            help="which epoch to load? set to latest to use latest cached model")
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output for debugging')
        parser.add_argument(
            '--suffix',
            type=str,
            default='',
            help='Custom suffix for experiment name formatting')
        parser.add_argument(
            '--phase',
            type=str,
            default='train',
            choices=['train', 'val', 'test'],
            help='Experiment phase')

        self.initialized = True
        return parser

    def gather_options(self) -> argparse.Namespace:
        """Parse command-line arguments and return the configuration.

        Returns:
            Parsed command-line arguments as a Namespace object.
        """
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                description='Configuration for pix2pix model training')
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt: argparse.Namespace) -> None:
        """Print and save configuration options.

        Args:
            opt: Parsed command-line arguments.
        """
        message = ['------------ Options -------------']
        for key, value in sorted(vars(opt).items()):
            default = self.parser.get_default(key)
            comment = f'\t[default: {default}]' if value != default else ''
            message.append(f'{key:>25}: {value:<30}{comment}')
        message.append('-------------- End ----------------')
        print('\n'.join(message))

        # Save configuration to file
        expr_dir: Path = Path(opt.checkpoints_dir) / opt.name
        expr_dir.mkdir(parents=True, exist_ok=True)
        with (expr_dir / 'opt.txt').open('w') as f:
            f.write('\n'.join(message) + '\n')

    def parse(self) -> argparse.Namespace:
        """Parse and process command-line arguments, including suffix handling.

        Returns:
            Processed configuration as a Namespace object.
        """
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        if opt.suffix:
            opt.name = f"{opt.name}_{opt.suffix.format(**vars(opt))}"
        self.print_options(opt)
        self.opt = opt
        return opt