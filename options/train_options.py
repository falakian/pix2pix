from dataclasses import dataclass
from .base_options import BaseOptions
import argparse

@dataclass
class TrainOptions(BaseOptions):
    """Configuration class for training-specific parameters, extending BaseOptions."""
    
    def __init__(self) -> None:
        """Initialize TrainOptions with training-specific defaults and call parent constructor."""
        super().__init__()
        self.isTrain: bool = True

    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Configure training-specific command-line arguments, extending base options.

        Args:
            parser: ArgumentParser object from BaseOptions to add training arguments to.

        Returns:
            Configured ArgumentParser object with training parameters.
        """
        # Inherit base options
        parser = super().initialize(parser)

        # Training process parameters
        parser.add_argument(
            '--print_freq',
            type=int,
            default=100,
            help='Frequency of printing training progress to console (in iterations)')
        parser.add_argument(
            '--save_epoch',
            type=int,
            default=5,
            help='Save model checkpoint every N epochs')
        parser.add_argument(
            '--continue_train',
            action='store_true',
            help='Resume training from a previous checkpoint')
        parser.add_argument(
            '--epoch_count',
            type=int,
            default=1,
            help='Starting epoch number for training (useful for resuming)')
        parser.add_argument(
            '--n_epochs',
            type=int,
            default=30,
            help='Number of epochs with the initial learning rate')
        parser.add_argument(
            '--n_epochs_decay',
            type=int,
            default=70,
            help='Number of epochs to linearly decay learning rate to zero')
        parser.add_argument(
            '--pretrain_epochs',
            type=int,
            default=5,
            help='Number of initial epochs where only the generator (G) is trained')
        
        # Optimizer parameters
        parser.add_argument(
            '--lr_G',
            type=float,
            default=0.0002,
            help='Initial learning rate for the generator optimizer')
        parser.add_argument(
            '--lr_D',
            type=float,
            default=0.0002,
            help='Initial learning rate for the discriminator optimizer')
        
        parser.add_argument(
            '--beta1',
            type=float,
            default=0.5,
            help='Momentum term (beta1) for the Adam optimizer')
        parser.add_argument(
            '--lambda_lpips',
            type=float,
            default=1.0,
            help='Weight for lpips loss in the objective function')
        parser.add_argument(
            '--lambda_fm',
            type=float,
            default=10.0,
            help='Weight for feature matching loss')
        
        # Learning rate scheduling
        parser.add_argument(
            '--lr_policy',
            type=str,
            default='linear',
            choices=['linear', 'step', 'plateau', 'cosine'],
            help='Learning rate decay policy')
        parser.add_argument(
            '--lr_decay_iters',
            type=int,
            default=50,
            help='Multiply learning rate by gamma every lr_decay_iters iterations')

        # GAN-specific parameters
        parser.add_argument(
            '--gan_mode',
            type=str,
            default='vanilla',
            choices=['vanilla', 'lsgan', 'hinge'],
            help='GAN objective type (vanilla: cross-entropy, lsgan: least squares, hinge')

        return parser