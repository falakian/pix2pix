import torch
import torch.nn as nn
from typing import Dict, List, Any
from .base_model import BaseModel
from . import networks

class Pix2PixModel(BaseModel):
    """Pix2Pix model implementing a U-Net generator and PatchGAN discriminator for image-to-image translation."""
    
    def __init__(self, opt: Dict[str, Any]):
        """
        Initialize the Pix2Pix model with generator and discriminator networks.

        Args:
            opt: Configuration dictionary containing:
                - input_nc: Number of input channels
                - output_nc: Number of output channels
                - ngf: Number of generator filters
                - netG: Generator architecture type
                - norm: Normalization type
                - no_dropout: Whether to disable dropout
                - init_type: Network initialization type
                - init_gain: Initialization gain
                - ndf: Number of discriminator filters
                - netD: Discriminator architecture type
                - n_layers_D: Number of discriminator layers
                - gan_mode: GAN loss mode
                - lr: Learning rate
                - beta1: Adam optimizer beta1 parameter
                - lambda_L1: Weight for L1 loss
        """
        super().__init__(opt)
        
        # Define loss names for tracking
        self.loss_names: List[str] = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        
        # Define model names based on training mode
        self.model_names: List[str] = ['G', 'D'] if self.isTrain else ['G']
        
        # Initialize generator network
        self.netG: nn.Module = networks.define_G(
            input_nc=opt.get('input_nc', 3),
            output_nc=opt.get('output_nc', 3),
            ngf=opt.get('ngf', 64),
            netG=opt.get('netG', 'unet_256'),
            norm=opt.get('norm', 'batch'),
            use_dropout=not opt.get('no_dropout', False),
            init_type=opt.get('init_type', 'normal'),
            init_gain=opt.get('init_gain', 0.02)
        )
        
        # Initialize discriminator network (only in training mode)
        if self.isTrain:
            self.netD: nn.Module = networks.define_D(
                input_nc=opt.get('input_nc', 3) + opt.get('output_nc', 3),
                ndf=opt.get('ndf', 64),
                netD=opt.get('netD', 'basic'),
                n_layers_D=opt.get('n_layers_D', 3),
                norm=opt.get('norm', 'batch'),
                init_type=opt.get('init_type', 'normal'),
                init_gain=opt.get('init_gain', 0.02)
            )
            
            # Define loss functions
            self.criterionGAN = networks.GANLoss(opt.get('gan_mode', 'lsgan')).to(self.device)
            self.criterionL1 = nn.L1Loss()
            
            # Initialize optimizers
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.get('lr', 0.0002),
                betas=(opt.get('beta1', 0.5), 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt.get('lr', 0.0002),
                betas=(opt.get('beta1', 0.5), 0.999)
            )
            self.optimizers = [self.optimizer_G, self.optimizer_D]
        
        # Define visualization names
        self.visual_names: List[str] = ['real_input', 'output_generator', 'real_output']

    def set_input(self, data: Dict[str, Any]) -> None:
        """
        Unpack and preprocess input data from dataloader.

        Args:
            data: Dictionary containing:
                - input: Input image tensor
                - output: Target image tensor
                - input_paths: List of input image paths
                - output_paths: List of target image paths
        """
        self.real_input: torch.Tensor = data['input'].to(self.device)
        self.real_output: torch.Tensor = data['output'].to(self.device)
        self.input_paths: List[str] = data['input_paths']
        self.output_paths: List[str] = data['output_paths']

    def forward(self) -> None:
        """Run the generator forward pass to produce output images."""
        self.output_generator: torch.Tensor = self.netG(self.real_input)

    def backward_D(self) -> None:
        """Calculate and backpropagate discriminator losses."""
        # Create fake input-output pair
        fake_AB = torch.cat((self.real_input, self.output_generator), dim=1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake: torch.Tensor = self.criterionGAN(pred_fake, False)
        
        # Create real input-output pair
        real_AB = torch.cat((self.real_input, self.real_output), dim=1)
        pred_real = self.netD(real_AB)
        self.loss_D_real: torch.Tensor = self.criterionGAN(pred_real, True)
        
        # Combine losses and backpropagate
        self.loss_D: torch.Tensor = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self) -> None:
        """Calculate and backpropagate generator losses (GAN + L1)."""
        # Calculate GAN loss
        fake_AB = torch.cat((self.real_input, self.output_generator), dim=1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN: torch.Tensor = self.criterionGAN(pred_fake, True)
        
        # Calculate L1 loss
        self.loss_G_L1: torch.Tensor = self.criterionL1(self.output_generator, self.real_output) * self.opt.get('lambda_L1', 100.0)
        
        # Combine losses and backpropagate
        self.loss_G: torch.Tensor = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self) -> None:
        """Update network weights: optimize discriminator first, then generator."""
        # Forward pass
        self.forward()
        
        # Update discriminator
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        
        # Update generator
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()