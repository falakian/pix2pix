import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
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
        self.loss_names: List[str] = ['G_GAN','G_LPIPS', 'G_L1', 'D_real', 'D_fake']
        self.loss_G_GAN = 0
        self.loss_G_LPIPS = 0
        self.loss_G_L1 = 0
        self.loss_D_real = 0
        self.loss_D_fake = 0

        # Define model names based on training mode
        self.model_names: List[str] = ['G', 'D'] if self.isTrain else ['G']
        
        # Initialize generator network
        self.netG: nn.Module = networks.define_G(
            input_nc=opt.input_nc,
            output_nc=opt.output_nc,
            ngf=opt.ngf,
            num_downs=opt.num_downs,
            height_down_layers=opt.height_down_layers,
            norm=opt.norm,
            use_dropout=not opt.no_dropout,
            init_type=opt.init_type,
            init_gain=opt.init_gain
        )
        
        # Initialize discriminator network (only in training mode)
        if self.isTrain:
            self.netD: nn.Module = networks.define_D(
                input_nc=opt.input_nc + opt.output_nc,
                ndf=opt.ndf,
                n_layers_D=opt.n_layers_D,
                norm=opt.norm,
                init_type=opt.init_type,
                init_gain=opt.init_gain
            )
            
            # Define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = nn.L1Loss()
            self.criterionLPIPS = lpips.LPIPS(net='alex').to(self.device)

            # Initialize optimizers
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr_G,
                betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt.lr_D,
                betas=(opt.beta1, 0.999)
            )
            self.optimizers = [self.optimizer_G, self.optimizer_D]

            
            self.scales = [0.5, 0.25]
            self.scale_weights = [1, 0.5]
            self.lambda_lpips = opt.lambda_lpips
            self.pretrain_epochs = opt.pretrain_epochs

        # Define visualization names
        self.visual_names: List[str] = ['real_input', 'output_generator', 'real_output']

    def set_input(self, data: Dict[str, Any]) -> None:
        """
        Unpack and preprocess input data from dataloader.

        Args:
            data: Dictionary containing:
                - input: Input image tensor
                - output: Target image tensor
        """
        self.real_input: torch.Tensor = data['input'].to(self.device)
        self.real_output: torch.Tensor = data['output'].to(self.device)

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

    def multiscale_l1_loss(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for scale, weight in zip(self.scales, self.scale_weights):
            if scale != 1.0:
                fake_scaled = F.interpolate(fake, scale_factor=scale, mode='bilinear', align_corners=False)
                real_scaled = F.interpolate(real, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                fake_scaled, real_scaled = fake, real
            total_loss += weight * self.criterionL1(fake_scaled, real_scaled)
        return total_loss
    
    def backward_G(self, epoch: int) -> None:
        """Calculate and backpropagate generator losses """
        # Multi-scale L1
        self.loss_G_L1: torch.Tensor = self.multiscale_l1_loss(self.output_generator, self.real_output)

        # LPIPS
        scale_lpips = 0.25
        generator_scaled = F.interpolate(self.output_generator, scale_factor=scale_lpips, mode='bilinear', align_corners=False)
        real_scaled = F.interpolate(self.real_output, scale_factor=scale_lpips, mode='bilinear', align_corners=False)
        self.loss_G_LPIPS: torch.Tensor = self.criterionLPIPS(generator_scaled, real_scaled).mean() * self.lambda_lpips

        self.loss_G: torch.Tensor = self.loss_G_L1 + self.loss_G_LPIPS

        if epoch >= self.pretrain_epochs:
            fake_AB = torch.cat((self.real_input, self.output_generator), dim=1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN: torch.Tensor = self.criterionGAN(pred_fake, True, is_generator=True)
            self.loss_G += self.loss_G_GAN
        else:
            self.loss_G_GAN = torch.tensor(0.0, device=self.device)
        
        self.loss_G.backward()

    def optimize_parameters(self, epoch: int) -> None:
        """Update network weights: optimize discriminator first, then generator."""
        # Forward pass
        self.forward()
        
        # Update only generator
        if epoch < self.pretrain_epochs:
            self.set_requires_grad(self.netD, False)
            self.optimizer_G.zero_grad()
            self.backward_G(epoch)
            self.optimizer_G.step()
        else:
            # Update discriminator
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            
            # Update generator
            self.set_requires_grad(self.netD, False)
            self.optimizer_G.zero_grad()
            self.backward_G(epoch)
            self.optimizer_G.step()