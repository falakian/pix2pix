import random
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import lpips

from utile.DiffAugment_pytorch import DiffAugment
from .base_model import BaseModel
from .ocr_kraken import KrakenOCRWrapper
from . import networks, losses

class Pix2PixModel(BaseModel):
    """
    Pix2Pix model implementing a U-Net generator and a multi-scale PatchGAN discriminator
    for image-to-image translation tasks.
    """
    
    def __init__(self, opt: Dict[str, Any]):
        """
        Initialize the Pix2Pix model.

        Args:
            opt: Configuration dictionary containing training and architecture parameters.
        """
        super().__init__(opt)
        
        # Trackable loss names (used in training logs)
        self.loss_names: List[str] = [
            'G_GAN', 'G_FM', 'G_OCR', 'G_Perceptual', 'G_L1',
            'D_real', 'D_fake'
        ]

        # Initialize all losses with zero tensors
        self.loss_G_GAN = torch.tensor(0.0, device=self.device)
        self.loss_G_FM = torch.tensor(0.0, device=self.device)
        self.loss_G_OCR = torch.tensor(0.0, device=self.device)
        self.loss_G_Perceptual = torch.tensor(0.0, device=self.device)
        self.loss_G_L1 = torch.tensor(0.0, device=self.device)
        self.loss_D_real = torch.tensor(0.0, device=self.device)
        self.loss_D_fake = torch.tensor(0.0, device=self.device)

        # Flags for optional losses
        self.use_OCR_loss = opt.lambda_ocr > 0.0
        self.use_FM_loss = opt.lambda_fm > 0.0
        self.use_Perceptual_loss = opt.lambda_perceptual > 0.0
        self.use_L1_loss = not opt.no_L1_loss

        # Model names (used when saving/loading)
        self.model_names: List[str] = ['G', 'D'] if self.isTrain else ['G']
        
        # -------------------------
        # Generator
        # -------------------------
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
        
        # -------------------------
        # Discriminator + Losses (only if training)
        # -------------------------
        if self.isTrain:
            self.netD: nn.Module = networks.define_D(
                input_nc=opt.input_nc + opt.output_nc,
                ndf=opt.ndf,
                n_layers_D=opt.n_layers_D,
                num_D=opt.num_D,
                init_type=opt.init_type,
                init_gain=opt.init_gain
            )
            
            # Loss functions
            self.criterionGAN = losses.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = nn.L1Loss()
            self.featOCR_criterion = nn.L1Loss()

            if opt.loss_Perceptual == 'contextual':
                self.criterionPerceptual = losses.ContextualLoss(
                    layers=['conv2_2', 'conv3_4'],
                    h=0.3,
                    pretrained_vgg=True,
                    device=self.device
                )
            elif(opt.loss_Perceptual == 'lpips'):
                self.criterionPerceptual = lpips.LPIPS(net='alex').to(self.device)

            # OCR wrapper
            if self.use_OCR_loss:
                modelOCR_path = Path(opt.checkpoints_dir) / opt.name / "persian_best.mlmodel"
                self.ocr = KrakenOCRWrapper(modelOCR_path, device=self.device)

            # Optimizers
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr_G, betas=(opt.beta1_G, opt.beta2_G)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr_D, betas=(opt.beta1_D, opt.beta2_D)
            )

            # Training hyperparameters
            self.scales = [0.5, 0.25]
            self.scale_weights = [1, 0.5]
            self.lambda_perceptual = opt.lambda_perceptual
            self.ctx_use_patches = opt.ctx_use_patches
            self.lambda_fm = opt.lambda_fm
            self.lambda_ocr = opt.lambda_ocr
            self.lambda_Kl_logits = opt.lambda_Kl_logits
            self.lambda_l1_feat = opt.lambda_l1_feat
            self.start_epoch_ocr = opt.start_epoch_ocr
            self.pretrain_epochs = opt.pretrain_epochs
            self.fm_warmup_epochs = opt.fm_warmup_epochs
            self.ocr_warmup_epochs = opt.ocr_warmup_epochs
            self.dataroot = opt.dataroot
            self.val_dir = opt.val_dir

        # Images to visualize during training/validation
        self.visual_names: List[str] = ['real_input', 'output_generator', 'real_output']

    # -------------------------
    # Input Handling
    # -------------------------
    def set_input(self, data: Dict[str, Any]) -> None:
        """
        Unpack and preprocess input data from dataloader.

        """
        self.real_input: torch.Tensor = data['input'].to(self.device)
        self.real_output: torch.Tensor = data['output'].to(self.device)
        self.feature_lstm: torch.Tensor = data['ocr']['feat'].to(self.device)
        self.logits: torch.Tensor = data['ocr']['logits'].to(self.device)
    
    # -------------------------
    # Forward / Validation
    # -------------------------
    def forward(self) -> None:
        """Forward pass through the generator."""
        self.output_generator: torch.Tensor = self.netG(self.real_input)

        if self.isTrain:
            # Concatenate input and output for discriminator training
            self.fake_AB = torch.cat((self.real_input, self.output_generator), dim=1)
            self.real_AB = torch.cat((self.real_input, self.real_output), dim=1)

    def validate(self, val_loader, epoch: int, max_batches: int = 10) -> None:
        """
        Run validation loop and save generated sample images.
        """

        self.eval()
        results_dir = Path(self.dataroot) / self.val_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():  
            for i, data in enumerate(val_loader):
                if i >= max_batches:
                    break

                self.set_input(data)
                self.forward()

                # Concatenate input / output / target for visualization
                grid = torch.cat([
                    self.real_input.cpu(),
                    self.output_generator.cpu(),
                    self.real_output.cpu()
                ], dim=2)

                save_path = results_dir / f"epoch{epoch:03d}_sample{i:02d}.png"
                vutils.save_image(grid, save_path, normalize=True)

        self.train()

    # -------------------------
    # Training Losses
    # -------------------------
    def backward_D(self) -> None:
        """
        Backward pass for discriminator.
        Uses DiffAugment to improve robustness.
        """
        fake_AB_aug = DiffAugment(self.fake_AB, policy='brightness,translation')
        real_AB_aug = DiffAugment(self.real_AB, policy='brightness,translation')
        
        pred_fake_multi = self.netD(fake_AB_aug.detach())
        pred_real_multi = self.netD(real_AB_aug)

        loss_D_fake, loss_D_real = 0, 0
        for pred_fake, pred_real in zip(pred_fake_multi, pred_real_multi):
            loss_D_fake += self.criterionGAN(pred_fake, False)
            loss_D_real += self.criterionGAN(pred_real, True)

        self.loss_D_fake = loss_D_fake / len(pred_fake_multi)
        self.loss_D_real = loss_D_real / len(pred_real_multi)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        
        self.loss_D.backward()

    def backward_G(self, epoch: int) -> None:
        """
        Backward pass for generator.
        Includes L1, perceptual, adversarial, FM, and OCR losses.
        """
        # Base L1 loss (multi-scale)
        if(self.use_L1_loss):
            self.loss_G_L1: torch.Tensor = self.multiscale_l1_loss(self.output_generator, self.real_output)

        # Perceptual loss
        if(self.use_Perceptual_loss):
            if(self.ctx_use_patches):
                self.loss_G_Perceptual = self.compute_ctx_on_patches(
                    self.output_generator, self.real_output, num_crops=3, crop_w=512
                ) * self.lambda_perceptual
            else:
                self.loss_G_Perceptual = (
                    self.criterionPerceptual(self.output_generator, self.real_output)
                    * self.lambda_perceptual
                )

        self.loss_G = self.loss_G_L1 + self.loss_G_Perceptual

        # Adversarial, FM, OCR losses (after pretrain phase)
        if epoch > self.pretrain_epochs:
            fake_AB_aug = DiffAugment(self.fake_AB, policy='brightness,translation')
            real_AB_aug = DiffAugment(self.real_AB, policy='brightness,translation')

            pred_fake_multi = self.netD(fake_AB_aug, return_features=True)
            pred_real_multi = self.netD(real_AB_aug, return_features=True)

            loss_G_GAN, loss_G_FM = 0, 0
            for (pred_fake, fake_feats), (_, real_feats) in zip(pred_fake_multi, pred_real_multi):
                loss_G_GAN += self.criterionGAN(pred_fake, True, is_generator=True)
                if self.use_FM_loss:
                    loss_G_FM += self.feature_matching_loss_from_feats(
                        fake_feats, [f.detach() for f in real_feats]
                    )

            # Scale FM loss during warmup
            fm_epoch = max(0, epoch - self.pretrain_epochs)
            lambda_FM_current = min(self.lambda_fm, self.lambda_fm * fm_epoch / self.fm_warmup_epochs)

            self.loss_G_GAN = loss_G_GAN / len(pred_fake_multi)
            self.loss_G_FM = (loss_G_FM / len(pred_fake_multi)) * lambda_FM_current

            # OCR loss (if enabled)
            if(self.use_OCR_loss and epoch >= self.start_epoch_ocr):
                ocr_epoch = max(0, epoch - self.start_epoch_ocr + 1)
                lambda_ocr_current = min(self.lambda_ocr, self.lambda_ocr * ocr_epoch / self.ocr_warmup_epochs)

                fake_feats_vec, fake_logits, _ = self.ocr.forward_features_logits(self.output_generator)
                real_feats_vec = self.feature_lstm.detach().to(fake_feats_vec.device)
                real_logits = self.logits.detach().to(fake_logits.device)

                # Feature alignment + KL divergence on logits
                loss_feat = self.featOCR_criterion(fake_feats_vec, real_feats_vec) * self.lambda_l1_feat
                fake_log_probs = F.log_softmax(fake_logits, dim=-1)
                real_probs = F.softmax(real_logits, dim=-1)
                kl_loss = F.kl_div(fake_log_probs, real_probs, reduction='batchmean') * self.lambda_Kl_logits

                self.loss_G_OCR = (loss_feat + kl_loss) * lambda_ocr_current
                self.loss_G += self.loss_G_OCR

            self.loss_G += self.loss_G_GAN + self.loss_G_FM

        self.loss_G.backward()

    # -------------------------
    # Optimizer Step
    # -------------------------
    def optimize_parameters(self, epoch: int) -> None:
        """
        Update network weights:
        - Pretrain phase: only generator is updated.
        - Later: discriminator first, then generator.
        """
        self.forward()
        
        # Update only generator
        if epoch <= self.pretrain_epochs:
            self.set_requires_grad(self.netD, False)
            self.optimizer_G.zero_grad()
            self.backward_G(epoch)
            self.optimizer_G.step()
        else:
            # Update discriminator
            seed = random.randint(0, 10000)
            torch.manual_seed(seed)
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            
            # Update generator
            self.set_requires_grad(self.netD, False)
            self.optimizer_G.zero_grad()
            self.backward_G(epoch)
            self.optimizer_G.step()

    # -------------------------
    # Helper Loss Functions
    # -------------------------
    def multiscale_l1_loss(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale L1 loss."""
        total_loss = 0.0
        for scale, weight in zip(self.scales, self.scale_weights):
            if scale != 1.0:
                fake_scaled = F.interpolate(fake, scale_factor=scale, mode='bilinear', align_corners=False)
                real_scaled = F.interpolate(real, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                fake_scaled, real_scaled = fake, real
            total_loss += weight * self.criterionL1(fake_scaled, real_scaled)
        return total_loss
    
    def feature_matching_loss_from_feats(
        self, fake_feats: List[torch.Tensor], real_feats: List[torch.Tensor]
        ) -> torch.Tensor:
        """Compute FM loss from precomputed feature maps."""
        fm_loss = torch.tensor(0.0, device=self.device)
        n_layers = max(1, len(fake_feats))
        for f_fake, f_real in zip(fake_feats, real_feats):
            fm_loss = fm_loss + torch.mean(torch.abs(f_fake - f_real))
        return fm_loss / float(n_layers)
    
    def compute_ctx_on_patches(self, fake, real, num_crops=3, crop_w=512):
        """Compute contextual loss on random patches for efficiency."""
        B,_,_,W = fake.shape
        total, count = 0.0, 0
        for b in range(B):
            for _ in range(num_crops):
                if W > crop_w:
                    sx = torch.randint(0, W - crop_w + 1, (1,)).item()
                    fake_patch = fake[b:b+1, :, :, sx:sx+crop_w]
                    real_patch = real[b:b+1, :, :, sx:sx+crop_w]
                else:
                    fake_patch, real_patch = fake[b:b+1], real[b:b+1]
                total += self.criterionPerceptual(fake_patch, real_patch)
                count += 1
        return total / count if count > 0 else 0.0
    
    def _get_ocr_targets_from_real(self):
        with torch.no_grad():
            texts = self.ocr.ocr_text(self.real_output)
        targets, target_lengths = self.ocr.encode_texts(texts)
        return targets, target_lengths
    