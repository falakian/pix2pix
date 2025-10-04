import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import lpips
from pathlib import Path
from typing import Dict, List, Any
from .base_model import BaseModel
from .ocr_kraken import KrakenOCRWrapper
from . import networks
from . import losses
import random
from utile.DiffAugment_pytorch import DiffAugment

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
        self.loss_names: List[str] = ['G_GAN', 'G_FM', 'G_OCR', 'G_Perceptual', 'G_L1', 'D_real', 'D_fake']
        self.loss_G_GAN = torch.tensor(0.0, device=self.device)
        self.loss_G_FM = torch.tensor(0.0, device=self.device)
        self.loss_G_OCR = torch.tensor(0.0, device=self.device)
        self.loss_G_Perceptual = torch.tensor(0.0, device=self.device)
        self.loss_G_L1 = torch.tensor(0.0, device=self.device)
        self.loss_D_real = torch.tensor(0.0, device=self.device)
        self.loss_D_fake = torch.tensor(0.0, device=self.device)

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
                num_D=opt.num_D,
                init_type=opt.init_type,
                init_gain=opt.init_gain
            )
            
            # Define loss functions
            self.criterionGAN = losses.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = nn.L1Loss()
            self.featOCR_criterion = nn.L1Loss()
            if(opt.loss_Perceptual == 'contextual'):
                self.criterionPerceptual = losses.ContextualLoss(layers=['conv2_2', 'conv3_4'], h=0.3, pretrained_vgg=True, device=self.device)
            elif(opt.loss_Perceptual == 'lpips'):
                self.criterionPerceptual = lpips.LPIPS(net='alex').to(self.device)

            modelOCR_path = Path(opt.checkpoints_dir) / opt.name / "persian_best.mlmodel"
            self.use_ocr_loss = opt.lambda_ocr > 0.0
            if self.use_ocr_loss:
                self.ocr = KrakenOCRWrapper(modelOCR_path, device=self.device)
        
            # Initialize optimizers
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr_G,
                betas=(opt.beta1_G, opt.beta2_G)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt.lr_D,
                betas=(opt.beta1_D, opt.beta2_D)
            )

            
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
        self.feature_lstm: torch.Tensor = data['ocr']['feat'].to(self.device)
        self.logits: torch.Tensor = data['ocr']['logits'].to(self.device)
    
    def validate(self, val_loader, epoch: int, max_batches: int = 10) -> None:
        """
    Run validation loop and save sample outputs.
    
    Args:
        val_loader: iterable (e.g., dataset or dataloader) for validation data
        epoch: current epoch number
        max_batches: number of validation batches to process (default=10)
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

                input_img = self.real_input.detach().cpu()
                output_img = self.output_generator.detach().cpu()
                target_img = self.real_output.detach().cpu()

                grid = torch.cat([input_img, output_img, target_img], dim=2)

                save_path = results_dir / f"epoch{epoch:03d}_sample{i:02d}.png"
                vutils.save_image(grid, save_path, normalize=True)

        self.train()


    def forward(self) -> None:
        """Run the generator forward pass to produce output images."""
        self.output_generator: torch.Tensor = self.netG(self.real_input)

        if self.isTrain:
            self.fake_AB = torch.cat((self.real_input, self.output_generator), dim=1)
            self.real_AB = torch.cat((self.real_input, self.real_output), dim=1)

    def backward_D(self) -> None:
        fake_AB_aug = DiffAugment(self.fake_AB, policy='brightness,translation')
        real_AB_aug = DiffAugment(self.real_AB, policy='brightness,translation')
        
        pred_fake_multi = self.netD(fake_AB_aug.detach())
        pred_real_multi = self.netD(real_AB_aug)

        loss_D_fake = 0
        loss_D_real = 0
        for pred_fake, pred_real in zip(pred_fake_multi, pred_real_multi):
            loss_D_fake += self.criterionGAN(pred_fake, False)
            loss_D_real += self.criterionGAN(pred_real, True)

        self.loss_D_fake = loss_D_fake / len(pred_fake_multi)
        self.loss_D_real = loss_D_real / len(pred_real_multi)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        
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
    
    def feature_matching_loss_from_feats(self, fake_feats: List[torch.Tensor], real_feats: List[torch.Tensor]) -> torch.Tensor:
        """Compute FM loss from precomputed feature maps (avoid repeated netD calls)."""
        fm_loss = torch.tensor(0.0, device=self.device)
        n_layers = max(1, len(fake_feats))
        for f_fake, f_real in zip(fake_feats, real_feats):
            fm_loss = fm_loss + torch.mean(torch.abs(f_fake - f_real))
        return fm_loss / float(n_layers)
    
    def compute_ctx_on_patches(self, fake, real, num_crops=3, crop_w=512):
        B,C,H,W = fake.shape
        total = 0.0
        count = 0
        for b in range(B):
            for i in range(num_crops):
                if W > crop_w:
                    sx = torch.randint(0, W - crop_w + 1, (1,)).item()
                    fake_patch = fake[b:b+1, :, :, sx:sx+crop_w]
                    real_patch = real[b:b+1, :, :, sx:sx+crop_w]
                else:
                    fake_patch = fake[b:b+1]
                    real_patch = real[b:b+1]
                total += self.criterionPerceptual(fake_patch, real_patch)
                count += 1
        return total / count
    
    def _get_ocr_targets_from_real(self):
        with torch.no_grad():
            texts = self.ocr.ocr_text(self.real_output)
        targets, target_lengths = self.ocr.encode_texts(texts)
        return targets, target_lengths
    
    def backward_G(self, epoch: int) -> None:
        """Calculate and backpropagate generator losses """
        # Multi-scale L1
        self.loss_G_L1: torch.Tensor = self.multiscale_l1_loss(self.output_generator, self.real_output)

        if(self.ctx_use_patches):
            self.loss_G_Perceptual = self.compute_ctx_on_patches(self.output_generator, self.real_output, num_crops=3, crop_w=512) * self.lambda_perceptual
        else:
            self.loss_G_Perceptual: torch.Tensor = self.criterionPerceptual(self.output_generator, self.real_output) * self.lambda_perceptual

        self.loss_G: torch.Tensor = self.loss_G_L1 + self.loss_G_Perceptual

        if epoch > self.pretrain_epochs:
            fake_AB_aug = DiffAugment(self.fake_AB, policy='brightness,translation')
            real_AB_aug = DiffAugment(self.real_AB, policy='brightness,translation')

            pred_fake_multi = self.netD(fake_AB_aug, return_features=True)
            pred_real_multi = self.netD(real_AB_aug, return_features=True)

            loss_G_GAN = 0
            loss_G_FM = 0

            for (pred_fake, fake_feats), (_, real_feats) in zip(pred_fake_multi, pred_real_multi):
                loss_G_GAN += self.criterionGAN(pred_fake, True, is_generator=True)
                loss_G_FM += self.feature_matching_loss_from_feats(fake_feats, [f.detach() for f in real_feats])

            fm_epoch = max(0, epoch - self.pretrain_epochs)
            lambda_FM_current = min(self.lambda_fm, self.lambda_fm * fm_epoch / self.fm_warmup_epochs)

            self.loss_G_GAN = loss_G_GAN / len(pred_fake_multi)
            self.loss_G_FM = (loss_G_FM / len(pred_fake_multi)) * lambda_FM_current

            if(self.use_ocr_loss and epoch >= self.start_epoch_ocr):
                ocr_epoch = max(0, epoch - self.start_epoch_ocr + 1)
                lambda_ocr_current = min(self.lambda_ocr, self.lambda_ocr * ocr_epoch / self.ocr_warmup_epochs)
                fake_feats_vec, fake_logits, fake_logit_lengths = self.ocr.forward_features_logits(self.output_generator)

                real_feats_vec = self.feature_lstm.detach().to(fake_feats_vec.device)
                real_logits = self.logits.detach().to(fake_logits.device)

                loss_feat = self.featOCR_criterion(fake_feats_vec, real_feats_vec) * self.lambda_l1_feat

                fake_log_probs = F.log_softmax(fake_logits, dim=-1)
                real_probs = F.softmax(real_logits, dim=-1)

                kl_loss = F.kl_div(fake_log_probs, real_probs, reduction='batchmean') * self.lambda_Kl_logits

                self.loss_G_OCR = (loss_feat + kl_loss) * lambda_ocr_current

                self.loss_G += self.loss_G_OCR

            self.loss_G += self.loss_G_GAN + self.loss_G_FM

        self.loss_G.backward()

    def optimize_parameters(self, epoch: int) -> None:
        """Update network weights: optimize discriminator first, then generator."""
        # Forward pass
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