import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, List, Any, Dict
from torchvision import models

# GAN Loss
class GANLoss(nn.Module):
    """Implements GAN loss functions (vanilla GAN, LSGAN, or Hinge)."""
    
    def __init__(self, gan_mode: str = 'vanilla', target_real_label: float = 1.0, target_fake_label: float = 0.0):
        """
        Initialize GAN loss function.

        Args:
            gan_mode: GAN loss type ('vanilla' for BCE, 'lsgan' for MSE, 'hinge' for Hinge loss)
            target_real_label: Label value for real samples (used for vanilla and lsgan)
            target_fake_label: Label value for fake samples (used for vanilla and lsgan)

        Raises:
            ValueError: If gan_mode is not supported
        """
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.loss = None
        else:
            raise ValueError(f"GAN mode '{gan_mode}' is not supported")

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """
        Create target tensor for loss computation (used for vanilla and lsgan).

        Args:
            prediction: Discriminator prediction tensor
            target_is_real: Whether the target is real (True) or fake (False)

        Returns:
            Target tensor with appropriate labels
        """
        if self.gan_mode in ['vanilla', 'lsgan']:
            target_label = self.real_label if target_is_real else self.fake_label
            return target_label.expand_as(prediction)
        return None  # Not used for hinge loss

    def forward(self, prediction, target_is_real, is_generator=False):
        if self.gan_mode == 'hinge':
            if is_generator:
                return -prediction.mean()
            else:
                if target_is_real:
                    return torch.relu(1.0 - prediction).mean()
                else:
                    return torch.relu(1.0 + prediction).mean()
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)
        
class VGGFeatureExtractor(nn.Module):
    """
    Extract intermediate feature maps from VGG19 at specified layer names.
    Layer names supported: 'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
    'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
    'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
    'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'
    """
    def __init__(self, layer_names: List[str] = ['conv2_2', 'conv3_4'], pretrained=True, device='cuda'):
        super().__init__()
        vgg = models.vgg19(pretrained=pretrained).features
        self.device = device
        self.vgg = vgg.to(device)
        self.vgg.eval()
        # freeze params
        for p in self.vgg.parameters():
            p.requires_grad = False

        # mapping conv layer name -> index in vgg.features
        self.layer_name_to_idx = {
            'conv1_1': 0, 'conv1_2': 2,
            'conv2_1': 5, 'conv2_2': 7,
            'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_4': 16,
            'conv4_1': 19, 'conv4_2': 21, 'conv4_3': 23, 'conv4_4': 25,
            'conv5_1': 28, 'conv5_2': 30, 'conv5_3': 32, 'conv5_4': 34
        }

        # choose indices to capture
        self.selected_indices = [self.layer_name_to_idx[name] for name in layer_names]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            if idx in self.selected_indices:
                outs.append(x)
        return outs


# Contextual Loss implementation (based on Mechrez et al.)

class ContextualLoss(nn.Module):
    def __init__(self, layers: List[str] = ['conv2_2', 'conv3_4'],
                 h: float = 0.3, pretrained_vgg=True, device='cuda', resize: float = 0.5):
        super().__init__()
        self.vgg_extractor = VGGFeatureExtractor(layer_names=layers, pretrained=pretrained_vgg, device=device)
        self.h = h
        self.eps = 1e-5
        self.device = device
        self.resize = resize

    def preprocess_for_vgg(self, x: torch.Tensor) -> torch.Tensor:
        # x in [-1,1], grayscale or multi-channel
        x = (x + 1.0) / 2.0  # -> [0,1]
        if x.shape[1] == 1:  # grayscale -> replicate to 3 channels
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3, :, :]
        if self.resize is not None and self.resize != 1.0:
            x = F.interpolate(x, scale_factor=self.resize, mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        return (x - mean) / std

    def _cx_single(self, fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
        # fx,fy: (C, H, W)
        C, H, W = fx.shape
        fx = fx.view(C, -1)   # (C, N)
        fy = fy.view(C, -1)   # (C, M) typically N==M
        # L2 normalize across channels
        fx_norm = fx / (fx.norm(dim=0, keepdim=True) + self.eps)
        fy_norm = fy / (fy.norm(dim=0, keepdim=True) + self.eps)
        # cosine sim -> distance
        sim = torch.matmul(fx_norm.t(), fy_norm)  # (N, M)
        d = 1.0 - sim
        d_min, _ = torch.min(d, dim=1, keepdim=True)  # (N,1)
        d_rel = d / (d_min + self.eps)
        w = torch.exp((1.0 - d_rel) / self.h)
        w_norm = w / (torch.sum(w, dim=1, keepdim=True) + self.eps)
        max_w, _ = torch.max(w_norm, dim=1)
        cx = torch.mean(max_w)
        loss = -torch.log(cx + self.eps)
        return loss

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x,y in [-1,1], shapes (B,C,H,W)
        x_p = self.preprocess_for_vgg(x)
        y_p = self.preprocess_for_vgg(y)
        feats_x = self.vgg_extractor(x_p)
        feats_y = self.vgg_extractor(y_p)
        total_loss = 0.0
        n_layers = len(feats_x)
        B = x.shape[0]
        for fx_layer, fy_layer in zip(feats_x, feats_y):
            layer_loss = 0.0
            for b in range(B):
                layer_loss = layer_loss + self._cx_single(fx_layer[b], fy_layer[b])
            total_loss = total_loss + (layer_loss / float(B))
        return total_loss / float(n_layers)
    
    