from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, Union, Any

import torch
import torch.nn as nn

from . import networks

class BaseModel(nn.Module):
    """
    Base class for all models.
    Provides shared functionality for training, inference, saving, and loading.
    """

    def __init__(self, opt: Dict[str, any]):
        """
        Initialize the base model with configuration options.

        Args:
            opt: Dictionary containing model configuration options
        """
        super().__init__()
        self.opt = opt
        self.isTrain: bool = opt.isTrain
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir: Path = Path(opt.checkpoints_dir) / opt.name
        
        # Enable CuDNN optimization for fixed-size inputs
        torch.backends.cudnn.benchmark = True
        
        # Tracking attributes
        self.loss_names: List[str] = []
        self.model_names: List[str] = []
        self.visual_names: List[str] = []
        self.optimizer_G: torch.optim.Optimizer
        self.optimizer_D: torch.optim.Optimizer
        self.image_paths: List[str] = []
        self.metric: float = 0.0

        # Ensure save directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Abstract methods (must be implemented by subclasses)
    # ============================================================
    def forward(self, *args, **kwargs) -> None:
        """Forward pass (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement forward method")

    def set_input(self, input: Dict[str, torch.Tensor]) -> None:
        """Set input data (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement set_input method")

    def optimize_parameters(self) -> None:
        """Optimize parameters (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement optimize_parameters method")

    # ============================================================
    # Setup & Training Utilities
    # ============================================================

    def setup(self, opt: Dict[str, any]) -> None:
        """
        Set up schedulers, load checkpoints if needed.
        """
        if self.isTrain:
            self.scheduler_G = networks.get_scheduler(self.optimizer_G, opt)
            self.scheduler_D = networks.get_scheduler(self.optimizer_D, opt)

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        
        self.print_networks(opt.verbose)

    def eval(self) -> None:
        """Switch all networks to evaluation mode."""
        for name in self.model_names:
            net = getattr(self, f"net{name}")
            net.eval()

    def test(self) -> None:
        """Forward pass in test mode (no gradients)."""
        with torch.no_grad():
            self.forward()

    def update_learning_rate(self, epoch) -> None:
        """Update learning rates for schedulers."""
        if (not self.optimizer_G) or (not self.optimizer_D):
            return
        
        old_lr_G = self.optimizer_G.param_groups[0]['lr']
        old_lr_D = self.optimizer_D.param_groups[0]['lr']
        
        self.scheduler_G.step()
        if epoch > self.pretrain_epochs:
            self.scheduler_D.step()

        new_lr_G = self.optimizer_G.param_groups[0]['lr']
        new_lr_D = self.optimizer_D.param_groups[0]['lr']
        if(old_lr_G !=new_lr_G):
            print(f"Learning rate generator updated: {old_lr_G:.7f} -> {new_lr_G:.7f}")
        if(old_lr_D !=new_lr_D):
            print(f"Learning rate discriminator updated: {old_lr_D:.7f} -> {new_lr_D:.7f}")

    # ============================================================
    # Losses & Visuals
    # ============================================================

    def get_current_losses(self):
        """Return current training losses."""
        losses = OrderedDict()
        for name in self.loss_names:
            attr_name = f"loss_{name}"
            val = getattr(self, attr_name, 0.0)

            if isinstance(val, torch.Tensor):
                losses[name] = float(val.detach().cpu())
            else:
                losses[name] = float(val)
        return losses

    def get_current_visuals(self) -> OrderedDict:
        """Return current visualization tensors."""
        return OrderedDict(
            (name, getattr(self, name)) for name in self.visual_names
        )

    # ============================================================
    # Checkpointing
    # ============================================================

    def save_networks(self, epoch: Union[str, int]) -> None:
        """
        Save all networks to disk.

        Args:
            epoch: Epoch number or checkpoint identifier.
        """
        for name in self.model_names:
            save_path = self.save_dir / f"{epoch}_net_{name}.pth"
            net = getattr(self, f"net{name}")
            if isinstance(net, nn.DataParallel):
                net = net.module
            torch.save(net.cpu().state_dict(), save_path)
            if torch.cuda.is_available():
                net.to(self.device)

    def load_networks(self, epoch: Union[str, int]) -> None:
        """
        Load networks from disk.

        Args:
            epoch: Epoch number or checkpoint identifier.
        """
        for name in self.model_names:
            load_path = self.save_dir / f"{epoch}_net_{name}.pth"
            print(f"Loading {name} from {load_path}")
            net = getattr(self, f"net{name}")
            if isinstance(net, nn.DataParallel):
                net = net.module
            state_dict = torch.load(load_path, map_location=self.device, weights_only=True)
            net.load_state_dict(state_dict)

    def print_networks(self, verbose: bool = False) -> None:
        """
        Print summary of networks (params & structure).
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            net = getattr(self, f"net{name}")
            num_params = sum(p.numel() for p in net.parameters())
            print(f"[{name}] Parameters: {num_params/1e6:.3f}M")
            if verbose:
                print(net)
        print("---------------------------------------------")

    # ============================================================
    # Utility
    # ============================================================
    def set_requires_grad(self, nets: Union[nn.Module, List[nn.Module]], requires_grad: bool = False) -> None:
        """
        Enable/disable gradient computation for given networks.

        Args:
            nets: Single network or list of networks
            requires_grad: Whether gradients are required
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad