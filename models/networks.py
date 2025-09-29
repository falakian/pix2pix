import torch
import torch.nn as nn
import functools
from typing import Callable, Optional, List, Any, Dict

# Helper Functions
def get_norm_layer(norm_type: str = 'instance') -> Optional[Callable]:
    """
    Return a normalization layer based on the specified type.

    Args:
        norm_type: Type of normalization ('batch', 'instance', or 'none')

    Returns:
        Callable normalization layer or None if 'none' is specified

    Raises:
        ValueError: If norm_type is not supported
    """
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        return lambda x: x  # Identity function for no normalization
    else:
        raise ValueError(f"Normalization layer '{norm_type}' is not supported")

def init_weights(net: nn.Module, init_type: str = 'normal', init_gain: float = 0.02) -> None:
    """
    Initialize network weights using the specified initialization method.

    Args:
        net: PyTorch neural network module
        init_type: Initialization method ('normal', 'xavier', 'kaiming', 'orthogonal')
        init_gain: Scaling factor for initialization

    Raises:
        ValueError: If init_type is not supported
    """
    def init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
            if init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=init_gain)
            else:
                raise ValueError(f"Initialization method '{init_type}' is not supported")
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight, 1.0, init_gain)
            nn.init.constant_(m.bias, 0.0)

    net.apply(init_func)

def init_net(net: nn.Module, init_type: str = 'normal', init_gain: float = 0.02) -> nn.Module:
    """
    Initialize a network and move it to GPU if available.

    Args:
        net: PyTorch neural network module
        init_type: Initialization method
        init_gain: Scaling factor for initialization

    Returns:
        Initialized network
    """
    if torch.cuda.is_available():
        net = net.to("cuda")
    init_weights(net, init_type, init_gain)
    return net

def get_scheduler(optimizer: torch.optim.Optimizer, opt: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Return a learning rate scheduler based on the specified policy.

    Args:
        optimizer: PyTorch optimizer
        opt: Configuration dictionary containing:
            - lr_policy: Learning rate policy ('linear', 'step', 'plateau', 'cosine')
            - epoch_count: Starting epoch
            - n_epochs: Total number of epochs
            - n_epochs_decay: Number of epochs for decay
            - lr_decay_iters: Step size for step scheduler
            - n_epochs: Total number of epochs for cosine scheduler

    Returns:
        Learning rate scheduler

    Raises:
        ValueError: If lr_policy is not supported
    """
    lr_policy = opt.lr_policy
    if lr_policy == 'linear':
        def lambda_rule(epoch: int) -> float:
            return 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        raise ValueError(f"Learning rate policy '{lr_policy}' is not supported")


# U-Net Generator
class UnetSkipConnectionBlock(nn.Module):
    """U-Net block with skip connections for encoder-decoder architecture."""
    
    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: Optional[int] = None,
        submodule: Optional[nn.Module] = None,
        norm_layer: Callable = nn.InstanceNorm2d,
        innermost: bool = False,
        outermost: bool = False,
        use_dropout: bool = False,
        halve_height = True
    ):
        """
        Initialize a U-Net skip connection block.

        Args:
            outer_nc: Number of output channels
            inner_nc: Number of inner channels
            input_nc: Number of input channels (defaults to outer_nc)
            submodule: Nested U-Net block (if any)
            norm_layer: Normalization layer
            innermost: Whether this is the innermost block
            outermost: Whether this is the outermost block
            use_dropout: Whether to apply dropout
        """
        super().__init__()
        self.outermost = outermost
        kernel_size = (4,4)
        stride = (2,2) if(halve_height) else (1,2)
        padding = (1,1)
        input_nc = input_nc if input_nc is not None else outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = norm_layer(inner_nc) if norm_layer != (lambda x: x) else nn.Identity()
        uprelu = nn.ReLU(inplace=True)
        upnorm = norm_layer(outer_nc) if norm_layer != (lambda x: x) else nn.Identity()

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=kernel_size, stride=stride, padding=padding)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            self.model = nn.Sequential(*down, submodule, *up)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=kernel_size, stride=stride, padding=padding)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            self.model = nn.Sequential(*down, *up)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=kernel_size, stride=stride, padding=padding)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            up = up + [nn.Dropout(0.5)] if use_dropout else up
            self.model = nn.Sequential(*down, submodule, *up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net block.

        Args:
            x: Input tensor

        Returns:
            Output tensor (with skip connection if not outermost)
        """
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)

class UnetGenerator(nn.Module):
    """U-Net generator with skip connections for image-to-image translation."""
    
    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        num_downs: int,
        height_down_layers: int,
        ngf: int = 64,
        norm_layer: Callable = nn.BatchNorm2d,
        use_dropout: bool = False
    ):
        """
        Initialize the U-Net generator.

        Args:
            input_nc: Number of input channels
            output_nc: Number of output channels
            num_downs: Number of downsampling layers
            ngf: Number of filters in the first conv layer
            norm_layer: Normalization layer
            use_dropout: Whether to apply dropout
        """
        super().__init__()
        num_downs = max(num_downs, 5)
        height_down_layers = max(height_down_layers, 1)
        halve_height = True if num_downs == height_down_layers else False
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer, halve_height=halve_height)
        for i in range(num_downs-2, 0, -1):
            if (num_downs - height_down_layers) < 0:
                halve_height = True
            if(i > 3):
                unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, halve_height=halve_height)
            else:
                unet_block = UnetSkipConnectionBlock(ngf *(2**(i-1)) , ngf * (2**i), submodule=unet_block, norm_layer=norm_layer, halve_height=halve_height)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, halve_height=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net generator.

        Args:
            input: Input tensor

        Returns:
            Generated output tensor
        """
        return self.model(input)

# PatchGAN Discriminator
class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator for local feature discrimination."""
    
    def __init__(self, input_nc: int, ndf: int = 64, n_layers: int = 3):
        """
        Initialize the PatchGAN discriminator.

        Args:
            input_nc: Number of input channels
            ndf: Number of filters in the first conv layer
            n_layers: Number of conv layers
            norm_layer: Normalization layer
        """
        super().__init__()
        self.feature_layers = nn.ModuleList()

        sequence = [
            nn.utils.spectral_norm(
                nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.feature_layers.append(nn.Sequential(*sequence))

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers+1):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            block = [
                nn.utils.spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=4, stride=2 if n < n_layers else 1, padding=1, bias=True)
                ),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            self.feature_layers.append(nn.Sequential(*block))

        self.final_layer = nn.utils.spectral_norm(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1, bias=True)
        )

    def forward(self, input: torch.Tensor, return_features=False) -> torch.Tensor:
        """
        Forward pass through the PatchGAN discriminator.

        Args:
            input: Input tensor

        Returns:
            Discriminator output (patch map)
        """
        features = []
        out = input
        for layer in self.feature_layers:
            out = layer(out)
            features.append(out)
        out = self.final_layer(out)
        if return_features:
            return out, features
        return out

class MultiScaleDiscriminator(nn.Module):
    """Multi-scale PatchGAN discriminator."""

    def __init__(self, input_nc: int, ndf: int = 64, n_layers: int = 3, num_D: int = 3):
        """
        Initialize the multi-scale discriminator.

        Args:
            input_nc: Number of input channels
            ndf: Number of filters in first conv layer of each discriminator
            n_layers: Number of conv layers in each discriminator
            num_D: Number of discriminators at different scales
        """
        super().__init__()
        self.num_D = num_D
        self.discriminators = nn.ModuleList([
            NLayerDiscriminator(input_nc, ndf, n_layers) for _ in range(num_D)
        ])
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> List:
        """
        Run input through multiple discriminators at different scales.

        Args:
            x: Input tensor
            return_features: Whether to return intermediate features for feature matching

        Returns:
            List of outputs (and optionally features) from each scale
        """
        results = []
        input_downsampled = x
        for i in range(self.num_D):
            out = self.discriminators[i](input_downsampled, return_features=return_features)
            results.append(out)
            if i != self.num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        return results

# Factory Functions
def define_G(
    input_nc: int,
    output_nc: int,
    ngf: int,
    num_downs: int,
    height_down_layers: int,
    # netG: str = 'unet_256',
    norm: str = 'instance',
    use_dropout: bool = False,
    init_type: str = 'normal',
    init_gain: float = 0.02
) -> nn.Module:
    """
    Create and initialize a generator network.

    Args:
        input_nc: Number of input channels
        output_nc: Number of output channels
        ngf: Number of filters in the first conv layer
        netG: Generator architecture ('unet_256')
        norm: Normalization type
        use_dropout: Whether to apply dropout
        init_type: Initialization method
        init_gain: Initialization gain

    Returns:
        Initialized generator network
    """
    norm_layer = get_norm_layer(norm)
    net = UnetGenerator(input_nc, output_nc, num_downs=num_downs,height_down_layers=height_down_layers, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    return init_net(net, init_type, init_gain)

def define_D(
    input_nc: int,
    ndf: int,
    n_layers_D: int = 3,
    num_D: int = 3,
    init_type: str = 'normal',
    init_gain: float = 0.02
) -> nn.Module:
    """Create and initialize a multi-scale discriminator."""
    net = MultiScaleDiscriminator(input_nc, ndf, n_layers_D, num_D)
    return init_net(net, init_type, init_gain)