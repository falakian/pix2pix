from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class BaseDataset(Dataset, ABC):
    """Abstract base class for datasets, providing common functionality and interface."""

    def __init__(self, opt) -> None:
        """
        Initialize the base dataset with configuration options.

        Raises:
            ValueError: If dataroot is not provided or invalid.
        """
        super().__init__()
        if not opt.dataroot:
            raise ValueError("dataroot must be specified in configuration")
        
        self.opt = opt
        self.root: Path = Path(opt.dataroot)

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieve a sample from the dataset at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Dict[str, Any]: Dictionary containing sample data and metadata.
        """
        pass

    @staticmethod
    def crop(img: Image.Image, pos: Tuple[int, int], size: Tuple[int, int]) -> Image.Image:
        """
        Crop an image at the specified position with the given size.

        Args:
            img (Image.Image): Input PIL image.
            pos (Tuple[int, int]): Top-left corner (x, y) of the crop.
            size (Tuple[int, int]): Size of the crop (width, height).

        Returns:
            Image.Image: Cropped image.

        Raises:
            ValueError: If crop dimensions are invalid or exceed image boundaries.
        """
        width, height = img.size
        x, y = pos
        crop_width, crop_height = size

        if width < crop_width or height < crop_height:
            raise ValueError(
                f"Crop size ({crop_width}, {crop_height}) exceeds image dimensions ({width}, {height})"
            )
        if x < 0 or y < 0 or x + crop_width > width or y + crop_height > height:
            raise ValueError(
                f"Invalid crop position ({x}, {y}) for image size ({width}, {height})"
            )

        return img.crop((x, y, x + crop_width, y + crop_height))

    @staticmethod
    def flip(img: Image.Image) -> Image.Image:
        """
        Horizontally flip an image.

        Args:
            img (Image.Image): Input PIL image.

        Returns:
            Image.Image: Flipped image.
        """
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

def get_params(opt, size: Tuple[int, int]) -> Dict[str, Union[Tuple[int, int], bool]]:
    """
    Generate transformation parameters for image preprocessing.

    Args:
        opt (Config): Configuration object with preprocessing options.
        size (Tuple[int, int]): Image size (width, height).

    Returns:
        Dict[str, Union[Tuple[int, int], bool]]: Dictionary containing:
            - 'crop_pos': (x, y) coordinates for cropping.
            - 'flip': Boolean indicating whether to flip the image.

    Raises:
        ValueError: If preprocessing options are invalid.
    """
    width, height = size
    crop_pos = (width // 2, height // 2)  # Default to center crop
        
    if opt.preprocess != 'none':
        new_width, new_height = opt.load_size_width, opt.load_size_height
        crop_width, crop_height = opt.crop_size_width, opt.crop_size_height
        if new_width < crop_width or new_height < crop_height:
            raise ValueError(
                f"load_size ({new_width}, {new_height}) must be >= crop_size ({crop_width}, {crop_height})"
            )
        crop_pos = (
            random.randint(0, max(0, new_width - crop_width)),
            random.randint(0, max(0, new_height - crop_height))
        )

    flip = random.random() > 0.5 if not opt.no_flip else False

    return {'crop_pos': crop_pos, 'flip': flip}

def get_transform(
    opt,
    params: Dict[str, Union[Tuple[int, int], bool]],
    grayscale: bool = False
) -> transforms.Compose:
    """
    Create a transformation pipeline for image preprocessing.

    Args:
        opt (Config): Configuration object with preprocessing options.
        params ([Dict]): Transformation parameters (crop_pos, flip).
        grayscale (bool): Whether to convert the image to grayscale.

    Returns:
        transforms.Compose: Composed transformation pipeline.

    Raises:
        ValueError: If preprocessing options are invalid.
    """
    transform_list = []

    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    if 'resize' in opt.preprocess:
        transform_list.append(
            transforms.Resize(
                size=(opt.load_size_height, opt.load_size_width),
                interpolation=transforms.InterpolationMode.BICUBIC
            )
        )

    if 'crop' in opt.preprocess:
        transform_list.append(
            transforms.Lambda(
                lambda img: BaseDataset.crop(img, params['crop_pos'], (opt.crop_size_width, opt.crop_size_height))
            )
        )

    if not opt.no_flip:
        transform_list.append(
            transforms.Lambda(lambda img: BaseDataset.flip(img))
        )

    if 'augment' in opt.preprocess:
        transform_list.extend([
        transforms.RandomAffine(
            degrees=3,
            scale=(0.95, 1.05),
            fill=(255,)
        ),
        ])

    transform_list.append(transforms.ToTensor())
    if grayscale:
        transform_list.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))
    else:
        transform_list.append(
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        )

    return transforms.Compose(transform_list)