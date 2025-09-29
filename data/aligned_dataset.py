from pathlib import Path
from typing import Dict, List, Tuple, Union
from PIL import Image
import torch
import random
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

class AlignedDataset(BaseDataset):
    '''A dataset class for aligned image pairs (e.g., input-output pairs like facades).'''
    def __init__(self, opt) -> None:
        super().__init__(opt)
        
        # Define directory paths using pathlib for robust path handling
        self.dir_input: Path = Path(opt.dataroot) / opt.input_dir
        self.dir_output: Path = Path(opt.dataroot) / opt.output_dir
        
        # Verify directories exist
        if not self.dir_input.exists():
            raise FileNotFoundError(f"Input directory not found: {self.dir_input}")
        if not self.dir_output.exists():
            raise FileNotFoundError(f"Output directory not found: {self.dir_output}")
        
        # Load and sort image paths
        self.input_paths: List[Path] = sorted(
            make_dataset(str(self.dir_input), self.opt.max_dataset_size)
        )
        self.output_paths: List[Path] = sorted(
            make_dataset(str(self.dir_output), self.opt.max_dataset_size)
        )
        
        # Validate dataset consistency
        if len(self.input_paths) != len(self.output_paths):
            raise ValueError(
                f"Mismatched dataset sizes: {len(self.input_paths)} input images, "
                f"{len(self.output_paths)} output images"
            )
        
        # Store channel configurations
        self.input_nc: int = opt.input_nc
        self.output_nc: int = opt.output_nc
        

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Retrieve an aligned input-output image pair at the specified index.

        Args:
            index (int): Index of the image pair to retrieve.

        Returns:
            Dict[str, Union[torch.Tensor, str]]: Dictionary containing:
                - 'input': Transformed input image tensor
                - 'output': Transformed output image tensor

        """
        input_path = self.input_paths[index]
        output_path = self.output_paths[index]
        
        # Open and convert images to RGB
        input_img = Image.open(input_path).convert('RGB')
        output_img = Image.open(output_path).convert('RGB')
        
        # Get transformation parameters
        params = get_params(self.opt, input_img.size)
        
        # Apply transformations
        seed = random.randint(0, 10000)
        torch.manual_seed(seed)
        input_transform = get_transform(self.opt, params, grayscale=(self.input_nc == 1))
        output_transform = get_transform(self.opt, params, grayscale=(self.output_nc == 1))
        
        input_tensor = input_transform(input_img)
        output_tensor = output_transform(output_img)

        return {
            'input': input_tensor,
            'output': output_tensor,
            'name': str(Path(input_path).stem)
        }
        

    def __len__(self) -> int:
        """
        Get the total number of image pairs in the dataset.

        Returns:
            int: Minimum of input/output dataset sizes and max_dataset_size.
        """
        return min(len(self.input_paths), self.opt.max_dataset_size)
    