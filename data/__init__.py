import torch
from torch.utils.data import DataLoader
from data.aligned_dataset import AlignedDataset

def create_dataset(opt, verbose: bool = True) -> DataLoader:
    """
    Create a DataLoader for the specified dataset.

    Args:
        verbose (bool, optional): If True, logs dataset creation details. Defaults to True.

    Returns:
        DataLoader: A PyTorch DataLoader configured with the specified dataset and parameters.

    """
    # Initialize the dataset
    dataset = AlignedDataset(opt)
    if verbose:
        print(f"Dataset [{type(dataset).__name__}] created with {len(dataset)} images.")

    # Configure the DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads),
        pin_memory=torch.cuda.is_available(),  # Optimize for GPU if available
    )

    return dataloader