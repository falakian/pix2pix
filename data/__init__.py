import torch
import copy
from torch.utils.data import DataLoader
from data.aligned_dataset import AlignedDataset

def create_dataset(opt, verbose: bool = True , is_validation: bool = False) -> DataLoader:
    """
    Create a DataLoader for the specified dataset.

    Args:
        verbose (bool, optional): If True, logs dataset creation details. Defaults to True.

    Returns:
        DataLoader: A PyTorch DataLoader configured with the specified dataset and parameters.

    """
    opt_database = copy.deepcopy(opt)
    if is_validation:
        opt_database.preprocess='none'
        opt_database.no_flip = True
        opt_database.max_dataset_size = opt.max_batches_val
        opt_database.serial_batches = True
        opt_database.input_dir = opt.input_dir_val
        opt_database.output_dir = opt.output_dir_val

    # Initialize the dataset
    dataset = AlignedDataset(opt_database)
    if verbose:
        print(f"Dataset [{type(dataset).__name__}] created with {len(dataset)} images.")

    # Configure the DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=opt_database.batch_size,
        shuffle=not opt_database.serial_batches,
        num_workers=int(opt_database.num_threads),
        pin_memory=torch.cuda.is_available(),  # Optimize for GPU if available
    )

    return dataloader