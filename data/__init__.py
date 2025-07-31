from data.aligned_dataset import AlignedDataset
import torch.utils.data

def create_dataset(opt):
    dataset = AlignedDataset(opt)
    print(f"Dataset [{type(dataset).__name__}] created with {len(dataset)} images.")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads)
    )
    return dataloader
