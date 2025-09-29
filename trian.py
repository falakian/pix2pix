import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import time
from typing import Dict, Any
from options.train_options import TrainOptions
from data import create_dataset
from models.pix2pix_model import Pix2PixModel

def main() -> None:
    """Main training loop for the pix2pix model."""
    # Parse training options
    opt: TrainOptions = TrainOptions().parse()

    # Initialize dataset
    dataset = create_dataset(opt)
    dataset_size: int = len(dataset)
    print(f"Number of training images: {dataset_size}")

    # Initialize validation dataset
    if(opt.val_count !=-1):
        dataset_val = create_dataset(opt , is_validation=True)

    # Initialize model
    model: Pix2PixModel = Pix2PixModel(opt)
    model.setup(opt)

    # Track total iterations across epochs
    total_iters: int = 0

    # Training loop over epochs
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time: float = time.time()
        epoch_iter = 0
        losses_means: Dict[str, float] = {}
        # Iterate over dataset
        for data in dataset:
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            # Forward pass and optimization
            model.set_input(data)
            model.optimize_parameters(epoch)

            losses: Dict[str, float] = model.get_current_losses()
            for k, v in losses.items():
                losses_means[k] = losses_means.get(k, 0.0) + v

            # Print losses at specified frequency
            if total_iters % opt.print_freq == 0:
                message = f"(epoch: {epoch}, iters: {epoch_iter}) "
                for k, v in losses.items():
                    message += f", {k}: {v:.3f}"
                print(message)

        # Update learning rate after each epoch
        model.update_learning_rate(epoch)

        # Save model checkpoints at specified epochs
        if epoch % opt.save_epoch == 0:
            print(f"Saving model at epoch {epoch}")
            model.save_networks("latest")
            #model.save_networks(epoch)

        # Print epoch completion time
        epoch_duration: int = int(time.time() - epoch_start_time)
        print(f"End of epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay} - "
              f"Time: {epoch_duration} sec")
        
        if opt.val_count !=-1 and epoch % opt.val_count == 0:
            model.validate(dataset_val, epoch)

        message = ''
        for k, v in losses_means.items():
            avg_loss = v / epoch_iter if epoch_iter > 0 else 0.0
            message += f", avg {k}: {avg_loss:.3f}"
        print(message)
        
if __name__ == "__main__":
    main()