import os
import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models.pix2pix_model import Pix2PixModel

if __name__ == "__main__":
    opt = TrainOptions().parse()

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"Number of training images: {dataset_size}")

    model = Pix2PixModel(opt)
    model.setup(opt)

    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            total_iters += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                print(f"[Epoch {epoch}] [Iter {total_iters}] Losses: {losses}")

            if total_iters % opt.save_latest_freq == 0:
                print(f"Saving model at iter {total_iters}")
                suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(suffix)

        model.update_learning_rate()

        if epoch % opt.save_epoch_freq == 0:
            print(f"Saving model at epoch {epoch}")
            model.save_networks("latest")
            model.save_networks(epoch)

        print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} - Time: {int(time.time() - epoch_start_time)} sec")
