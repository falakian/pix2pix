# Pix2Pix Persian Handwritten → Typed

**Short description**
This repository provides a modular, argument-driven implementation inspired by the original **Pix2Pix** model for image-to-image translation. The Pix2Pix model, introduced in the paper *“Image-to-Image Translation with Conditional Adversarial Networks”* by Isola et al. (2017), established a general framework for translating images from one domain to another using conditional GANs. For the original implementation and paper, see the official GitHub repository: [phillipi/pix2pix](https://github.com/phillipi/pix2pix).

**Pix2pix: [Project](https://phillipi.github.io/pix2pix/) | [Paper](https://arxiv.org/pdf/1611.07004.pdf) | [Torch](https://github.com/phillipi/pix2pix) |
[Tensorflow Core Tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix) | [PyTorch Colab](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb)**

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.png" width="800px"/>

The present project reimplements and extends Pix2Pix in PyTorch, with inspiration and utilities drawn from the excellent [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository. This version emphasizes modularity, configurability, and experimental flexibility.

---

## Features

* Modular codebase with separate directories for data, models, options, and utilities.
* Extensive command-line argument configuration for flexible experimentation.
* Supports multiple normalization and initialization schemes.
* Integrates perceptual, feature-matching, and OCR-based auxiliary losses.
* Includes data augmentation utilities (DiffAugment) for improved generalization.

---

## Project structure

```
│   README.md
│   requirements.txt
│   test.py
│   train.py
│
├───data
│       aligned_dataset.py
│       base_dataset.py
│       image_folder.py
│       __init__.py
│
├───models
│       base_model.py
│       losses.py
│       networks.py
│       ocr_kraken.py
│       pix2pix_model.py
│       __init__.py
│
├───options
│       base_options.py
│       test_options.py
│       train_options.py
│
└───utile
        DiffAugment_pytorch.py
```

---

## Quick start

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Train the model**

```bash
python train.py --dataroot /path/to/data --name my_experiment --n_epochs 100 --n_epochs_decay 100
```

3. **Test / inference**

```bash
python test.py --dataroot /path/to/data --name my_experiment --results_dir ./results --num_test 200
```

---

## Command-line arguments (summary)

The project is fully configurable via command-line arguments. Below is a summary of the key options available.

### Core experiment parameters

* `--dataroot`: Path to the dataset (must contain folders like `trainA`, `trainB`).
* `--name`: Name of the experiment (used for saving checkpoints and logs).
* `--checkpoints_dir`: Directory where model checkpoints are saved.

### Model architecture parameters

* `--input_nc` / `--output_nc`: Number of input and output image channels.
* `--ngf`: Number of generator filters in the last convolutional layer.
* `--norm`: Normalization type (`instance`, `batch`, or `none`).
* `--init_type` / `--init_gain`: Weight initialization type and scaling.
* `--no_dropout`: Disable dropout in generator layers.
* `--num_downs`: Number of downsampling layers in the U-Net generator.
* `--height_down_layers`: Number of layers reducing feature map height.

### Data loading and preprocessing

* `--batch_size`: Batch size for training.
* `--load_size_height` / `--load_size_width`: Resize dimensions before cropping.
* `--crop_size_height` / `--crop_size_width`: Final cropped dimensions.
* `--preprocess`: Image preprocessing strategy (`resize`, `crop`, `augment`, etc.).
* `--no_flip`: Disable random horizontal flipping.
* `--num_threads`: Number of data loading threads.
* `--max_dataset_size`: Maximum number of samples to load.

### Training process

* `--n_epochs` / `--n_epochs_decay`: Number of epochs for constant and decayed learning rate.
* `--save_epoch`: Frequency (in epochs) to save checkpoints.
* `--continue_train`: Resume training from the latest checkpoint.
* `--epoch_count`: Starting epoch index.
* `--pretrain_epochs`: Epochs for generator-only warmup.
* `--start_epoch_ocr`: Epoch to start OCR loss.
* `--val_count`: Frequency of validation runs.
* `--max_batches_val`: Number of validation batches.

### Loss weighting and types

* `--loss_Perceptual`: Type of perceptual loss (`lpips` or `contextual`).
* `--lambda_perceptual`: Weight of the perceptual loss term.
* `--lambda_fm`: Weight for feature-matching loss.
* `--lambda_ocr`: Weight for OCR loss.
* `--lambda_Kl_logits`: Weight for logits-based regularization.
* `--lambda_l1_feat`: Weight for L1 feature-matching loss.
* `--no_L1_loss`: Disable the L1 reconstruction loss.

### Optimization and learning rate

* `--lr_G` / `--lr_D`: Learning rates for generator and discriminator.
* `--beta1_G`, `--beta2_G`, `--beta1_D`, `--beta2_D`: Adam optimizer parameters.
* `--lr_policy`: Learning rate scheduling policy (`linear`, `step`, `cosine`, etc.).
* `--lr_decay_iters`: Steps before applying learning rate decay.

### GAN configuration

* `--gan_mode`: Type of GAN objective (`vanilla`, `lsgan`, `hinge`).
* `--ndf`: Number of discriminator filters.
* `--n_layers_D`: Number of layers in multi-scale discriminators.
* `--num_D`: Number of discriminators.

### Output and debugging

* `--results_dir`: Directory to save test results.
* `--verbose`: Print detailed logs.
* `--suffix`: Add custom suffix to experiment name.

---

## Extending the code

* Add new generator or discriminator architectures in `models/networks.py`.
* Create custom loss functions in `models/losses.py`.
* Integrate OCR-based constraints in `models/ocr_kraken.py`.
* Adjust preprocessing logic in `data/aligned_dataset.py` for new tasks.
