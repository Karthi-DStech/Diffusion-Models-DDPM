# Diffusion-Models-MNIST

# Diffusion Model Training for Image Generation using Mnist Dataset

This project contains the code necessary to train a diffusion model for image generation. It includes a PyTorch implementation of the U-Net model, several building blocks used in the model architecture, and scripts for training and logging.

## Generated Image after 40_000 Epochs
<img width="259" alt="40_000 Epochs - 1" src="https://github.com/Karthi-DStech/Diffusion-Models-MNIST/assets/126179797/a4150c59-83c8-4b72-b60b-3b868d6fb2f9"><img width="257" alt="40_000 Epochs - 2" src="https://github.com/Karthi-DStech/Diffusion-Models-MNIST/assets/126179797/e3216984-d600-43ab-9656-d5dba6a407a3">
<img width="256" alt="40_000 Epochs - 3" src="https://github.com/Karthi-DStech/Diffusion-Models-MNIST/assets/126179797/7ad07086-60bb-4b15-a558-f139e6a861e8"><img width="256" alt="40_000 Epochs - 4" src="https://github.com/Karthi-DStech/Diffusion-Models-MNIST/assets/126179797/f93aa76e-e1f3-4e8c-b3ee-ed0d9a7e2cbf">







## Project Structure

- `models/`: Contains the individual modules used to build the diffusion model.
    - `attention_block.py`: Defines the attention mechanisms.
    - `diffusion_model.py`: The main diffusion model class.
    - `downsampling_block.py`: Modules for downsampling feature maps.
    - `nin_block.py`: Network in network block.
    - `resnet_block.py`: ResNet blocks.
    - `timestep_embedding.py`: Embedding layers for timesteps.
    - `unet.py`: U-Net model architecture.
    - `upsampling_block.py`: Modules for upsampling feature maps.
- `options/`:
    - `base_options.py`: Command-line arguments for the training script.
- `utils/`:
    - `images_utils.py`: Utilities for image handling.
- `train.py`: Script for training the model without TensorBoard logging.
- `updated_train.py`: Script for training the model with TensorBoard logging.

## Requirements

To run the code, you need the following:

- Python 3.8 or above
- PyTorch 1.7 or above
- torchvision
- tqdm
- matplotlib

Install the necessary packages using pip:


## Dataset

The training scripts are set up to use the MNIST dataset, which is automatically downloaded. If you wish to use a different dataset, you'll need to modify the `images_utils.py` file and potentially the training scripts to handle your dataset's loading and processing.

## Models Saving

The trained models are saved to the disk every 5000 epochs by default. You can change this frequency in the training scripts.



