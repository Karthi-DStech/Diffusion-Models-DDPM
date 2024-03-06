import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.diffusion_model import DiffusionModel
from models.unet import UNet

from options.base_options import get_arguments

args = get_arguments()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


"""
Train the diffusion model
-------------------------    
"""
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
diffusion_model = DiffusionModel(args.Time_steps_FD, model, device)

"""
Training Process
----------------

The training process includes logging various metrics and outputs to TensorBoard for visualization and analysis,
such as training loss, learning rate, model parameters, and gradients. Additionally, it saves training loss plots
and samples of generated images at specified intervals.

The main training loop iterates over a specified number of epochs, performing training steps, logging relevant
information to TensorBoard, and saving outputs periodically.

Features:
- Training loss is logged and downloaded the loss for every 100 epochs.
- Generated images and a batch of real images are logged every 2000 epochs for comparison.
- Training loss plots are saved to disk every 100 epochs.
- Generated images are saved to disk every 2000 epochs.

Note: It's essential to define the UNet model, DiffusionModel, and the sample_batch function, along with initializing
the optimizer, device, learning rate, batch size, number of images to generate, total epochs, and the training dataset.

"""

training_loss = []
for epochs in tqdm(range(args.epochs)):
    loss = diffusion_model.training_steps(args.batch_size, optimizer)
    print(f'Epoch {epochs}, Loss: {loss:.4f}')
    training_loss.append(loss)

    if epochs % 100 == 0:
        plt.plot(training_loss)
        plt.savefig('training_loss.png')
        plt.close()

        plt.plot(training_loss[-1000:])
        plt.savefig('training_loss_cropped.png')
        plt.close()

    if epochs % 2000 == 0:
        samples = diffusion_model.sampling(n_samples=args.nb_images, use_tqdm=False)
        plt.figure(figsize=(10, 10))
        for i in range(args.nb_images):
            plt.subplot(4, 4, i+1)
            plt.imshow(samples[i].squeeze().clip(0, 1).data.cpu().numpy(), cmap='gray')
            plt.axis('off')
        plt.savefig(f'sample_images_of_epoch_{epochs}.png')
        plt.close()

        torch.save(model.cpu(), f'model_paper2_epoch_{epochs}')
        model.cuda()
