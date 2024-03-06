from torch.utils.tensorboard import SummaryWriter

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.unet import UNet
from models.diffusion_model import DiffusionModel
from utils.images_utils import sample_batch
from utils.images_utils import train_dataset

from options.base_options import get_arguments

args = get_arguments()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = UNet() 
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
print(sum([p.numel() for p in model.parameters()]))
print(model)

"""
Training Process
----------------

The training process includes logging various metrics and outputs to TensorBoard for visualization and analysis,
such as training loss, learning rate, model parameters, and gradients. Additionally, it saves training loss plots
and samples of generated images at specified intervals.

The main training loop iterates over a specified number of epochs, performing training steps, logging relevant
information to TensorBoard, and saving outputs periodically.

Features:
- Training loss is logged and visualized in TensorBoard at every epoch.
- Model parameters and gradients are logged every 100 epochs.
- The learning rate is logged at every epoch.
- Generated images and a batch of real images are logged every 4000 epochs for comparison.
- Training loss plots are saved to disk every 100 epochs.
- Generated images are saved to disk every 4000 epochs.

The script concludes by closing the TensorBoard writer to ensure all logs are properly saved.

Note: It's essential to define the UNet model, DiffusionModel, and the sample_batch function, along with initializing
the optimizer, device, learning rate, batch size, number of images to generate, total epochs, and the training dataset.

"""

diffusion_model = DiffusionModel(args.Time_steps_FD, model, device)  

writer = SummaryWriter()  # Initialize TensorBoard SummaryWriter

training_loss = []

for epoch in tqdm(range(args.epochs)):
    loss = diffusion_model.training_steps(args.batch_size, optimizer)
    print(f'Epoch {epoch}, Loss: {loss:.4f}')
    training_loss.append(loss)

    # TensorBoard: Log training loss
    writer.add_scalar("Training Loss", loss, epoch)

    # Log model parameters and gradients every 100 epochs
    if epoch % 100 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(f"Params/{name}", param, epoch)
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

        # Log learning rate
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        # Save training loss plot
        plt.plot(training_loss)
        plt.savefig(f'training_loss.png')
        plt.close()

        plt.plot(training_loss[-1000:])
        plt.savefig('training_loss_cropped.png')
        plt.close()

    # Every 1000 epochs, log generated and real images
    if epoch % 4000 == 0:
        samples = diffusion_model.sampling(n_samples=args.nb_images, use_tqdm=False)

        # Normalize and prepare samples for TensorBoard
        samples_tb = samples.clip(0, 1)
        writer.add_images("Generated Images", samples_tb, epoch)

        # Visualize generated images
        plt.figure(figsize=(10, 10))
        for i in range(args.nb_images):
            plt.subplot(4, 4, i+1)
            plt.imshow(samples[i].cpu().numpy().transpose(1, 2, 0), cmap='gray')
            plt.axis('off')
        plt.savefig(f'sample_images_epoch_{epoch}.png')
        plt.close()

        # Log real images for comparison
        x0_real = sample_batch(args.nb_images, train_dataset, device)  # Ensure this function and dataset are correctly defined
        x0_real_tb = x0_real.clip(0, 1)  # Normalize for TensorBoard
        writer.add_images("Real Images", x0_real_tb, epoch)

        torch.save(model.cpu(), f'model_paper2_epoch_{epoch}')
        model.cuda()

# Close the TensorBoard writer after the loop
writer.close()

