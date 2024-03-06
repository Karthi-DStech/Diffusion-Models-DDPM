import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from options.base_options import get_arguments

args = get_arguments()


transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=args.image_channels),
        transforms.ToTensor(),
        transforms.Resize(args.image_size, antialias=True),
    ]
)

train_dataset = datasets.MNIST(
    root="./data", download=True, train=True, transform=transform
)


def sample_batch(batch_size, dataset, device):

    sampler = SubsetRandomSampler(torch.randperm(len(dataset))[:batch_size])
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    data_iter = iter(loader)
    images, _ = next(data_iter)
    images = images.to(device)
    return images
