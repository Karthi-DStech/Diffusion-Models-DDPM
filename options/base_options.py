import argparse


def get_arguments():
    """
    arguments for the training of the diffusion model

    Returns
    -------
    argparse.Namespace
        Arguments for the training of the diffusion model
    """

    parser = argparse.ArgumentParser(description="Train a diffusion model")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size")

    parser.add_argument(
        "--image_channels",
        type=int, 
        default=1, 
        help="Number of image channels"
    )

    parser.add_argument(
        "--image_size", 
        type=int, 
        default=32, 
        help="Size of the image")

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate")

    parser.add_argument(
        "--Time_steps_FD",
        type=int,
        default=1000,
        help="Number of time steps for the diffusion model",
    )

    parser.add_argument(
        "--epochs", 
        type=int, 
        default=2, 
        help="Number of epochs")

    parser.add_argument(
        "--nb_images", 
        type=int, 
        default=16, 
        help="Number of images to generate"
    )

    return parser.parse_args()
