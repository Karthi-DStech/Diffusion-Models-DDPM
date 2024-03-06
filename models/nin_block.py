import torch
import torch.nn as nn
import numpy as np



# IMPLEMENTING NIN


class Nin(nn.Module):
    """
    This class implements the Network in Network block for the UNet model.
    
    Parameters
    ----------
    in_dim : int
        Input dimension
    
    out_dim : int
        Output dimension
        
    scale : float
        Scaling factor for the weights
    """
    def __init__(self, in_dim, out_dim, scale=1e-10):
        super(Nin, self).__init__()

        n = (in_dim + out_dim) / 2
        limit = np.sqrt(3 * scale / n)

        self.W = torch.nn.Parameter(
            torch.zeros((in_dim, out_dim), dtype=torch.float32).uniform_(-limit, limit)
        )
        self.b = torch.nn.Parameter(
            torch.zeros((1, out_dim, 1, 1), dtype=torch.float32)
        )

    def forward(self, x):
        """
        The forward pass for the Network in Network block.
        
        Parameters
        ----------
        x : torch.Tensor
        
        Returns
        -------
        torch.Tensor"""
        return torch.einsum("bchw, co->bowh", x, self.W) + self.b



# Test the function of the Network in Network block
    
"""

t = (torch.rand(100) * 10).long()
get_timestep_embedding(t, 64)

downsample = DownSample(64)
img = torch.randn((10, 64, 400, 400))
hidden = downsample(img)

upsample = DownSample(64)
img = upsample(hidden)
print(img.shape)

nin = Nin(64, 128)
print(nin(img).shape)

"""


