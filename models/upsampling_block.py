import torch.nn as nn
import torch.nn.functional as F




# UPSAMPLING BLOCK


class UpSample(nn.Module):
    """
    This class implements the upsampling block for the UNet model.
    
    Parameters
    ----------
    C : int
        Number of channels in the input image  
    """
    def __init__(self, C):
        super(UpSample, self).__init__()

        self.conv = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        The forward pass for the upsampling block.
        
        Parameters
        ----------
        x : torch.Tensor
            Input image
        
        Returns
        -------
        torch.Tensor
            Upsampled image
        """
        B, C, H, W = x.shape

        x = F.interpolate(x, size=None, scale_factor=2, mode="nearest")
        x = self.conv(x)

        assert x.shape == (B, C, H * 2, W * 2)
        return x



# Test the function of the upsampling block
    
"""
t = (torch.rand(100) * 10).long()
get_timestep_embedding(t, 64)

downsample = DownSample(64)
img = torch.randn((10, 64, 400, 400))
hidden = downsample(img)

upsample = UpSample(64)
img = upsample(hidden)
print(img.shape)

"""