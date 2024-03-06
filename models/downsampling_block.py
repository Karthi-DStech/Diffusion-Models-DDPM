import torch.nn as nn





# DOWNSAMPLE BLOCK

class DownSample(nn.Module):
    """
    This class implements the downsampling block for the UNet model.

    Parameters
    ----------
    C : int
        Number of channels in the input image
    """
    def __init__(self, C):
        super(DownSample, self).__init__()

        self.conv = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        The forward pass for the downsampling block.

        Parameters
        ----------
        x : torch.Tensor
            Input image
        
        Returns
        -------
        torch.Tensor
            Downsampled image   
        """
        B, C, H, W = x.shape
        x = self.conv(x)

        assert x.shape == (B, C, H // 2, W // 2)
        return x




# Test the function of the downsampling block

"""
t = (torch.rand (100) * 10).long()
get_timestep_embedding (t, 64)

downsample = DownSample(64)
img = torch.randn((10, 64, 400, 400))
downsample(img)

"""