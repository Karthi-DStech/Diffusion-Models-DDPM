import torch
import torch.nn as nn
import torch.nn.functional as F

from models.downsampling_block import DownSample
from models.upsampling_block import UpSample
from models.resnet_block import ResNetBlock
from models.attention_block import AttentionBlock
from models.timestep_embedding import get_timestep_embedding




# IMPLEMENTING UNET
    
"""
Implements a diffusion model using a modified U-Net architecture with ResNet and Attention blocks,
    designed for generating or manipulating images through a series of forward and reverse diffusion
    processes. This architecture facilitates the gradual transformation of noise into structured images
    (or vice versa) by modeling the distribution of training data.

    The model structure includes three main sections - `down`, `middle`, and `up` - each contributing
    to handling different stages of the diffusion process:

    - `down`: This part of the model applies a series of ResNet blocks, Attention blocks, and down-sampling
      operations to encode the input into a compact, feature-rich representation. It's essential for
      extracting and enhancing relevant features from the input (or noise) and providing a strong foundation
      for the diffusion process.
      
    - `middle`: Serving as the bottleneck of the architecture, this section processes the encoded features
      from the `down` path, employing both ResNet and Attention blocks to capture the most abstract and
      critical features necessary for the diffusion process.
      
    - `up`: This sequence progressively up-samples and refines the feature maps, incorporating learned
      features from the `down` path through skip connections. The `up` path aims to reconstruct the output
      image from the abstract features, gradually transforming noise into coherent images or vice versa,
      depending on the direction of the diffusion process.

    The model utilizes ResNet blocks for robust feature extraction and representation learning, Attention
    blocks to focus on relevant features, and DownSample/UpSample blocks to modify the spatial resolution
    of the feature maps.
"""


class UNet(nn.Module):
    """
    This class implements the UNet model for the diffusion model.
    
    Parameters
    ----------
    ch : int
        Number of channels in the input image
        
    in_ch : int
            Number of input channels
    """

    def __init__(self, ch=128, in_ch=1):

        super(UNet, self).__init__()

        self.ch = ch
        self.linear1 = nn.Linear(ch, 4 * ch)
        self.linear2 = nn.Linear(4 * ch, 4 * ch)

        self.conv1 = nn.Conv2d(in_ch, ch, 3, stride=1, padding=1)

        """
        Attributes:
        -----------
        down (nn.ModuleList): Contains the layers for the contracting path of the model, including ResNet blocks,
                              Attention blocks, and down-sampling layers.

        middle (nn.ModuleList): Contains the layers for the bottleneck of the model, including ResNet blocks and
                                an Attention block for capturing the most abstract features.

        up (nn.ModuleList): Contains the layers for the expansive path of the model, including ResNet blocks, 
                            Attention blocks, and up-sampling layers. It is designed to refine and up-sample 
                            the feature maps to reconstruct the output image.

        final_conv (nn.Conv2d): A convolutional layer that maps the output of the `up` path to the desired 
                                number of output channels.
        """

        self.down = nn.ModuleList(
            [
                ResNetBlock(ch, 1 * ch),
                ResNetBlock(1 * ch, 1 * ch),
                DownSample(1 * ch),
                ResNetBlock(1 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                DownSample(2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
                DownSample(2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
            ]
        )

        self.middle = nn.ModuleList(
            [
                ResNetBlock(2 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
            ]
        )

        self.up = nn.ModuleList(
            [
                ResNetBlock(4 * ch, 2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                UpSample(2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                UpSample(2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                ResNetBlock(3 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                UpSample(2 * ch),
                ResNetBlock(3 * ch, ch),
                ResNetBlock(2 * ch, ch),
                ResNetBlock(2 * ch, ch),
            ]
        )

        self.final_conv = nn.Conv2d(ch, in_ch, 3, stride=1, padding=1)

    def forward(self, x, t):

        """
        The forward pass for the UNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input image

        t : torch.Tensor
            Timesteps for the diffusion model

        Returns
        -------
        torch.Tensor
            Output image    
        """
        

        temb = get_timestep_embedding(t, self.ch)
        temb = torch.nn.functional.silu(self.linear1(temb))
        temb = self.linear2(temb)
        assert temb.shape == (t.shape[0], self.ch * 4)

        x1 = self.conv1(x)

        # Down
        x2 = self.down[0](x1, temb)
        x3 = self.down[1](x2, temb)
        x4 = self.down[2](x3)
        x5 = self.down[3](x4, temb)
        x6 = self.down[4](x5)
        x7 = self.down[5](x6, temb)
        x8 = self.down[6](x7)
        x9 = self.down[7](x8)
        x10 = self.down[8](x9, temb)
        x11 = self.down[9](x10, temb)
        x12 = self.down[10](x11)
        x13 = self.down[11](x12, temb)
        x14 = self.down[12](x13, temb)

        # Middle
        x = self.middle[0](x14, temb)
        x = self.middle[1](x)
        x = self.middle[2](x, temb)

        # Up
        x = self.up[0](torch.cat((x, x14), dim=1), temb)
        x = self.up[1](torch.cat((x, x13), dim=1), temb)
        x = self.up[2](torch.cat((x, x12), dim=1), temb)
        x = self.up[3](x)
        x = self.up[4](torch.cat((x, x11), dim=1), temb)
        x = self.up[5](torch.cat((x, x10), dim=1), temb)
        x = self.up[6](torch.cat((x, x9), dim=1), temb)
        x = self.up[7](x)
        x = self.up[8](torch.cat((x, x8), dim=1), temb)
        x = self.up[9](x)
        x = self.up[10](torch.cat((x, x6), dim=1), temb)
        x = self.up[11](x)
        x = self.up[12](torch.cat((x, x4), dim=1), temb)
        x = self.up[13](x)
        x = self.up[14](x)
        x = self.up[15](torch.cat((x, x3), dim=1), temb)
        x = self.up[16](torch.cat((x, x2), dim=1), temb)
        x = self.up[17](torch.cat((x, x1), dim=1), temb)

        x = F.silu(F.group_norm(x, num_groups=32))
        x = self.final_conv(x)

        return x



# Test the function of the UNet model


"""


t = (torch.rand (10) * 10).long()
temb= get_timestep_embedding (t, 512)

downsample = DownSample(64)
img = torch.randn((10, 64, 16, 16))
hidden  = downsample(img)

upsample = UpSample(64)
img = upsample(hidden)
#print(img.shape)

nin = Nin(64, 128)
#print(nin(img).shape)
img = nin(img)

resnet = ResNetBlock(128, 128, 0.1)
img = resnet(img, temb)
#print(img.shape)

resnet = ResNetBlock(128, 64, 0.1)
img = resnet(img, temb)
#print(img.shape)

att = AttentionBlock(64)
img = att(img)
#print(img.shape)

img = torch.randn((10, 1, 32, 32))
model = UNet()
img = model(img, t)

print(img.shape)

print(sum([p.numel() for p in model.parameters()]))

"""