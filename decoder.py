import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, channels)
        self.attention   = SelfAttention(1, channels)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, height, width)

        residue = x

        n,c,h,w = x.shape

        # (batch_size, Features , height, width) -> (batch_size, Features, height * width)
        x.view(n,c,h*w)

        # (batch_size, Features, height * width) -> (batch_size, height * width, Features)
        # note: we have a sequence of pixels with each its own embedding and we relate them to each other
        x = x.transpose(-1,-2)
        
        # (batch_size, height * width, Features) -> (batch_size, height * width, Features)
        x = self.attention(x) #doesn't change the shape

        # (batch_size, height * width, Features) -> (batch_size, Features, height * width)
        x = x.transpose(-1,-2)

        x = x.view((n,c,h,w))
        
        x += residue

        return x
    


    
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, height, width)

        residue =x 

        x = self.groupnorm_1(x)
        
        x = F.silu(x)

        x = self.conv_1(x)

        x = self.groupnorm_2(x)

        x = F.silu(x)

        x = self.conv_2(x)

        return x + self.residual_layer(residue)