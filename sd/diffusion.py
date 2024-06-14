import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):

    def __init__(self, n_embd:int):
        super().__init__()

        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, 4*n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (1, 320)
        x = self.linear_1(x)

        x = F.silu(x)

        x = self.linear_2(x)

        # (1, 1280)
        return x

class UpSample(nn.Module):

    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, featrues, height, width) -> (batch_size, features, height * 2, width * 2)
        x = F.interpolate(x, scale_factor=2, mode='neartest')
        return self.conv(x)

class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context) #note: UNET_attention will compute the cross attention between latent and prompt
            elif isinstance(layer, UNET_ResidualBlock): #note: will match latent with its timestamp
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoders - nn.Module([
            # (batch_size, 4, height/8, width/8) -> (batch_size, 320, height/8, width/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),

            # (batch_size, 320, height/8, width/8) -> (batch_size, 320, height/16, width/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320,640), UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(640,640), UNET_AttentionBlock(8,80)),

            # (batch_size, 640, height/16, width/16) -> (batch_size, 640, height/32, width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640,1280), UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(1280,1280), UNET_AttentionBlock(8,160)),

            # (batch_size, 1280, height/32, width/32) -> (batch_size, 1280, height/64, width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280,1280)),

            # (batch_size, 1280, height/64, width/64) -> (batch_size, 1280, height/64, width/64)
            SwitchSequential(UNET_ResidualBlock(1280,1280))

        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280,1280),

            UNET_AttentionBlock(8,160),

            UNET_ResidualBlock(1280,1280)
        )

        self.decoders = nn.ModuleList([
            # (batch_size, 2560, height/64, width/64) -> (batch_size, 1280, height/64, width/64)
            SwitchSequential(UNET_ResidualBlock(2560,1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280), UpSample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(2560,1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920,1280), UNET_AttentionBlock(8, 160), UpSample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920,640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1280,640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(960,640), UNET_AttentionBlock(8, 80), UpSample(640)),

            SwitchSequential(UNET_ResidualBlock(960,320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640,320), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640,320), UNET_AttentionBlock(8, 40)),

        ])
    
class Unet_OutputLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    
    def forward(self, x):
        # (batch_size, 320, height/8, width/8)
        x = self.groupnorm(x)

        x = F.silu(x)

        x = self.conv(x)

        return x
class Diffusion(nn.Module):

    def __init__(self):
        # note: we need to give the U-net not only the noisified image but also the time step at which the image was noisified.
        self.time_embedding = TimeEmbedding(320) #320 is the size of time embedding
        self.unet = UNET()
        self.final = Unet_OutputLayer(320,4)

    
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        #latent (batch_size, 4, height/8, width/8)
        # context: (batch_size, seq_len, dim)
        # time: (1, 320)

        # (1,320) -> (1, 1280)
        time = self.time_embedding(time)

        #batch_size, 4, height/8, width/8) -> (batch_size, 320, height/8, width/8)
        output = self.unet(latent, context, time)

        # (batch_size, 320, height/8, width/8) -> (batch_size, 4, height/8, width/8
        output =  self.final(output)

        # (batch_size, 4, height/8, width/8)
        return output

