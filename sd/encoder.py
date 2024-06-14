import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            #(batch_size, channels, height, width) -> (batch_size, 128, height, width)
            nn.Conv2d(3,128, kernel_size=3, padding=1),

            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            #(batch_size, 128, height, width) - > (batch_size, 128, height/2, width/2)
            nn.Conv2d(128,128, kernel_size=3, stride=2, padding=0),

            # (batch_size, 128, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(128,256),

            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(256,256),

            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/4, width/4)
            nn.Conv2d(256,256, kernel_size=3, stride=2, padding=0),

            # (batch_size, 256, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(256,512),

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(512,512),

            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8
            VAE_AttentionBlock(512), #relate pixels to each other in the image. (size remains the same)

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512,512),

            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            nn.GroupNorm(32, 512), #normalization does not change the size of the image

            nn.SiLU(), #practically it works better

             # (batch_size, 512, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1), #"bottleneck" of the encoder, reduce the features from 512 to 8

             # (batch_size, 8, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(8,8, kernel_size=1, padding=0)
            )
    
    def forward(self, x:torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channel, height, width)
        # noise: (batch_size, out_channels, height/8, width/8)

        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                # (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom)
                x = F.pad(x, (0,1,0,1)) #asymmetrical padding for convolutions with stride==2
            
            x = module(x)
        
        # (batch_size, 8, height/8, width/8) -> (batch_size, 4, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1) #split the tensor into 2 parts

        # (batch_size, 4, height/8, width/8) -> (batch_size, 4, height/8, width/8)
        log_variance = torch.clamp(log_variance, min=-30, max=20) #clamping the log_variance to avoid numerical instability

        # (batch_size, 4, height/8, width/8) -> (batch_size, 4, height/8, width/8)
        variance = torch.exp(log_variance)

        # (batch_size, 4, height/8, width/8) -> (batch_size, 4, height/8, width/8)
        stdev = torch.sqrt(variance) #standard deviation

        #Z = N(0,1) - > N(mean, stdev)?
        # X = mean + stdev * Z

        x = mean + stdev * noise

        #Scale the output by a constant (done in the original stable diffusion)
        x *= 0.18215

        return x
