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
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(320,320), UNET_AttentionBlock(8,40)),

            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320,640), UNET_AttentionBlock(8,80)),

        ])
    
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

