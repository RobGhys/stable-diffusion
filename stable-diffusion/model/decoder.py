import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape: bs, in_channels, h, w
        residue = x
        n, c, h, w = x.shape

        # bs, in_channels, h, w -> bs, in_channels, (h * w)
        # in the sequence, each item represents a pixel (because h*w)
        x = x.view(n, c, h * w)

        # bs, in_channels, (h * w) -> bs, (h * w), in_channels
        x = x.transpose(-1, -2)

        # bs, (h * w), in_channels -> bs, (h * w), in_channels
        x = self.attention(x)

        # bs, (h * w), in_channels -> bs, in_channels, (h * w)
        x = x.transpose(-1, -2)

        # bs, in_channels, (h * w) -> bs, in_channels, h, w
        x = x.view((n, c, h, w))

        x += residue

        return x

class VAE_ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual_layer = None

        self.groupnorm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape: bs, in_channels, h, w
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, padding=0),
            nn.Conv2d(in_channels=4, out_channels=512, kernel_size=3, padding=1),
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            VAE_AttentionBlock(channels=512),
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            VAE_ResidualBlock(in_channels=512, out_channels=512),

            # bs, 512, h/8, w/8 -> bs, 512, h/8, w/8
            VAE_ResidualBlock(in_channels=512, out_channels=512),

            # replicates the pixels twice to double height and width
            # bs, 512, h/8, w/8 -> bs, 512, h/4, w/4
            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # bs, 512, h/4, w/4 -> bs, 512, h/2, w/2
            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # bs, 256, h/2, w/2 -> bs, 256, h, w
            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(num_groups=32, num_channels=128),

            nn.SiLU(),

            # bs, 128, h, w -> bs, 3, h, w
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape: bs, 4, h/8, w/8
        # reverse scaling
        x /= 0.18215

        for module in self:
            x = module(x)

        # bs, 3, h, w
        return x