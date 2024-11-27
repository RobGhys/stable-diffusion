import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # bs, in_channels, height, width) -> (bs, out_channels=128, height, width)
            # padding ensures h_1 = h2 & w_1 = w_2
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),

            # bs, 128, h, w -> bs, 128, h, w
            VAE_ResidualBlock(in_channels=128, out_channels=128),
            # bs, 128, h, w -> bs, 128, h, w
            VAE_ResidualBlock(in_channels=128, out_channels=128),

            # bs, 128, h, w -> bs, 128, h/2, w/2
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0),

            # bs, 128, h, w -> bs, 256, h/2, w/2
            VAE_ResidualBlock(in_channels=128, out_channels=256),
            # bs, 256, h, w -> bs, 256, h/2, w/2
            VAE_ResidualBlock(in_channels=256, out_channels=256),

            # bs, 256, h/2, w/2 -> bs, 256, h/4, w/4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0),

            # bs, 256, h/4, w/4 -> bs, 512, h/4, w/4
            VAE_ResidualBlock(in_channels=256, out_channels=512),
            # bs, 512, h/4, w/4 -> bs, 512, h/4, w/4
            VAE_ResidualBlock(in_channels=512, out_channels=512),

            # bs, 512, h/4, w/4 -> bs, 512, h/8, w/8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),

            # bs, 512, h/8, w/8 -> bs, 512, h/8, w/8
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            # bs, 512, h/8, w/8 -> bs, 512, h/8, w/8
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            # bs, 512, h/8, w/8 -> bs, 512, h/8, w/8
            VAE_ResidualBlock(in_channels=512, out_channels=512),

            # bs, 512, h/8, w/8 -> bs, 512, h/8, w/8
            VAE_AttentionBlock(channels=512),

            # bs, 512, h/8, w/8 -> bs, 512, h/8, w/8
            VAE_ResidualBlock(in_channels=512, out_channels=512),

            # bs, 512, h/8, w/8 -> bs, 512, h/8, w/8
            nn.GroupNorm(num_groups=32, num_channels=512),

            # bs, 512, h/8, w/8 -> bs, 512, h/8, w/8
            nn.SiLU(),

            # Encoder Bottleneck
            # bs, 512, h/8, w/8 -> bs, 8, h/8, w/8
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, padding=1),

            # bs, 8, h/8, w/8 -> bs, 8, h/8, w/8
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x is: bs, in_channel, h, w
        # noise is: bs, out_channel, h/8, w/8
        for module in self:
            # when stride=2
            if getattr(module, 'stride', None) == (2, 2):
                # padding_left, padding_right, padding_top, padding_bottom
                # hence, add padding -> on the right_side and bottom
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        # bs, 8, h/8, w/8 -> TWO tensors of shape: bs, 4, h/8, w/8
        mean, log_variance = torch.chunk(x, chunks=2, dim=1)

        # bs, 4, h / 8, w / 8 -> bs, 4, h/8, w/8
        # clamp the log_variance so that it does not become too small nor too big
        log_variance = torch.clamp(log_variance, min=-30., max=20.)

        # bs, 4, h / 8, w / 8 -> bs, 4, h/8, w/8
        # exp to get the variance
        variance = log_variance.exp()
        # bs, 4, h / 8, w / 8 -> bs, 4, h/8, w/8
        std = variance.sqrt()

        # Z = N(0, 1) -> X = N(mean, variance)
        # X = mean + std * Z --> transfroms Z distribution into X distribution
        x = mean + std * noise

        # scale the output by a constant
        x *= 0.18215

        return x
