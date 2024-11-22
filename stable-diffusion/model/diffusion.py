import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, n_embeddings: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embeddings, n_embeddings * 4,)
        self.linear_2 = nn.Linear(n_embeddings * 4, n_embeddings * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        # (1, 1280)
        return x

class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        # feature shape: bs, in_channels, h, w
        # time shape: 1, 1280
        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        time = F.silu(time)
        time = self.linear_time(time)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class UNetAttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_emb: int, d_context: int=768):
        super().__init__()
        channels = n_head * n_emb
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        # x: bs, features, h, w
        # context: bs, seq_len, dim
        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        n, c, h, w = x.shape

        # bs, features, h, w -> bs, features, h * w
        x = x.view((n, c, h*w))
        # bs, features, h * w -> bs, h * w, features
        x = x.transpose(-1, -2)
        # normalization + self attention with skip connection
        residue_short = x

        x = self.layernorm_1(x)
        self.attention_1(x)
        x += residue_short

        residue_short = x

        # normalization + cross attention with skip connection
        x = self.layernorm_2(x)

        # Cross Attention
        self.attention_2(x, context)

        x += residue_short

        residue_short = x

        # normalization + FF with GeGLU and skip connection
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)
        x += residue_short

        # bs, h * w, features -> bs, features, h * w
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long

class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # bs, features, h, w -> bs, features, h*2, w*2
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time:torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNetAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNetResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # reduce the size of the image, but increases number of features
        self.encoders = nn.Module([
            # bs, 4, h/8, w/8
            SwitchSequential(nn.Conv2d(4, 320, kenrel_size=3, padding=1)),
            SwitchSequential(UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)),
            SwitchSequential(UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)),

            # bs, 320, h/8, w/8 -> bs 320, h/16, w/16
            SwitchSequential(nn.Conv2d(320, 320, kenrel_size=3, stride=2, padding=1)),
            SwitchSequential(UNetResidualBlock(320, 640), UNetAttentionBlock(8, 80)),
            SwitchSequential(UNetResidualBlock(640, 640), UNetAttentionBlock(8, 80)),

            # bs 640, h / 16, w / 16 -> bs 640, h/32, w/32
            SwitchSequential(nn.Conv2d(640, 640, kenrel_size=3, stride=2, padding=1)),
            SwitchSequential(UNetResidualBlock(640, 1280), UNetAttentionBlock(8, 160)),
            SwitchSequential(UNetResidualBlock(1280, 1280), UNetAttentionBlock(8, 160)),

            # bs 1280, h/32, w/32 -> bs 1280, h/64, w/64
            SwitchSequential(nn.Conv2d(1280, 1280, kenrel_size=3, stride=2, padding=1)),
            SwitchSequential(UNetResidualBlock(1280, 1280)),
            # bs, 1280, h/64, w/64 -> bs, 1280, h/64, w/64
            SwitchSequential(UNetResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNetResidualBlock(1280, 1280),
            UNetAttentionBlock(8, 160),
            UNetResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            # bs, 2560, h/64, w/64 -> bs, 1280, h/64, w/64
            SwitchSequential(UNetResidualBlock(2560, 1280)),
            SwitchSequential(UNetResidualBlock(2560, 1280)),
            SwitchSequential(UNetResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)),
            SwitchSequential(UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)),
            SwitchSequential(UNetResidualBlock(1920, 1280), UNetAttentionBlock(8, 160), Upsample(1280)),
            SwitchSequential(UNetResidualBlock(1920, 640), UNetAttentionBlock(8, 80)),
            SwitchSequential(UNetResidualBlock(1280, 640), UNetAttentionBlock(8, 80)),
            SwitchSequential(UNetResidualBlock(960, 640), UNetAttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(UNetResidualBlock(960, 320), UNetAttentionBlock(8, 40)),
            SwitchSequential(UNetResidualBlock(640, 320), UNetAttentionBlock(8, 80)),
            SwitchSequential(UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)),

        ])

class UNetOutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: bs, 320, h/8, w/8
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)

        # bs, 4, h/8, w/8
        return x

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNetOutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent has shape : bs, 4, h/8, w/8
        # context has shape : bs, seq_len, dim
        # time has shape : (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        # bs, 4, h/8, w/8 -> bs, 320, h/8, w/8
        output = self.unet(latent, time)

        # bs, 320, h/8, w/8 -> bs, 4, h/8, w/8
        output = self.final(output)

        # bs, 4, h/8, w/8
        return output # latent 'z'