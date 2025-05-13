from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 32,
        latent_dim: int = 256,
        channel_mult: List[int] = [2, 4],
        image_size: Tuple[int, int] = (32, 32),
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        enable_bn: bool = True,
    ):
        super(Encoder, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.latent_dim = latent_dim
        self.channel_mult = channel_mult
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.layers = nn.ModuleList()

        current_processing_channels = self.in_channels

        self.layers.append(
            nn.Conv2d(
                in_channels=current_processing_channels,
                out_channels=self.model_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        )
        if enable_bn:
            self.layers.append(nn.BatchNorm2d(self.model_channels))
        self.layers.append(nn.LeakyReLU())
        current_processing_channels = self.model_channels

        for mult in self.channel_mult:
            out_ch = current_processing_channels * mult
            self.layers.append(
                nn.Conv2d(
                    in_channels=current_processing_channels,
                    out_channels=out_ch,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                )
            )
            if enable_bn:
                self.layers.append(nn.BatchNorm2d(out_ch))
            self.layers.append(nn.LeakyReLU())
            current_processing_channels = out_ch

        final_H = self.image_size[0]
        final_W = self.image_size[1]

        num_strided_conv_layers = 1 + len(self.channel_mult)

        for _ in range(num_strided_conv_layers):
            final_H = (final_H - self.kernel_size + 2 * self.padding) // self.stride + 1
            final_W = (final_W - self.kernel_size + 2 * self.padding) // self.stride + 1

        self.layers.append(nn.Flatten())

        num_flattened_features = current_processing_channels * final_H * final_W

        self.layers.append(
            nn.Linear(
                num_flattened_features,
                2 * self.latent_dim,
            )
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        mu = x[:, : self.latent_dim]
        log_var = x[:, self.latent_dim :]

        return mu, log_var


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        output_image_channels: int = 3,
        model_channels: int = 32,
        channel_mult: List[int] = [2, 4],
        target_image_size: Tuple[int, int] = (32, 32),
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        enable_bn: bool = True,
    ):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_image_channels = output_image_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.target_image_size = target_image_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.layers = nn.ModuleList()

        channels_at_bottleneck = self.model_channels
        for mult_val in self.channel_mult:
            channels_at_bottleneck *= mult_val

        num_strided_conv_layers_encoder = 1 + len(self.channel_mult)

        current_H = self.target_image_size[0]
        current_W = self.target_image_size[1]
        for _ in range(num_strided_conv_layers_encoder):
            current_H = (
                current_H - self.kernel_size + 2 * self.padding
            ) // self.stride + 1
            current_W = (
                current_W - self.kernel_size + 2 * self.padding
            ) // self.stride + 1

        num_flattened_features = channels_at_bottleneck * current_W * current_H

        self.layers.append(nn.Linear(self.latent_dim, num_flattened_features))
        if enable_bn:
            self.layers.append(nn.BatchNorm1d(num_flattened_features))
        self.layers.append(nn.LeakyReLU())

        current_processing_channels = channels_at_bottleneck

        for mult_val in self.channel_mult[::-1]:
            out_c_deconv = current_processing_channels // mult_val
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=current_processing_channels,
                    out_channels=out_c_deconv,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                )
            )
            if enable_bn:
                self.layers.append(nn.BatchNorm2d(out_c_deconv))
            self.layers.append(nn.LeakyReLU())
            current_processing_channels = out_c_deconv

        self.layers.append(
            nn.ConvTranspose2d(
                in_channels=self.model_channels,
                out_channels=self.output_image_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        )

        self.layers.append(nn.Sigmoid())

        self.bottleneck_channels = channels_at_bottleneck
        self.bottleneck_H = current_H
        self.bottleneck_W = current_W

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.layers[0](z)
        x = self.layers[1](x)
        x = x.view(-1, self.bottleneck_channels, self.bottleneck_H, self.bottleneck_W)
        for layer in self.layers[2:]:
            x = layer(x)

        return x


class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 32,
        latent_dim: int = 256,
        channel_mult: List[int] = [2, 4],
        image_size: Tuple[int, int] = (32, 32),
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        enable_bn: bool = True,
    ):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = Encoder(
            in_channels=in_channels,
            model_channels=model_channels,
            latent_dim=latent_dim,
            channel_mult=channel_mult,
            image_size=image_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            enable_bn=enable_bn,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_image_channels=in_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            target_image_size=image_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            enable_bn=enable_bn,
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, log_var = self.encoder(x)

        z = self.reparameterize(mu, log_var)

        reconstructed_x = self.decoder(z)

        return {
            "reconstructed_x": reconstructed_x,
            "mu": mu,
            "log_var": log_var,
            "z": z,
        }

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z_sampled = torch.randn(num_samples, self.latent_dim).to(device)

        generated_samples = self.decoder(z_sampled)
        return generated_samples
