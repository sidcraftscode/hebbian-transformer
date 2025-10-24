import torch
import torch.nn as nn


class Conv4Encoder(nn.Module):
    """
    Standard four-layer convolutional embedding used in many few-shot learning benchmarks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 64,
        output_dim: int = 256,
        input_resolution: int = 32,
    ) -> None:
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(4):
            layers.extend(
                [
                    nn.Conv2d(channels, hidden_size, kernel_size=3, padding=1),
                    nn.BatchNorm2d(hidden_size),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            channels = hidden_size
        self.encoder = nn.Sequential(*layers)
        self.output_dim = output_dim
        dummy = torch.zeros(1, in_channels, input_resolution, input_resolution)
        with torch.no_grad():
            projected = self.encoder(dummy)
        flat_dim = projected.view(1, -1).size(1)
        self.proj = nn.Linear(flat_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        flattened = features.view(features.size(0), -1)
        return self.proj(flattened)
