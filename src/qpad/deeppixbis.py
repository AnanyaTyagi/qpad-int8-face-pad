"""DeepPixBiS (George & Marcel, ICB 2019).

Reference implementation: Idiap ``bob.paper.deep_pix_bis_pad`` /
``bob.learn.pytorch`` ``DeepPixBiS`` network, reproduced here:

* encoder = first 8 children of ImageNet-pretrained DenseNet-161 ``features``
  (conv0, norm0, relu0, pool0, denseblock1, transition1, denseblock2, transition2),
  giving a 384 x 14 x 14 feature map for a 224 x 224 input;
* ``dec`` = 1x1 convolution 384 -> 1 followed by a sigmoid -> 14 x 14 pixel map;
* ``linear`` = fully connected 196 -> 1 on the flattened map, sigmoid -> binary output.

Training loss: ``beta * BCE(map, y) + (1 - beta) * BCE(binary, y)`` with beta = 0.5,
where the map target is a constant 14 x 14 map of the presentation label
(1 = bona fide, 0 = attack). Score at test time: mean of the pixel map.
"""

from __future__ import annotations

import torch
from torch import nn

ENCODER_CHILDREN = 8
ENCODER_CHANNELS = 384
MAP_SIZE = 14


class DeepPixBiS(nn.Module):
    """DeepPixBiS network; ``forward`` returns ``(pixel_map, binary)`` (both in [0, 1])."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision import models

        weights = models.DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None
        dense = models.densenet161(weights=weights)
        features = list(dense.features.children())
        self.enc = nn.Sequential(*features[:ENCODER_CHILDREN])
        self.dec = nn.Conv2d(ENCODER_CHANNELS, 1, kernel_size=1, padding=0)
        self.linear = nn.Linear(MAP_SIZE * MAP_SIZE, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.enc(x)
        out_map = torch.sigmoid(self.dec(enc))                          # (N, 1, 14, 14)
        out = torch.sigmoid(self.linear(out_map.flatten(start_dim=1)))  # (N, 1)
        return out_map, out.flatten()


class DeepPixBiSLoss(nn.Module):
    """``beta * pixel-wise BCE + (1 - beta) * binary BCE``."""

    def __init__(self, beta: float = 0.5) -> None:
        super().__init__()
        self.beta = float(beta)
        self.bce = nn.BCELoss()

    def forward(self, out_map: torch.Tensor, out_bin: torch.Tensor, labels: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = labels.float()
        target_map = labels.view(-1, 1, 1, 1).expand_as(out_map)
        l_pix = self.bce(out_map.float(), target_map)
        l_bin = self.bce(out_bin.float(), labels)
        return self.beta * l_pix + (1.0 - self.beta) * l_bin, l_pix, l_bin


def map_to_frame_score(out_map: torch.Tensor) -> torch.Tensor:
    """Frame score = mean of the 14 x 14 realness map (higher = more bona fide)."""
    return out_map.flatten(start_dim=1).mean(dim=1)


class ScoreHead(nn.Module):
    """Wrapper whose single output is the per-frame score (used for some exports)."""

    def __init__(self, net: DeepPixBiS) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return map_to_frame_score(self.net(x)[0])
