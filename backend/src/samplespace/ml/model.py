"""Dual-head CNN for audio sample classification and embedding.

Architecture: 4 residual conv blocks with SE attention -> global average pooling -> dual head:
  - Channel progression: 1 -> 64 -> 128 -> 256 -> 512
  - Classification head: predicts sample type (kick, snare, pad, etc.)
  - Embedding head: 2-layer projection (512 -> 256 -> 128) with L2-normalized output

Training uses a combined loss:
  - Cross-entropy for classification
  - Supervised contrastive loss (SupCon) for embeddings
"""

from __future__ import annotations

import torch
import torch.nn as nn

from samplespace.ml.dataset import NUM_CLASSES

CNN_EMBEDDING_DIM = 128


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention.

    Learns to re-weight channels by modelling inter-channel dependencies.
    Lightweight: only two small FC layers per block.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale: torch.Tensor = self.squeeze(x).view(b, c)
        scale = self.excitation(scale).view(b, c, 1, 1)
        result: torch.Tensor = x * scale
        return result


class ConvBlock(nn.Module):
    """Two-conv residual block with SE attention.

    Structure: (Conv2d -> BN -> ReLU) -> (Conv2d -> BN) -> residual add -> ReLU -> SE -> MaxPool.
    Skip connection uses 1x1 conv when channel dimensions change.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        # Skip connection: identity when dims match, 1x1 conv when they don't
        if in_channels != out_channels:
            self.skip: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv_path(x)
        out = self.relu(out + residual)
        out = self.se(out)
        result: torch.Tensor = self.pool(out)
        return result


class SampleCNN(nn.Module):
    """Dual-head CNN for sample classification and embedding extraction.

    Input: mel spectrogram of shape (batch, 1, n_mels, time_frames)
    Outputs:
        - logits: (batch, num_classes) classification logits
        - embedding: (batch, 128) L2-normalized embedding vector
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        # 4 residual conv blocks: 1 -> 64 -> 128 -> 256 -> 512
        self.features = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Shared backbone output
        self.backbone_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Classification head
        self.classifier = nn.Linear(512, num_classes)

        # 2-layer projection head (SimCLR-style)
        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, CNN_EMBEDDING_DIM),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits, embedding)."""
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.backbone_fc(x)

        logits = self.classifier(x)

        emb = self.embedding(x)
        emb = nn.functional.normalize(emb, p=2, dim=1)  # L2 normalize

        return logits, emb
