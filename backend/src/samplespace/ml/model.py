"""Dual-head CNN for audio sample classification and embedding.

Architecture: 4 conv blocks -> global average pooling -> dual head:
  - Classification head: predicts sample type (kick, snare, pad, etc.)
  - Embedding head: produces 128-dim embedding for similarity search
"""

from __future__ import annotations

import torch
import torch.nn as nn

from samplespace.ml.dataset import NUM_CLASSES

CNN_EMBEDDING_DIM = 128


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.block(x)
        return result


class SampleCNN(nn.Module):
    """Dual-head CNN for sample classification and embedding extraction.

    Input: mel spectrogram of shape (batch, 1, n_mels, time_frames)
    Outputs:
        - logits: (batch, num_classes) classification logits
        - embedding: (batch, 128) normalized embedding vector
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        # 4 conv blocks: 1 -> 32 -> 64 -> 128 -> 256
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Shared backbone output
        self.backbone_fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Classification head
        self.classifier = nn.Linear(256, num_classes)

        # Embedding head
        self.embedding = nn.Sequential(
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
