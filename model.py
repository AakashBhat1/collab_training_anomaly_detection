from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class CNNLSTM(nn.Module):
    """ResNet18 backbone + LSTM head for clip-level action classification."""

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.temporal_head = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, channels, height, width = clips.shape
        flattened = clips.reshape(batch_size * timesteps, channels, height, width)
        frame_features = self.backbone(flattened)
        sequence_features = frame_features.reshape(batch_size, timesteps, -1)
        lstm_output, _ = self.temporal_head(sequence_features)
        return self.classifier(lstm_output[:, -1, :])

