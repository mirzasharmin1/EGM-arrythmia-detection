# Transformer Model for EGM Classification
import torch
from torch import nn


class EGMTransformer(nn.Module):
    def __init__(self, input_dim=32, d_model=128, num_layers=3, n_head=8, num_classes=1):
        super(EGMTransformer, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1600, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True

        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.positional_encoding.unsqueeze(0)
        x = self.transformer_encoder(x)

        # Global Pooling
        x = x.transpose(1, 2)
        x = self.global_pool(x)
        x = x.squeeze(-1)

        # Classification
        output = self.classifier(x)

        return output
