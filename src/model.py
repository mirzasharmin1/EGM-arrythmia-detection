import torch
from torch import nn


class EGMTransformer(nn.Module):
    def __init__(self, input_dim=32, d_model=128, num_layers=3, n_head=8, num_classes=1, dropout=0.2):
        super(EGMTransformer, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1600, d_model) * 0.02)

        # Layer normalization for input
        self.input_norm = nn.LayerNorm(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=512,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) = (batch_size, 1600, 32)

        # Input projection
        x = self.input_projection(x)  # (batch_size, 1600, d_model)

        # Add positional encoding
        x = x + self.positional_encoding.unsqueeze(0)  # (batch_size, 1600, d_model)

        # Input normalization
        x = self.input_norm(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch_size, 1600, d_model)

        # Global average pooling
        x = x.transpose(1, 2)  # (batch_size, d_model, 1600)
        x = self.global_pool(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)

        # Classification
        output = self.classifier(x)  # (batch_size, num_classes)

        return output
