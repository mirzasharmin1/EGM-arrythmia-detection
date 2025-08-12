import torch
from torch import nn


class EGMTransformer(nn.Module):
    def __init__(self, input_dim=32, d_model=64, num_layers=2, n_head=4, dropout=0.2):  # Reduced complexity
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Add input dropout
        self.input_dropout = nn.Dropout(0.2)

        # Smaller projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Simpler positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1600, d_model) * 0.01)  # Smaller init

        self.input_norm = nn.LayerNorm(d_model)

        # SMALLER transformer with MORE dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=256,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers  # Reduced layers
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # MUCH simpler classifier with more dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(d_model, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_dropout(x)  # Add input noise
        x = self.input_projection(x)
        x = x + self.positional_encoding.unsqueeze(0)
        x = self.input_norm(x)
        x = self.transformer_encoder(x)

        x = x.transpose(1, 2)
        x = self.global_pool(x)
        x = x.squeeze(-1)

        output = self.classifier(x)
        return output
