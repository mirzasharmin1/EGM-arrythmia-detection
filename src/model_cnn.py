import torch.nn as nn


class EGMCNN(nn.Module):
    """
    CNN for EGM data: 1600 time points × 32 electrodes
    Uses 2D convolutions to capture both temporal and inter-electrode patterns
    """

    def __init__(self, dropout=0.3):
        super(EGMCNN, self).__init__()

        # Input: (batch, 1, 1600, 32) - treating as image with time×electrodes

        # Block 1: Capture short temporal patterns across electrodes
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 3), padding=(3, 1))  # 7 time points, 3 electrodes
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))  # Pool only in time dimension

        # Block 2: Medium temporal patterns
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5, 3), padding=(2, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))

        # Block 3: Longer temporal + spatial electrode patterns
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(5, 5), padding=(2, 2))
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))  # Pool both dimensions

        # Block 4: Complex patterns
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        # Global pooling to handle any remaining dimensions
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Input shape: (batch_size, 1600, 32)
        # Add channel dimension: (batch_size, 1, 1600, 32)
        x = x.unsqueeze(1)

        # CNN blocks
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)  # (batch, 64, 800, 32)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)  # (batch, 128, 400, 32)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # (batch, 256, 200, 16)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)  # (batch, 512, 100, 8)

        # Global pooling
        x = self.global_pool(x)  # (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 512)

        # Classification layers
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x
