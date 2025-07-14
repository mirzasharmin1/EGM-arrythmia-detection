# Training Function for Transformer Model
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader

from src.dataset import prepare_data, EGMDataset
from src.model import EGMTransformer


def train_model(model, train_loader, test_loader, num_epochs=50, device='cuda'):
    # Calculate class weight for imbalance dataset
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy().flatten())

    sinus_count = train_labels.count(1)
    patient_count = train_labels.count(0)
    total = len(train_labels)

    pos_weight = torch.tensor([patient_count / sinus_count]).to(device)

    print(f"\n=== Training Setup ===")
    print(f"Device: {device}")
    print(f"Class weights - Sinus: {sinus_count}, Patient: {patient_count}")
    print(f"Positive weight for BCEWithLogitsLoss: {pos_weight.item():.3f}")

    # Loss function for imbalanced dataset

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.to(device)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # Training Loop

    for epoch in range(num_epochs):
        model.train(),
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for segments, labels in train_loader:
            segments = segments.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            outputs = model(segments)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_total += labels.size(0)
            train_predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (train_predicted == labels).sum().item()

        # validation

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for segments, labels in test_loader:
                segments = segments.to(device)
                labels = labels.float().to(device)

                outputs = model(segments)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

                # Accuracy
                test_predicted = (torch.sigmoid(outputs) > 0.5).float()
                test_total += labels.size(0)
                test_correct += (test_predicted == labels).sum().item()

        scheduler.step()

        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total

        print(f'Epoch [{epoch + 1}/{num_epochs}]:')
        print(f'  Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss: {test_loss / len(test_loader):.4f}, Test Acc: {test_acc:.2f}%')
        print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')

        train_losses.append(train_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }


def run_training_pipeline(patient_segments, sinus_segments):
    print("Starting EGM Classification Pipeline")

    # prepare Data

    all_data, all_labels = prepare_data(patient_segments, sinus_segments)

    # Train Test Split
    train_data, test_data, train_labels, test_labels = train_test_split(
        all_data,
        all_labels,
        train_size=0.7,
        stratify=all_labels,
        random_state=42
    )

    # Datasets

    train_dataset = EGMDataset(train_data, train_labels)
    test_dataset = EGMDataset(test_data, test_labels)

    # Data Loader

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nDataLoaders created:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    model = EGMTransformer(
        input_dim=32,  # Number of catheters
        d_model=128,  # Model dimension
        n_head=8,  # Number of attention heads
        num_layers=3,  # Number of transformer layers
        num_classes=1  # Binary classification
    )

    print(f"\nModel created:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history = train_model(model, train_loader, test_loader, num_epochs=50, device=device)

    return model, history
