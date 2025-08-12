import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader

from src.dataset import EGMDataset, oversample_with_smote
from src.model_cnn import EGMCNN
from src.model_transformer import EGMTransformer


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


def train_model(model, train_loader, test_loader, num_epochs, device='cpu', patience=5):
    # Training function with early stopping
    print(f"\n=== Training Setup ===")
    print(f"Device: {device}")
    print(f"Early stopping patience: {patience} epochs")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

    model.to(device)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for segments, labels in train_loader:
            segments = segments.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(segments).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += labels.size(0)
            train_predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (train_predicted == labels).sum().item()

        # Validation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for segments, labels in test_loader:
                segments = segments.to(device)
                labels = labels.float().to(device)

                outputs = model(segments).squeeze()
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                test_predicted = (torch.sigmoid(outputs) > 0.5).float()
                test_total += labels.size(0)
                test_correct += (test_predicted == labels).sum().item()

        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total

        # Check early stopping
        if early_stopping(avg_test_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            break

        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}]:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'stopped_epoch': len(train_losses)
    }


def predict_and_evaluate(model, data, labels, device='cpu', batch_size=32):
    print(f"Evaluating {len(data)} samples...")

    # Create dataset and loader
    dataset = EGMDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []
    total_loss = 0.0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for segments, batch_labels in data_loader:
            segments = segments.to(device)
            batch_labels = batch_labels.float().to(device)

            # Forward pass
            outputs = model(segments).squeeze()
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()

            # Get probabilities and predictions
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

            # Store results
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(batch_labels.cpu().numpy().flatten())
            all_probabilities.extend(probabilities.cpu().numpy().flatten())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    avg_loss = total_loss / len(data_loader)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Print results
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Return results
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'loss': avg_loss,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities
    }


def create_model(model_type):
    if model_type == 'transformer':
        model = EGMTransformer(
            input_dim=32,
            d_model=128,
            num_layers=3,
            n_head=8,
            dropout=0.2
        )
    else:
        model = EGMCNN(dropout=0.2)

    return model


def run_training_pipeline(train_data, train_labels, valid_data, valid_labels, device, model_type, use_smote=True, patience=5):
    print("Starting EGM Classification Pipeline")

    if use_smote:
        train_data_balanced, train_labels_balanced = oversample_with_smote(train_data, train_labels)
    else:
        train_data_balanced, train_labels_balanced = train_data, train_labels

    # Print class distribution when not using SMOTE
    sinus_count = train_labels.count(0)
    patient_count = train_labels.count(1)
    print(f"Training data distribution:")
    print(f"  Sinus: {sinus_count} segments ({sinus_count / len(train_labels) * 100:.1f}%)")
    print(f"  Patient: {patient_count} segments ({patient_count / len(train_labels) * 100:.1f}%)")

    # Create datasets and dataloaders
    train_dataset = EGMDataset(train_data_balanced, train_labels_balanced)
    valid_dataset = EGMDataset(valid_data, valid_labels)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nDataLoaders created:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(valid_loader)}")

    # Create model
    model = create_model(model_type)

    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

    history = train_model(model, train_loader, valid_loader, num_epochs=20, device=device, patience=patience)

    return model, history
