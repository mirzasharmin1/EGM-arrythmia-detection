import random

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE


# Dataset class
class EGMDataset(Dataset):
    def __init__(self, data, labels):
        # data List of numpy arrays, each of shape (1600, 32)
        # labels List of integers (0 for sinus, 1 for patient)
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # convert to torch tensors
        segment = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return segment, label


def oversample_with_smote(train_data, train_labels, random_state=42):
    print("=== SMOTE Oversampling ===")

    X = np.array([segment.flatten() for segment in train_data])
    y = np.array(train_labels)

    print(f"Before SMOTE:")
    print(f"  Shape: {X.shape}")
    print(f"  Sinus: {np.sum(y == 0)} segments")
    print(f"  Patient: {np.sum(y == 1)} segments")

    # Apply SMOTE
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print(f"\nAfter SMOTE:")
    print(f"  Shape: {X_resampled.shape}")
    print(f"  Sinus: {np.sum(y_resampled == 0)} segments")
    print(f"  Patient: {np.sum(y_resampled == 1)} segments")

    # Convert back to list of segments
    # Reshape flattened arrays back to original segment shape
    original_shape = train_data[0].shape
    train_data_resampled = [X_resampled[i].reshape(original_shape) for i in range(len(X_resampled))]
    train_labels_resampled = y_resampled.tolist()

    return train_data_resampled, train_labels_resampled


# Data Preparation
def prepare_data(patient_segments, sinus_segments, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, random_state=42):
    patient_files = list(patient_segments.keys())
    sinus_files = list(sinus_segments.keys())

    print(f"Patient files: {len(patient_files)}")
    print(f"Sinus files: {len(sinus_files)}")

    # First split: train vs (valid + test)
    patient_train_files, patient_temp_files = train_test_split(
        patient_files,
        train_size=train_ratio,
        random_state=random_state
    )

    sinus_train_files, sinus_temp_files = train_test_split(
        sinus_files,
        train_size=train_ratio,
        random_state=random_state
    )

    # Second split: valid vs test
    valid_test_ratio = valid_ratio / (valid_ratio + test_ratio)

    patient_valid_files, patient_test_files = train_test_split(
        patient_temp_files,
        train_size=valid_test_ratio,
        random_state=random_state
    )

    sinus_valid_files, sinus_test_files = train_test_split(
        sinus_temp_files,
        train_size=valid_test_ratio,
        random_state=random_state
    )

    print(f"\nFile distribution:")
    print(f"  Train: {len(patient_train_files)} patient + {len(sinus_train_files)} sinus")
    print(f"  Valid: {len(patient_valid_files)} patient + {len(sinus_valid_files)} sinus")
    print(f"  Test:  {len(patient_test_files)} patient + {len(sinus_test_files)} sinus")

    # Collect train data
    train_data = []
    train_labels = []

    for file_key in sinus_train_files:
        segments = sinus_segments[file_key]
        train_data.extend(segments)
        train_labels.extend([0] * len(segments))

    for file_key in patient_train_files:
        segments = patient_segments[file_key]
        train_data.extend(segments)
        train_labels.extend([1] * len(segments))

    # Collect valid data
    valid_data = []
    valid_labels = []

    for file_key in sinus_valid_files:
        segments = sinus_segments[file_key]
        valid_data.extend(segments)
        valid_labels.extend([0] * len(segments))

    for file_key in patient_valid_files:
        segments = patient_segments[file_key]
        valid_data.extend(segments)
        valid_labels.extend([1] * len(segments))

    # Collect test data
    test_data = []
    test_labels = []

    for file_key in sinus_test_files:
        segments = sinus_segments[file_key]
        test_data.extend(segments)
        test_labels.extend([0] * len(segments))

    for file_key in patient_test_files:
        segments = patient_segments[file_key]
        test_data.extend(segments)
        test_labels.extend([1] * len(segments))

    # Shuffle all sets
    random.seed(random_state)

    # Shuffle training data
    train_combined = list(zip(train_data, train_labels))
    random.shuffle(train_combined)
    train_data, train_labels = zip(*train_combined)
    train_data = list(train_data)
    train_labels = list(train_labels)

    # Shuffle validation data
    valid_combined = list(zip(valid_data, valid_labels))
    random.shuffle(valid_combined)
    valid_data, valid_labels = zip(*valid_combined)
    valid_data = list(valid_data)
    valid_labels = list(valid_labels)

    # Shuffle test data
    test_combined = list(zip(test_data, test_labels))
    random.shuffle(test_combined)
    test_data, test_labels = zip(*test_combined)
    test_data = list(test_data)
    test_labels = list(test_labels)

    # Print final statistics
    print(f"\n=== Final Dataset Statistics ===")
    print(f"Training segments: {len(train_data)} ({train_labels.count(0)} sinus, {train_labels.count(1)} patient)")
    print(f"Validation segments: {len(valid_data)} ({valid_labels.count(0)} sinus, {valid_labels.count(1)} patient)")
    print(f"Test segments: {len(test_data)} ({test_labels.count(0)} sinus, {test_labels.count(1)} patient)")

    return train_data, valid_data, test_data, train_labels, valid_labels, test_labels
