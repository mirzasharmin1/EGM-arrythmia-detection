import random

import torch
from torch.utils.data import Dataset


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
        segment = torch.FloatTensor(self.data[idx])
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)

        return segment, label


# Data Preparation
def prepare_data(patient_segments, sinus_segments, shuffle=True):
    # Merge patient and sinus data, create labels, and shuffle

    print("=== Data Preparation ===")

    all_data = []
    all_labels = []

    sinus_count = 0
    for file_key, segments in sinus_segments.items():
        for segment in segments:
            all_data.append(segment)
            all_labels.append(1)  # 1 for sinus label
            sinus_count += 1

    patient_count = 0
    for file_key, segments in patient_segments.items():
        for segment in segments:
            all_data.append(segment)
            all_labels.append(0)  # 0 for patient label
            patient_count += 1

    print(f"Sinus segments: {sinus_count}")
    print(f"Patient segments: {patient_count}")
    print(f"Total segments: {len(all_data)}")

    # Shuffle Data
    if shuffle:
        combined = list(zip(all_data, all_labels))
        random.shuffle(combined)
        all_data, all_labels = zip(*combined)
        all_data = list(all_data)
        all_labels = list(all_labels)

    print(f"Length check - Data: {len(all_data)}, Labels: {len(all_labels)}")
    return all_data, all_labels
