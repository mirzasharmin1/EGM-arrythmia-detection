import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns


def calculate_cross_correlation_matrix(segment1, segment2):
    num_electrodes = segment1.shape[1]
    correlation_matrix = np.zeros((num_electrodes, num_electrodes))

    # Calculate Pearson correlation for each electrode pair
    for i in range(num_electrodes):
        for j in range(num_electrodes):
            corr_coef, _ = pearsonr(segment1[:, i], segment2[:, j])
            correlation_matrix[i, j] = corr_coef

    return correlation_matrix


def print_correlation_info(correlation_matrix, segment_label="Segment"):
    """
    Print useful statistics about the correlation matrix
    """
    print(f"\n=== {segment_label} Correlation Analysis ===")
    print(f"Matrix shape: {correlation_matrix.shape}")

    # Overall statistics
    print(f"Mean correlation: {np.mean(correlation_matrix):.3f}")
    print(f"Max correlation: {np.max(correlation_matrix):.3f}")
    print(f"Min correlation: {np.min(correlation_matrix):.3f}")
    print(f"Std deviation: {np.std(correlation_matrix):.3f}")

    # Diagonal vs off-diagonal
    diagonal = np.diag(correlation_matrix)
    off_diagonal = correlation_matrix[~np.eye(correlation_matrix.shape[0], dtype=bool)]

    print(f"\nDiagonal (self-correlation): all = 1.000")
    print(f"Off-diagonal statistics:")
    print(f"  Mean: {np.mean(off_diagonal):.3f}")
    print(f"  Max: {np.max(off_diagonal):.3f}")
    print(f"  Min: {np.min(off_diagonal):.3f}")
    print(f"  Std: {np.std(off_diagonal):.3f}")

    # Count strong correlations
    strong_positive = np.sum(off_diagonal > 0.7)
    strong_negative = np.sum(off_diagonal < -0.7)
    weak_correlations = np.sum(np.abs(off_diagonal) < 0.3)

    print(f"\nCorrelation strength distribution:")
    print(f"  Strong positive (>0.7): {strong_positive} pairs")
    print(f"  Strong negative (<-0.7): {strong_negative} pairs")
    print(f"  Weak (|r| < 0.3): {weak_correlations} pairs")
    print(f"  Total electrode pairs: {len(off_diagonal)}")

    # Find highest correlations
    print(f"\nTop 5 electrode correlations:")
    # Get upper triangle indices to avoid duplicates
    triu_indices = np.triu_indices_from(correlation_matrix, k=1)
    triu_values = correlation_matrix[triu_indices]

    # Sort and get top 5
    sorted_indices = np.argsort(np.abs(triu_values))[::-1][:5]

    for idx in sorted_indices:
        i, j = triu_indices[0][idx], triu_indices[1][idx]
        corr_val = correlation_matrix[i, j]
        print(f"  E{i + 1} ↔ E{j + 1}: {corr_val:.3f}")

    print("-" * 50)


def egm_cross_correlation(segment, save_path, segment_label="EGM Segment"):
    # Cross-correlation within the same segment (electrode to electrode)
    correlation_matrix = calculate_cross_correlation_matrix(segment, segment)

    # Simplified plot for single segment - NO annotations for 32x32
    plt.figure(figsize=(10, 8))
    electrode_labels = [f'E{i + 1}' for i in range(segment.shape[1])]

    sns.heatmap(correlation_matrix,
                annot=False,  # Remove annotations for readability
                cmap='RdBu_r',
                center=0,
                square=True,
                xticklabels=electrode_labels,
                yticklabels=electrode_labels,
                cbar_kws={'label': 'Pearson Correlation'})

    plt.title(f'Electrode Cross-Correlation: {segment_label}')
    plt.xlabel('Electrodes')
    plt.ylabel('Electrodes')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return correlation_matrix
