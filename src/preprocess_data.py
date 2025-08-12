import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr


def downsample_df(data_df, original_freq=2000, target_freq=400):
    # Calculate sampling factor
    decimation_factor = original_freq // target_freq

    # Get signal columns
    signal_columns = list(data_df.columns)[4:]

    # First downsample time columns by taking every nth row
    result_df = data_df.iloc[::decimation_factor].copy().reset_index(drop=True)

    # Then downsample each signal column using scipy.signal.decimate
    for col in signal_columns:
        # Use scipy.signal.decimate (handles anti-aliasing automatically)
        downsampled_signal = signal.decimate(data_df[col].values, decimation_factor)

        # Update the column with downsampled data
        result_df[col] = downsampled_signal

    return result_df


def downsample_data(data_index, original_freq=2000, target_freq=400):
    downsampled_data = {}

    for key, unipolar_df in data_index.items():
        downsampled_data[key] = downsample_df(unipolar_df, original_freq, target_freq)

    return downsampled_data


def filter_egm_lowpass(data, fs=400, cutoff=50, order=4):
    # Calculate Nyquist frequency
    nyquist = fs / 2

    # Normalize cutoff frequency
    normal_cutoff = cutoff / nyquist

    # Design the filter
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    # Apply the filter
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def filter_egm_bandpass(data, fs=2000, low_cutoff=30, high_cutoff=200):
    # Calculate Nyquist frequency
    nyquist = fs / 2

    # Normalize cutoff frequencies
    low_norm = low_cutoff / nyquist
    high_norm = high_cutoff / nyquist

    # Design the bandpass filter
    b, a = signal.butter(4, [low_norm, high_norm], btype='band', analog=False)

    # Apply the filter
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def filter_single_dataframe(df, low_cutoff=30,  high_cutoff=200):
    copied_df = df.copy()
    for col in list(df.columns)[1:]:
        copied_df[col] = filter_egm_bandpass(df[col].to_numpy(), low_cutoff=low_cutoff, high_cutoff=high_cutoff)

    return copied_df


def filter_all_dataframes(data_index, low_cutoff=30, high_cutoff=200):
    filtered_dfs = {}

    for key, df in data_index.items():
        filtered_dfs[key] = filter_single_dataframe(df, low_cutoff=low_cutoff, high_cutoff=high_cutoff)

    return filtered_dfs


def normalize_data(dataframes_index):
    results = {}
    for key, df in dataframes_index.items():
        for column in list(df.columns)[1:]:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
        results[key] = df

    return results


def segment_data(dataframes_index, segment_duration_sec):
    data_segments = {}
    sampling_rate = 400
    samples_per_segment = sampling_rate * segment_duration_sec

    print(f"Segmenting into {segment_duration_sec}-second segments ({samples_per_segment} samples each)...")

    for file_key, dataframe in dataframes_index.items():
        total_samples = len(dataframe)
        num_segments = total_samples // samples_per_segment

        # Get signal columns only
        signal_columns = [col for col in dataframe.columns if col.startswith('c')]
        signal_data = dataframe[signal_columns].values

        segments_for_this_file = []

        for segment_idx in range(num_segments):
            start_idx = segment_idx * samples_per_segment
            end_idx = start_idx + samples_per_segment
            segment = signal_data[start_idx:end_idx, :]
            segments_for_this_file.append(segment)

        data_segments[file_key] = segments_for_this_file
        print(f"  {file_key}: {num_segments} segments created")

    return data_segments


def plot_cross_correlation_matrix(correlation_matrix, segment1_label="Segment 1",
                                  segment2_label="Segment 2"):
    num_electrodes = correlation_matrix.shape[0]
    electrode_labels = [f'E{i + 1}' for i in range(num_electrodes)]

    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(correlation_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                xticklabels=[f'{segment2_label}\n{label}' for label in electrode_labels],
                yticklabels=[f'{segment1_label}\n{label}' for label in electrode_labels],
                cbar_kws={'label': 'Pearson Correlation'})

    plt.title(f'Cross-Correlation Matrix: {segment1_label} vs {segment2_label}')
    plt.xlabel(f'{segment2_label} Electrodes')
    plt.ylabel(f'{segment1_label} Electrodes')
    plt.tight_layout()
    plt.show()


def egm_cross_correlation(segment, segment_label="EGM Segment"):
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

    def calculate_cross_correlation_matrix(segment1, segment2):
        num_electrodes = segment1.shape[1]
        correlation_matrix = np.zeros((num_electrodes, num_electrodes))

        # Calculate Pearson correlation for each electrode pair
        for i in range(num_electrodes):
            for j in range(num_electrodes):
                corr_coef, _ = pearsonr(segment1[:, i], segment2[:, j])
                correlation_matrix[i, j] = corr_coef

        return correlation_matrix

    def plot_cross_correlation_matrix(correlation_matrix, segment1_label="Segment 1",
                                      segment2_label="Segment 2"):
        num_electrodes = correlation_matrix.shape[0]
        electrode_labels = [f'E{i + 1}' for i in range(num_electrodes)]

        plt.figure(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(correlation_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    xticklabels=[f'{segment2_label}\n{label}' for label in electrode_labels],
                    yticklabels=[f'{segment1_label}\n{label}' for label in electrode_labels],
                    cbar_kws={'label': 'Pearson Correlation'})

        plt.title(f'Cross-Correlation Matrix: {segment1_label} vs {segment2_label}')
        plt.xlabel(f'{segment2_label} Electrodes')
        plt.ylabel(f'{segment1_label} Electrodes')
        plt.tight_layout()
        plt.show()

    def egm_cross_correlation(segment, segment_label="EGM Segment"):
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
        plt.show()

        return correlation_matrix

    plt.title(f'Electrode Cross-Correlation: {segment_label}')
    plt.xlabel('Electrodes')
    plt.ylabel('Electrodes')
    plt.tight_layout()
    plt.show()

    return correlation_matrix
