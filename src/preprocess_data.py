from scipy import signal
import numpy as np


def downsample_df(data_df, target_freq):
    """
    Downsample DataFrame using pandas operations

    Args:
        data_df: Input DataFrame with time and signal columns
        target_freq: Target frequency (e.g., 400 Hz)

    Returns:
        Downsampled DataFrame
    """

    # Calculate downsampling factor (how many samples to group together)
    factor = 2_000 // target_freq

    # Get signal columns (skip first 4 columns which are time columns)
    signal_columns = list(data_df.columns)[4:]

    # Create group numbers within each second
    # This replaces your manual loop through seconds
    data_df_copy = data_df.copy()
    data_df_copy['group_within_second'] = data_df_copy.groupby('t_secs').cumcount() // factor

    # Create unique group identifier combining second and group number
    # This replaces your second_indexed_data dictionary
    data_df_copy['group_id'] = (data_df_copy['t_secs'].astype(str) + '_' +
                                data_df_copy['group_within_second'].astype(str))

    # Define aggregation rules for different column types
    agg_rules = {}

    # Keep the second value (equivalent to your t_sec in downsampled_row)
    agg_rules['t_secs'] = 'first'

    # Average all signal columns (equivalent to your sum then divide by factor)
    for col in signal_columns:
        agg_rules[col] = 'mean'

    # Group by the unique identifier and apply aggregation
    # This replaces your nested loops and manual averaging
    downsampled_df = data_df_copy.groupby('group_id').agg(agg_rules).reset_index(drop=True)

    # Rename t_secs to t_sec to match your original output
    downsampled_df = downsampled_df.rename(columns={'t_secs': 't_sec'})

    return downsampled_df


def downsample_data(data_index, target_freq):
    downsampled_data = {}

    for key, (_, unipolar_df) in data_index.items():
        downsampled_data[key] = downsample_df(unipolar_df, target_freq)

    return downsampled_data


def filter_egm_lowpass(data, fs=400, cutoff=50, order=4):
    """
    Apply low-pass Butterworth filter to remove high-frequency noise

    Args:
        data: EGM signal data (1D array or pandas Series)
        fs: Sampling frequency (400 Hz)
        cutoff: Cutoff frequency in Hz (50 Hz removes high freq noise)
        order: Filter order (higher = steeper rolloff)

    Returns:
        Filtered signal
    """
    # Calculate Nyquist frequency
    nyquist = fs / 2

    # Normalize cutoff frequency
    normal_cutoff = cutoff / nyquist

    # Design the filter
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    # Apply the filter
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def filter_single_dataframe(df, cutoff=50):
    copied_df = df.copy()
    for col in list(df.columns)[1:]:
        copied_df[col] = filter_egm_lowpass(df[col].to_numpy(), cutoff=cutoff)

    return copied_df


def filter_all_dataframes(data_index, cutoff=50):
    filtered_dfs = {}

    for key, df in data_index.items():
        filtered_dfs[key] = filter_single_dataframe(df, cutoff=cutoff)

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
