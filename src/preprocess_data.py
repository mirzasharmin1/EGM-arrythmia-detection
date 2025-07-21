from scipy import signal


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
