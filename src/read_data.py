import os

import pandas as pd

from src.utils import get_dirs_in_path


def read_csv_with_line_count(file_path, column_count):
    start_row_idx = None
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            num_cols = len(line.strip().split(','))
            if num_cols == column_count:
                start_row_idx = i
                break

    if start_row_idx is None:
        raise ValueError(
            f"No row with {column_count} column found in {file_path} "
        )
    df = pd.read_csv(
        file_path,
        skiprows=start_row_idx,
        index_col=False,
    )
    return df


def process_ecg_df(ecg_df):
    return ecg_df


def process_unipolar_df(unipolar_df):
    return unipolar_df


def read_sinus_data(base_path):
    sinus_data_base_path = os.path.join(base_path, "sinus-data")

    sinus_data_dirs = get_dirs_in_path(sinus_data_base_path)

    results = {}

    for dir_path in sinus_data_dirs:
        ecg_waveform_file_path = os.path.join(dir_path, 'ECG_Waveforms_Filtered.csv')
        unipolar_waveform_file_path = os.path.join(dir_path, 'EP_Catheter_Unipolar_Waveforms_Filtered.csv')

        ecg_waveform_df = read_csv_with_line_count(ecg_waveform_file_path, column_count=40)
        unipolar_waveform_df = read_csv_with_line_count(unipolar_waveform_file_path, column_count=605)

        processed_ecg_df = process_ecg_df(ecg_waveform_df)
        processed_unipolar_df = process_unipolar_df(unipolar_waveform_df)

        results[dir_path] = (processed_ecg_df, processed_unipolar_df)

    return results


def read_patient_data(base_path):
    patient_data_base_path = os.path.join(base_path, "patient-data")
    patient_data_dirs = get_dirs_in_path(patient_data_base_path)

    results = {}

    for first_level_dir_path in patient_data_dirs:
        second_level_dirs = get_dirs_in_path(first_level_dir_path)

        for second_level_dir_path in second_level_dirs:
            ecg_file_path = os.path.join(second_level_dir_path, 'ECG_Waveforms_Filtered.csv')
            unipolar_file_path = os.path.join(second_level_dir_path, 'EP_Catheter_Unipolar_Waveforms_Filtered.csv')

            if not os.path.exists(ecg_file_path):
                print(f'Error: {second_level_dir_path} does not contain a valid ECG file.')
                continue

            if not os.path.exists(unipolar_file_path):
                print(f'Error: {second_level_dir_path} does not contain a valid unipolar file.')
                continue

            ecg_waveform_df = read_csv_with_line_count(ecg_file_path, column_count=40)
            unipolar_waveform_df = read_csv_with_line_count(unipolar_file_path, column_count=605)

            processed_ecg_df = process_ecg_df(ecg_waveform_df)
            processed_unipolar_df = process_unipolar_df(unipolar_waveform_df)

            results[second_level_dir_path] = (processed_ecg_df, processed_unipolar_df)

    return results
