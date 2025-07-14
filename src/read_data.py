import os
import re
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


def read_all_data(base_dir):
    all_patient_directories = get_dirs_in_path(base_dir)

    directory_by_patients = {}
    for dir_path in all_patient_directories:
        last_path_segment = dir_path.split('/')[-1]

        match = re.search(r'Patient\s+(\d+)[\s-]', last_path_segment)
        if match:
            patient_id = match.group(1)
            directory_by_patients[patient_id] = get_dirs_in_path(dir_path)
        else:
            continue

    data_index = pd.read_csv('data/fibResults_export.csv')

    patient_dfs = {}
    sinus_dfs = {}

    for index, row in data_index.iterrows():
        patient_id = row['Patient'].split('_')[1]
        is_fibrillation = row['Classification'] == 'FIBRILLATION'
        timestamp = row['Timestamp'].strip().replace('-', '_')

        current_patient_directories = directory_by_patients[patient_id]

        matching_dir = None
        for current_dir in current_patient_directories:
            if current_dir.endswith(f"/{timestamp}"):
                matching_dir = current_dir

        if matching_dir is not None:
            try:
                ecg_df = read_csv_with_line_count(os.path.join(matching_dir, 'ECG_FILTERED.csv'), column_count=40)
            except FileNotFoundError:
                ecg_df = read_csv_with_line_count(os.path.join(matching_dir, 'ECG_Waveforms_FILTERED.csv'),
                                                  column_count=40)

            try:
                unipolar_df = read_csv_with_line_count(os.path.join(matching_dir, 'EPcathBIO_FILTERED.csv'),
                                                       column_count=401)
                columns_to_take = ['t_dws', 't_secs', 't_usecs', 't_ref', 'c40', 'c41', 'c42', 'c43', 'c44', 'c45',
                                   'c46', 'c47', 'c48', 'c49', 'c50', 'c51', 'c52', 'c53', 'c54', 'c55', 'c60', 'c61',
                                   'c62', 'c63', 'c64', 'c65', 'c66', 'c67', 'c68', 'c69', 'c70', 'c71', 'c72', 'c73',
                                   'c74', 'c75']
                unipolar_df = unipolar_df[columns_to_take]
            except FileNotFoundError:
                unipolar_df = read_csv_with_line_count(
                    os.path.join(matching_dir, 'EP_Catheter_Unipolar_Waveforms_Filtered.csv'), column_count=605)
                columns_to_take = ['t_dws', 't_secs', 't_usecs', 't_ref', 'c30', 'c31', 'c32', 'c33', 'c34', 'c35',
                                   'c36', 'c37', 'c38', 'c39', 'c40', 'c41', 'c42', 'c43', 'c44', 'c45', 'c98', 'c99',
                                   'c100', 'c101', 'c102', 'c103', 'c104', 'c105', 'c106', 'c107', 'c108', 'c109',
                                   'c110', 'c111', 'c112', 'c113']
                unipolar_df = unipolar_df[columns_to_take]

            fixed_columns = unipolar_df.columns[:4].tolist()
            new_columns = [f'c{i + 1}' for i in range(len(unipolar_df.columns) - 4)]
            unipolar_df.columns = fixed_columns + new_columns

            ecg_df = ecg_df.dropna()
            unipolar_df = unipolar_df.dropna()

            if is_fibrillation:
                patient_dfs[matching_dir] = (ecg_df, unipolar_df)
            else:
                sinus_dfs[matching_dir] = (ecg_df, unipolar_df)
        else:
            print(f'Directory for patient {patient_id} not found with timestamp {timestamp}')

    return patient_dfs, sinus_dfs


def merge_ecg_and_unipolar(dataframes_index):
    combined_dataframes = {}

    for file_key, (ecg_df, unipolar_df) in dataframes_index.items():
        combined_df = pd.merge(
            ecg_df,
            unipolar_df,
            on=['t_dws', 't_secs', 't_usecs'],
            how='inner'
        )
        combined_dataframes[file_key] = combined_df

    return combined_dataframes
