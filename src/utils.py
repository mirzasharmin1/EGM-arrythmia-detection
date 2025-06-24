import os


def get_files_in_path(dir_path):
    all_files = []

    for entry in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry)
        if not os.path.isdir(full_path):
            all_files.append(full_path)

    return all_files


def get_dirs_in_path(dir_path):
    all_files = []

    for entry in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry)
        if os.path.isdir(full_path):
            all_files.append(full_path)

    return all_files
