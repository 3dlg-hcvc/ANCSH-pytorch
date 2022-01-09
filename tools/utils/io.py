import os
import re
import shutil
import json


def file_extension(file_path):
    return os.path.splitext(file_path)[1]


def file_exist(file_path, ext=''):
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return False
    elif ext in file_extension(file_path) or not ext:
        return True
    return False


def folder_exist(folder_path):
    if not os.path.exists(folder_path) or os.path.isfile(folder_path):
        return False
    else:
        return True


def to_abs_path(path, abs_root):
    if not os.path.isabs(path):
        return os.path.join(abs_root, path)
    else:
        return path


def is_non_zero_file(file_path):
    return True if os.path.isfile(file_path) and os.path.getsize(file_path) > 0 else False


def ensure_dir_exists(path):
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def make_clean_folder(path_folder):
    try:
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        else:
            shutil.rmtree(path_folder)
            os.makedirs(path_folder)
    except OSError:
        if not os.path.isdir(path_folder):
            raise


def get_file_list(path, ext='', join_path=False):
    file_list = []
    if not os.path.exists(path):
        return file_list

    for filename in os.listdir(path):
        file_ext = file_extension(filename)
        if (ext in file_ext or not ext) and os.path.isfile(os.path.join(path, filename)):
            if join_path:
                file_list.append(os.path.join(path, filename))
            else:
                file_list.append(filename)
    return file_list


def sorted_alphanum(path_list, return_indices=False):
    """sort the path list by arrange the numbers in path in increasing order
    :param path_list: a path list
    :param return_indices: specify if return indices
    :return: sorted path list
    """
    if len(path_list) <= 1:
        if return_indices:
            return path_list, [0]
        else:
            return path_list

    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    if return_indices:
        indices = [i[0] for i in sorted(enumerate(path_list), key=lambda x: alphanum_key(x[1]))]
        return sorted(path_list, key=alphanum_key), indices
    else:
        return sorted(path_list, key=alphanum_key)


def alphanum_ordered_file_list(path, ext='', join_path=False):
    if not os.path.exists(path):
        raise OSError('Path {} not exist!'.format(path))

    file_list = get_file_list(path, ext, join_path)
    file_list = sorted_alphanum(file_list)
    return file_list


def alphanum_ordered_folder_list(path, join_path=False):
    if not os.path.exists(path):
        raise OSError('Path {} not exist!'.format(path))

    folder_list = []
    for foldername in os.listdir(path):
        if join_path:
            folder_list.append(os.path.join(path, foldername))
        else:
            folder_list.append(foldername)
    folder_list = sorted_alphanum(folder_list)
    return folder_list


def append_parent_path(file_list, parent_path):
    result_paths = []
    for file in file_list:
        result_paths.append(os.path.join(parent_path, file_list))
    return result_paths


def write_json(data, filename, indent=2):
    if folder_exist(os.path.dirname(filename)):
        with open(filename, "w+") as fp:
            json.dump(data, fp, indent=indent)
    if not file_exist(filename):
        raise OSError('Cannot create file {}!'.format(filename))


def read_json(filename):
    if file_exist(filename):
        with open(filename, "r") as fp:
            data = json.load(fp)
        return data
