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

def sorted_alphanum(file_list, return_indices=False):
    """sort the file list by arrange the numbers in filenames in increasing order
    :param file_list: a file list
    :return: sorted file list
    """
    if len(file_list) <= 1:
        return file_list, [0]

    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]

    if return_indices:
        indices = [i[0] for i in sorted(enumerate(file_list), key=lambda x: alphanum_key(x[1]))]
        return sorted(file_list, key=alphanum_key), indices
    else:
        return sorted(file_list, key=alphanum_key)

def alphanum_ordered_file_list(path, ext=''):
    if not os.path.exists(path):
        raise OSError('Path {} not exist!'.format(path))

    file_list = []
    for filename in os.listdir(path):
        file_ext = file_extension(filename)
        if (ext in file_ext or not ext) and os.path.isfile(os.path.join(path, filename)):
            file_list.append(os.path.join(path, filename))
    file_list = sorted_alphanum(file_list)
    return file_list

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
