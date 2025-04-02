import os
import pickle


def mkdirifnotexist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_dirpath(path, level=1):
    dirpath = path
    for i in range(level):
        dirpath = os.path.dirname(dirpath)
    return dirpath



def get_dataset_dir(dataset='vfips'):
    if dataset == 'vfips':
        datadir = 'VFIPS_DATASET_PATH'
    elif dataset == 'bvivfi':
        datadir = 'BVI-VFI_DATASET_PATH'
    else:
        raise ValueError
    return datadir


def save_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def writelist(filepath, src):
    with open(filepath, 'w') as f:
        for item in src:
            f.write('%s\n'%item)


def readlist(filepath):
    with open(filepath) as f:
        rst = f.read().splitlines()
    return rst


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

