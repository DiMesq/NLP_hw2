import os
import pickle as pkl


def save_pkl_data(data, fname, data_dir="hw2_data"):
    with open(os.path.join(data_dir, fname), 'wb') as fout:
        pkl.dump(data, fout)

def load_pkl_data(fname, data_dir="hw2_data"):
    with open(os.path.join(data_dir, fname), 'rb') as fin:
        data = pkl.load(fin)
    return data
