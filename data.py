import os.path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import utils


MAX_LEN = 85

def get_table_lookup(data_dir='vocab'):
    return utils.load_pkl_data('ind2vec.p', data_dir='vocab')

def get_loaders(batch_size, data_dir='hw2_data', test=False):
    train_ind = utils.load_pkl_data('snli_train_ind.p')
    val_ind = utils.load_pkl_data('snli_val_ind.p')
    train_target = utils.load_pkl_data('snli_train_target.p')
    val_target = utils.load_pkl_data('snli_val_target.p')
    if test:
        train_dataset = SNLI_Dataset(train_ind[:5*batch_size], train_target)
    else:
        train_dataset = SNLI_Dataset(train_ind, train_target)
    val_dataset = SNLI_Dataset(val_ind, val_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader

def get_max_len(train_data, train_val):
    def max_len(data):
        return max([max([len(premise), len(hypothesis)]) for premise, hypothesis in data])
    return max(max_len(train_data), max_len(train_val))

def collate_fn(datum):
    batch_size = len(datum)
    premise_data = np.zeros((batch_size, MAX_LEN), dtype=np.int64)
    hypo_data = np.zeros((batch_size, MAX_LEN), dtype=np.int64)
    premise_lens = []
    hypo_lens = []
    targets = []
    for ix, ([premise, hypothesis], [premise_len, hypo_len], target) in enumerate(datum):
        premise_lens.append(premise_len)
        hypo_lens.append(hypo_len)
        targets.append(target)

        premise_data[ix, :] = np.pad(premise, pad_width=(0, MAX_LEN-premise_len), mode='constant', constant_values=0)
        hypo_data[ix, :] = np.pad(hypothesis, pad_width=(0, MAX_LEN-hypo_len), mode='constant', constant_values=0)

    return (torch.from_numpy(premise_data),
            torch.from_numpy(hypo_data),
            torch.LongTensor(premise_lens),
            torch.LongTensor(hypo_lens),
            torch.LongTensor(targets))

class SNLI_Dataset(Dataset):
    max_len = MAX_LEN
    def __init__(self, data, target):
        self.data = [[premise[:self.max_len], hypo[:self.max_len]] for premise, hypo in data]
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        x = self.data[ix]
        lens = [len(x[0]), len(x[1])]
        target = self.target[ix]
        return x, lens, target


if __name__ == '__main__':
    train_data = utils.load_pkl_data('snli_train_ind.p')
    val_data = utils.load_pkl_data('snli_val_ind.p')
    print(f'Max length sentence: ', get_max_len(train_data, val_data))
