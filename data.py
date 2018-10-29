import numpy as np
import torch
from torch.utils.data import Dataset

import utils


BATCH_SIZE = 32
MAX_LEN = 85

def get_max_len(train_data, train_val):
    def max_len(data):
        return max([max([len(premise), len(hypothesis)]) for premise, hypothesis in data])
    return max(max_len(train_data), max_len(train_val))

def collate_fn(datum):
    premise_data = np.zeros((BATCH_SIZE, MAX_LEN), dtype=np.int64)
    hypo_data = np.zeros((BATCH_SIZE, MAX_LEN), dtype=np.int64)
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
            torch.LongTensor(indices)
            torch.LongTensor(targets))

class SNLIDataset(Dataset):
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
