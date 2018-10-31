import os
import pickle as pkl
import matplotlib.pyplot as plt

def save_pkl_data(data, fname, data_dir="hw2_data"):
    with open(os.path.join(data_dir, fname), 'wb') as fout:
        pkl.dump(data, fout)

def load_pkl_data(fname, data_dir="hw2_data"):
    with open(os.path.join(data_dir, fname), 'rb') as fin:
        data = pkl.load(fin)
    return data

def plot_curves(train_loss, val_loss, train_acc, val_acc, path):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(train_loss)
    ax[0].plot(val_loss)
    ax[1].plot(train_acc)
    ax[1].plot(val_acc)

    ax[0].legend(['Train', 'Val'])
    ax[1].legend(['Train', 'Val'])
    ax[0].set_title('Cross entropy Loss')
    ax[1].set_title('Accuracy')
    plt.savefig(path)
