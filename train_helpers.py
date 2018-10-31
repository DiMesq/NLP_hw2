import os
from shutil import copyfile
import time
from collections import defaultdict
import torch
import utils
import data


def compute_correct(out, targets):
    return (out.max(dim=1)[1] == targets).sum().item()

class TrainHelper():

    def __init__(self, device, model, loss_fn, optimizer, batch_params_key, model_dir='model', test=False):
        self.device = device
        self.model = model
        self.model_dir = model_dir
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_params_key = batch_params_key
        self.test = test

        self.model = model
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

    def evaluate(self, loader, accuracy=False):
        self.model.eval()
        loss = 0
        outs = []
        targets = []
        for ix, batch in enumerate(loader):
            batch = [param.to(self.device) for param in batch]
            target = batch[-1]
            outs.append(self.model(*batch[:-1]))
            targets.append(target)

        targets = torch.cat(targets, dim=0)
        outs = torch.cat(outs, dim=0)

        num_examples = outs.size(0)
        if accuracy:
            correct = compute_correct(outs, targets)
            loss = 100 * correct / num_examples
        else:
            loss = self.loss_fn(outs, targets).item()

        return loss

    def train(self, batch, targets):
        batch = {k: param.to(self.device) for k, param in batch.items()}
        targets = targets.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(**batch)
        loss = self.loss_fn(out, targets)
        loss.backward()
        self.optimizer.step()
        return out, loss.item()

    def model_name(self, epoch, ix):
        return f'epoch_{epoch}_batch_{ix}.pt'

    def model_path(self, epoch, ix):
        fname = self.model_name(epoch, ix)
        return os.path.join(self.model_dir, fname)

    def save_model(self, epoch, ix):
        path = self.model_path(epoch, ix)
        torch.save(self.model.to('cpu').state_dict(), path)
        self.model = self.model.to(self.device)

    def save_best_model(self, best_model):
        src_path = self.model_path(best_model['epoch'], best_model['batch_ix'])
        basename = os.path.basename(src_path)
        dst_path = os.path.join(self.model_dir, f'best_{basename}')
        copyfile(src_path, dst_path)

    def save_metrics(self, d):
        for k, v in d.items():
            utils.save_pkl_data(v, f'{k}.p', data_dir=self.model_dir)

    def run_update(self, val_loader, best_model, state, to_return, epoch, ix):
        train_acc, val_acc, val_loss, time_elapsed = self.get_metrics(state, val_loader)
        running_loss = state['running_loss'] / self.save_freq

        self.save_model(epoch, ix)
        to_return['train_loss'].append(running_loss)
        to_return['val_loss'].append(val_loss)
        to_return['train_acc'].append(train_acc)
        to_return['val_acc'].append(val_acc)
        if val_acc > best_model['acc']:
            best_model['acc'] = val_acc
            best_model['epoch'] = epoch
            best_model['batch_ix'] = ix
        self.print_update(epoch, ix, time_elapsed, running_loss, train_acc, val_acc, val_loss, best_model)
        self.save_metrics(to_return)

    def get_metrics(self, state, val_loader):
        train_acc = 100 * state['running_correct'] / state['running_n']
        val_acc = self.evaluate(val_loader, accuracy=True)
        val_loss = self.evaluate(val_loader)
        time_elapsed = (time.time() - state['t0']) / self.save_freq
        return train_acc, val_acc, val_loss, time_elapsed

    def print_update(self, epoch, batch_ix, time_elapsed, running_loss, train_acc, val_acc, val_loss, best_model):
        print(f'Epoch [{epoch}/{self.max_epochs}]; Batch [{batch_ix}/{self.n_batches}]:')
        print(f'\tTrain loss  : {running_loss:.3f}')
        print(f'\tVal loss    : {val_loss:.3f}')
        print(f'\tTrain acc   : {train_acc:.3f}')
        print(f'\tVal acc     : {val_acc}')
        print(f'\tBest val acc: {best_model["acc"]}')
        print(f'\tBest model  : {self.model_name(best_model["epoch"], best_model["batch_ix"])}')
        print(f'\tElapsed     : {(time_elapsed):.2f} s')
        print()

    def init_state(self, state=None):
        if state:
            for k in state: state[k] = time.time() if k == 't0' else 0
        else:
            return {'t0': time.time(), 'running_loss': 0, 'running_correct': 0, 'running_n': 0}

    def update_state(self, params, state):
        batch = {self.batch_params_key[ix]: param for ix, param in enumerate(params[:-1])}
        targets = params[-1]
        out, loss = self.train(batch, targets)

        state['running_loss'] += loss
        state['running_correct'] += compute_correct(out, targets.to(self.device))
        state['running_n'] += params[0].size(0)

    def train_loop(self, train_loader, val_loader, max_epochs, eps=1e-5, save_freq=1000):
        max_epochs = 5 if self.test else max_epochs
        self.max_epochs = max_epochs
        self.n_batches = len(train_loader)
        self.save_freq = 2 if self.test else save_freq

        best_model = {'acc': -1, 'epoch': -1, 'batch_ix': -1}
        n_batches = len(train_loader)
        state = self.init_state()
        to_return = defaultdict(list)
        for epoch in range(max_epochs):
            self.init_state(state)
            for ix, params in enumerate(train_loader):
                self.update_state(params, state)
                if ix % self.save_freq == self.save_freq - 1:
                    self.run_update(val_loader, best_model, state, to_return, epoch+1, ix)
                    self.init_state(state)

        self.save_best_model(best_model)

        return to_return['train_loss'], to_return['val_loss'], to_return['train_acc'], to_return['val_acc']
