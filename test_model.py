import sys
import torch
import utils
import models
import data
import train_helpers


model_dir = sys.argv[1]
hidden_size = int(sys.argv[2])
interaction_type = sys.argv[3]
kind = sys.argv[4]
epoch = sys.argv[5]
batch_ix = sys.argv[6]
batch_size = 32

ind2vec = utils.load_pkl_data('ind2vec.p', data_dir='vocab')
_, val_loader = data.get_loaders(batch_size, data_dir='hw2_data')
loss_fn = torch.nn.CrossEntropyLoss()
fmodel = f'epoch_{epoch}_batch_{batch_ix}.pt'
print('model: ' + fmodel)
model = models.SNLI_Model(ind2vec,
                            300,
                            hidden_size,
                            hidden_size,
                            80,
                            interaction_type,
                            'cpu',
                            kind,
                            num_layers=1,
                            bidirectional=True,
                            kernel_size=3)
model.load_state_dict(torch.load(f'{model_dir}/{fmodel}'))
train_helpers.TrainHelper
helper = train_helpers.TrainHelper('cpu', model, loss_fn, None, models.batch_params_key)
print("\tAcc: ", helper.evaluate(val_loader, accuracy=True))
print("\tLoss: ", helper.evaluate(val_loader, accuracy=False))
print()
