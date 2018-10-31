import os
import sys
import torch

import utils
import data
import models
import train_helpers

def main(premise_hidden_size,
         hypo_hidden_size,
         linear_hidden_size,
         interaction_type,
         device,
         kind,
         num_layers=1,
         bidirectional=True,
         kernel_size=3,
         lr=1e-4,
         test=False,
         model_dir='models'):
  valid_types = ('cat', 'element_wise_mult')
  if interaction_type not in valid_types:
    raise ValueError('interaction_type can only be: ', valid_types)

  # data
  batch_size = 32
  save_freq = 500
  max_epochs = 40
  train_loader, val_loader = data.get_loaders(batch_size, test=test)

  # model
  embed_size = 300
  ind2vec = data.get_table_lookup()
  if kind == 'rnn':
    model = models.SNLI_Model(ind2vec,
                              embed_size,
                              premise_hidden_size,
                              hypo_hidden_size,
                              linear_hidden_size,
                              interaction_type,
                              device,
                              kind='rnn',
                              num_layers=num_layers,
                              bidirectional=bidirectional)
  else:
    model = models.SNLI_Model(ind2vec,
                              embed_size,
                              premise_hidden_size,
                              hypo_hidden_size,
                              linear_hidden_size,
                              interaction_type,
                              device,
                              kind='cnn',
                              kernel_size=kernel_size)
  model = model.to(device)
  optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)
  loss_fn = torch.nn.CrossEntropyLoss()

  model_name = f'{kind}_model_{premise_hidden_size}_{interaction_type}'
  model_dir = os.path.join(model_dir, model_name)
  train_helper = train_helpers.TrainHelper(device, model, loss_fn, optimizer, models.batch_params_key, model_dir, test)
  train_loss, val_loss, train_acc, val_acc = train_helper.train_loop(train_loader, val_loader, max_epochs=max_epochs, save_freq=save_freq)

  if 'cpu' in device:
    os.makedirs('figures', exist_ok=True)
    path = f'figures/{model_name}'
    utils.plot_curves(train_loss, val_loss, train_acc, val_acc, path)

  utils.save_pkl_data(train_loss, 'train_loss.p', data_dir=model_dir)
  utils.save_pkl_data(val_loss, 'val_loss.p', data_dir=model_dir)
  utils.save_pkl_data(train_acc, 'train_acc.p', data_dir=model_dir)
  utils.save_pkl_data(val_acc, 'val_acc.p', data_dir=model_dir)

if __name__ == '__main__':
  if len(sys.argv) == 5 or len(sys.argv) == 6:
    test = False if len(sys.argv) == 5 else bool(sys.argv[5])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kind = sys.argv[1]
    model_dir = sys.argv[2]
    premise_hidden_size = int(sys.argv[3])
    hypo_hidden_size = premise_hidden_size
    linear_hidden_size = 80
    interaction_type = sys.argv[4]
    print("Kind            : ", kind)
    print("Device          : ", device)
    print("Hidden size     : ", premise_hidden_size)
    print("Interaction type: ", interaction_type)
    main(premise_hidden_size,
         hypo_hidden_size,
         linear_hidden_size,
         interaction_type,
         device,
         kind,
         test=test,
         model_dir=model_dir)
  else:
    print("Error:")
    print("\tUsage  : python train_rnn.py <kind> <out_model_dir> <hidden_size> <interaction_type>")
    print("\tExample: python train_rnn.py rnn models 150 cat")
