import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 3
batch_params_key = ['premise', 'hypo', 'premise_len', 'hypo_len', 'targets']

class CNN(nn.Module):
    def __init__(self, lookup_table, embed_size, hidden_size, kernel_size):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(lookup_table)
        self.cnn1 = nn.Conv1d(embed_size, hidden_size, kernel_size)
        self.cnn2 = nn.Conv1d(hidden_size, hidden_size, kernel_size)

    def forward(self, X, X_len):
        embedded = self.embed(X)
        hidden = F.relu(self.cnn1(embedded.transpose(1, 2))) # (B, H, T)
        hidden = F.relu(self.cnn2(hidden)).transpose(1, 2)   # (B, T, H)
        return hidden.max(dim=1)[0]

class GRU(nn.Module):
    def __init__(self, lookup_table, embed_size, hidden_size, num_layers, bidirectional, device):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.embed = nn.Embedding.from_pretrained(lookup_table)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

    def init_state(self, batch_size):
        return torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device)

    def forward(self, X, X_len):
        batch_size = X.size(0)
        sorted_X_len, sorted_indices = X_len.sort(descending=True)
        _, reverse_indices = sorted_indices.sort()
        sorted_X = X[sorted_indices, :]

        embeddings = self.embed(sorted_X) # B, T, E

        h_0 = self.init_state(batch_size)
        packed_embed = nn.utils.rnn.pack_padded_sequence(embeddings, sorted_X_len, batch_first=True)
        _, hidden_state = self.rnn(packed_embed, h_0) #(B, T, D * H), (L * D, B, H)
        hidden_state = torch.cat([hidden_state[ix, :, :] for ix in range(hidden_state.size(0))], dim=1) # (B, D*L*H)

        return hidden_state[reverse_indices, :]

class SNLI_Model(nn.Module):
    def __init__(self, lookup_table, embed_size, premise_hidden_size, hypo_hidden_size, linear_hidden_size, interaction_type, device, kind='rnn', num_layers=None, bidirectional=None, kernel_size=None):
        super().__init__()
        self.kind = kind
        self.num_directions = 2 if bidirectional else 1
        self.interaction_type = interaction_type
        hidden_size = premise_hidden_size + hypo_hidden_size if interaction_type == 'cat' else premise_hidden_size

        if self.kind == 'rnn':
            self.premise_net = GRU(lookup_table, embed_size, premise_hidden_size, num_layers, bidirectional, device)
            self.hypo_net = GRU(lookup_table, embed_size, hypo_hidden_size, num_layers, bidirectional, device)
            hidden_size = num_layers * self.num_directions * hidden_size
        else:
            self.premise_net = CNN(lookup_table, embed_size, premise_hidden_size, kernel_size)
            self.hypo_net = CNN(lookup_table, embed_size, premise_hidden_size, kernel_size)

        self.linear1 = nn.Linear(hidden_size, linear_hidden_size)
        self.linear2 = nn.Linear(linear_hidden_size, NUM_CLASSES)

    def forward(self, premise, hypo, premise_len, hypo_len):
        premise_hidden = self.premise_net(premise, premise_len)
        hypo_hidden = self.hypo_net(hypo, hypo_len)

        if self.interaction_type == 'cat':
            joint = torch.cat([premise_hidden, hypo_hidden], dim=1)
        else:
            joint = premise_hidden * hypo_hidden

        linear_hidden = F.relu(self.linear1(joint))
        return self.linear2(linear_hidden)
