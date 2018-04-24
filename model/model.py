import torch
import torch.nn as nn
from torch.autograd import Variable

from apex import RNN

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(nhid, ntoken)
        self.rnn=getattr(RNN, rnn_type)(ninp, nhid, nlayers, dropout=dropout)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.decoder.bias.data.fill_(0)
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, reset_mask=None):
        emb = self.drop(self.encoder(input))
        self.rnn.detach_hidden()
        output, hidden = self.rnn(emb, reset_mask=reset_mask)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = {}
        sd['encoder'] = self.encoder.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd['rnn'] = self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd = {'encoder': sd}
        sd['decoder'] = self.decoder.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        if 'decoder' in state_dict:
            self.decoder.load_state_dict(state_dict['decoder'], strict=strict)
        self.encoder.load_state_dict(state_dict['encoder']['encoder'], strict=strict)
        self.rnn.load_state_dict(state_dict['encoder']['rnn'], strict=strict)

class RNNFeaturizer(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, all_layers=False):
        super(RNNFeaturizer, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn=getattr(RNN, rnn_type)(ninp, nhid, nlayers, dropout=dropout)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.all_layers = all_layers
        self.output_size = self.nhid if not self.all_layers else self.nhid * self.nlayers

    def forward(self, input, seq_len=None, get_hidden=False):
        self.rnn.detach_hidden()
        if seq_len is None:
            for i in range(input.size(0)):
                emb = self.drop(self.encoder(input[i]))
                _, hidden = self.rnn(emb.unsqueeze(0), collectHidden=True)
            cell = self.get_features(hidden)
            if get_hidden:
                cell = (self.get_features(hidden, get_hidden=True), cell)
        else:
            last_cell = 0
            last_hidden = 0
            for i in range(input.size(0)):
                emb = self.drop(self.encoder(input[i]))
                _, hidden = self.rnn(emb.unsqueeze(0), collectHidden=True)
                cell = self.get_features(hidden)
                if get_hidden:
                    hidden = self.get_features(hidden, get_hidden=True)
                if i > 0:
                    cell = get_valid_outs(i, seq_len, cell, last_cell)
                    if get_hidden:
                        hidden = get_valid_outs(i, seq_len, hidden, last_hidden)
                last_cell = cell
                if get_hidden:
                    last_hidden = hidden
            if get_hidden:
                cell = (hidden, cell)

        return cell

    def get_cell_features(self, hidden, get_hidden=False):
        if not get_hidden:
            cell = hidden[1]
        else:
            cell = hidden[0]
        #get cell state from layers
        if self.all_layers:
            cell = torch.cat(cell, -1)
        else:
            cell = cell[-1]
        return cell[-1]


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = {}
        sd['encoder'] = self.encoder.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd['rnn'] = self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        self.encoder.load_state_dict(state_dict['encoder'], strict=strict)
        self.rnn.load_state_dict(state_dict['rnn'], strict=strict)

def get_valid_outs(timestep, seq_len, out, last_out):
    invalid_steps = timestep >= seq_len
    if (invalid_steps.long().sum() == 0):
        return out
    return selector_circuit(out, last_out, invalid_steps)

def selector_circuit(val0, val1, selections):
    selections = selections.type_as(val0.data).view(-1, 1).contiguous()
    return (val0*(1-selections)) + (val1*selections)
