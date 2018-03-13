import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from model import RNNFeaturizer

class BinaryClassifier(nn.Module):
    def __init__(self, num_features=4096):
        super().__init__()

        self.dense0 = nn.Linear(num_features, 1)
        self.neurons = None

    def forward(self, X, **kwargs):
        return torch.sigmoid(self.linear(X)).float()
        #return F.sigmoid(self.linear(X), dim=-1).float()

    def linear(self, X):
        weight = self.dense0.weight
        if self.neurons is not None:
            #weight = weight[torch.arange(weight.size(0)).unsqueeze(1), self.neurons].contiguous()
            weight = weight[:, self.neurons].contiguous()
            if X.size(-1) == self.dense0.weight.size(-1):
                X = X[:, self.neurons].contiguous()
            torch.cuda.synchronize()
        return F.linear(X, weight, self.dense0.bias) 

    def set_neurons(self, num_neurons=None):
        if num_neurons is None:
            self.neurons = None
            return self.get_neurons()
        neurons, values = self.get_neurons(num_neurons=num_neurons)
        self.neurons = neurons
        return neurons, values

    def get_neurons(self, num_neurons=None):
        if num_neurons is None:
            return self.dense0.weight
        values, neurons = torch.topk(self.dense0.weight.abs().float(), num_neurons, 1)
        neurons = neurons[0]
        values = self.dense0.weight[:, neurons]
        return neurons, values

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = self.dense0.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd['neurons'] = self.neurons
        return sd

    def load_state_dict(self, state_dict, strict=True):
        if 'neurons' in state_dict:
            self.neurons = state_dict['neurons']

        sd = {}
        for k, v in state_dict.items():
            if k != 'neurons':
                sd[k] = v

        self.dense0.load_state_dict(sd, strict=strict)

class SentimentClassifier(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, all_layers=False):
        super().__init__()
        self.encoder = RNNFeaturizer(rnn_type, ntoken, ninp, nhid, nlayers, dropout=dropout, all_layers=all_layers)
        self.classifier = BinaryClassifier(num_features=self.encoder.output_size)
        
        self.neurons_ = None

    def forward(self, input, seq_len=None):
        self.encoder.rnn.reset_hidden(input.size(1))
        hidden = self.encoder(input, seq_len=seq_len)
        if self.neurons is not None:
            hidden = hidden[:, self.neurons].contiguous()
        return self.classifier(hidden)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = {}
        sd['encoder'] = self.encoder.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd['classifier'] = self.classifier.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        self.encoder.load_state_dict(state_dict['encoder'], strict=strict)
        self.classifier.load_state_dict(state_dict['classifier'], strict=strict)
        self.neurons = self.classifier.neurons

    def get_neurons(self, **kwargs):
        return self.classifier.get_neurons(**kwargs)

    def set_neurons(self, num_neurons=None):
        rtn = self.classifier.set_neurons(num_neurons=num_neurons)
        self.neurons_ = self.classifier.neurons
        return rtn

    @property
    def neurons(self):
        return self.neurons_

    @neurons.setter
    def neurons(self, val):
        self.neurons_ = val
        self.classifier.neurons = val
