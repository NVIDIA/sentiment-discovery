import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from itertools import chain
from model import RNNFeaturizer, TransformerFeaturizer
from .transformer_utils import GeLU

class BinaryClassifier(nn.Module):
    def __init__(self, num_features=4096):
        super().__init__()

        self.dense0 = nn.Linear(num_features, 1)
        self.neurons = None
        self.thresholds = [.5]
        self.final = 1
        print('init BinaryClassifier with %d features' % num_features)

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

    def get_thresholds(self):
        return self.thresholds

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = self.dense0.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd['neurons'] = self.neurons
        sd['thresholds'] = self.thresholds
        return sd

    def load_state_dict(self, state_dict, strict=True):
        if 'neurons' in state_dict:
            self.neurons = state_dict['neurons']

        if 'thresholds' in state_dict:
            self.thresholds = state_dict['thresholds']

        sd = {}
        for k, v in state_dict.items():
            if k != 'neurons' and k != 'thresholds':
                sd[k] = v

        self.dense0.load_state_dict(sd, strict=strict)

NONLINEARITY_MAP = {
    'prelu': nn.PReLU,
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'selu': nn.SELU,
    'leaky': nn.LeakyReLU,
    'gelu': GeLU
}

class MultiLayerBinaryClassifier(nn.Module):
    def __init__(self, input_layer_size, layer_sizes, dropout=0.1, init_dropout=True, heads_per_class=1, nonlinearity='PReLU', softmax=False):
        super().__init__()
        self.heads_per_class = heads_per_class
        if self.heads_per_class > 1:
            print('Using multiple heads per class: %d' % heads_per_class)
            layer_sizes = list(layer_sizes)
            layer_sizes[-1] = int(layer_sizes[-1]) * heads_per_class
        self.neurons = None
        self.layer_sizes = [input_layer_size] + list(map(int, layer_sizes))
        self.final = self.layer_sizes[-1]
        self.dropout = dropout
        assert nonlinearity.lower() in NONLINEARITY_MAP
        self.nonlinearity = NONLINEARITY_MAP[nonlinearity.lower()]()
        # layer_sizes are sizes of the input and hidden layers, so the final 1 is assumed.
        layer_list = []
        # Since we recieve input from the Transformer bottleneck... it may make sense to dropout on that input first
        if init_dropout:
            layer_list.extend([nn.Dropout(p=self.dropout)])
        layer_list.extend(list(chain.from_iterable(
            [[nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]), self.nonlinearity, nn.Dropout(p=self.dropout)] for i in range(len(self.layer_sizes) - 2)]
        )))
        self.final_layer = nn.Linear(*self.layer_sizes[-2:])
        extend_list = [self.final_layer]
        if not softmax:
            extend_list += [nn.Sigmoid()]
        layer_list.extend(extend_list)

        self.model = nn.Sequential(*layer_list)
        self.neurons = None
        self.softmax = softmax

        print('init MultiLayerBinaryClassifier with layers %s and dropout %s' % (self.layer_sizes, self.dropout))

    def forward(self, X, **kwargs):
        clf_out = self.model(X).float()
        if self.heads_per_class <= 1:
            return clf_out
        clf_out = clf_out.view(clf_out.size(0), -1, self.heads_per_class)
        probs = clf_out
        if self.softmax:
            probs = F.softmax(probs, 1)
        clf_mean = probs.mean(dim=2)
        clf_std = probs.std(dim=2)
        return clf_out, clf_mean, clf_std

    # HACK -- parameter to measure *variation* between last layer of the MLP.
    # Why? To support multihead -- for the same category, where we want multiple heads to predict with different functions
    # (similar to training a mixture of models) -- useful for uncertainty sampling
    def get_last_layer_variance(self):
        final_layer_weight = self.final_layer.weight
        fl_norm = torch.norm(final_layer_weight,2,1)
        final_layer_weight = final_layer_weight * (1.0 / fl_norm).unsqueeze(1)
        final_layer_dot = final_layer_weight @ torch.transpose(final_layer_weight, 0, 1)
        # Compute matrix of all NxN layers -- in normalized form
        final_layer_dot_loss = (torch.norm(final_layer_dot,2,1) - 1.)
        final_layer_dot_loss = final_layer_dot_loss/(self.final_layer.weight.shape[0]+0.00001)
        final_layer_dot_loss = torch.sum(final_layer_dot_loss)
        # Return the average loss -- per dot comparison.
        return final_layer_dot_loss 

    def get_neurons(self, *args, **kwargs):
        return None

    def set_neurons(self, *args, **kwargs):
        return None

class MultiHeadBCELoss(torch.nn.BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean', heads_per_class=1):
        super(MultiHeadBCELoss, self).__init__(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction)
        self.heads_per_class = heads_per_class

    def forward(self, input, target):
        input = input.permute(0, 2, 1)
        target = target.unsqueeze(1).expand(-1, self.heads_per_class, -1)
        return super(MultiHeadBCELoss, self).forward(input, target)

class MultiHeadCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='elementwise_mean'):
        super(MultiHeadCrossEntropyLoss, self).__init__(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction, ignore_index=ignore_index)
        self.heads_per_class = heads_per_class

    def forward(self, input, target):
        input = input.permute(0, 2, 1)
        target = target.unsqueeze(1).expand(-1, self.heads_per_class)
        return super(MultiHeadCrossEntropyLoss, self).forward(input, target)

class SentimentClassifier(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, model_type, ntoken, ninp, nhid, nlayers, classifier_hidden_layers=None, dropout=0.5, all_layers=False, concat_pools=[False] * 3, get_lm_out=False, args=None):
        super().__init__()
        self.model_type = model_type
        if model_type == 'transformer':
            self.encoder = TransformerFeaturizer(get_lm_out, args)
            out_size = args.decoder_embed_dim
        else:
            # NOTE: Dropout is for Classifier. Add separate RNN dropout or via params, if needed.
            self.encoder = RNNFeaturizer(model_type, ntoken, ninp, nhid, nlayers, dropout=0.0, all_layers=all_layers,
                                         concat_pools=concat_pools, get_lm_out=get_lm_out, hidden_warmup=args.num_hidden_warmup > 0)
            out_size = self.encoder.output_size
        self.encoder_dim = out_size

        if classifier_hidden_layers is None:
            self.classifier = BinaryClassifier(num_features=self.encoder_dim)
        else:
            self.classifier = MultiLayerBinaryClassifier(self.encoder_dim, classifier_hidden_layers, dropout=dropout, heads_per_class=args.heads_per_class, softmax=args.use_softmax)
        self.out_dim = self.classifier.final
        self.neurons_ = None
        self.thresholds_ = self.classifier.thresholds
        # If we want to output multiple heads, and average the output
        self.heads_per_class = args.heads_per_class

    def forward(self, input, seq_len=None, get_hidden=False):
        hidden, lm_out = self.encoder(input, seq_len, get_hidden)
        if get_hidden:
            hidden = hidden[0]
        if self.neurons is not None:
            hidden = hidden[:, self.neurons].contiguous()
        classifier_in = hidden
        class_out = self.classifier(classifier_in)

        return class_out, (lm_out, classifier_in)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = {}
        sd['encoder'] = self.encoder.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd['classifier'] = self.classifier.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        self.encoder.load_state_dict(state_dict['encoder'], strict=strict)
        self.classifier.load_state_dict(state_dict['classifier'], strict=strict)
        self.neurons = self.classifier.neurons
        self.thresholds_ = self.classifier.thresholds

    def get_thresholds(self):
        return self.classifier.get_thresholds()

    def set_thresholds(self, thresholds, dual_threshold=False):
        rtn = self.classifier.set_thresholds(thresholds, dual_threshold=dual_threshold)
        self.thresholds_ = self.classifier.thresholds
        return rtn

    def get_neurons(self, **kwargs):
        return self.classifier.get_neurons(**kwargs)

    def set_neurons(self, num_neurons=None):
        rtn = self.classifier.set_neurons(num_neurons=num_neurons)
        self.neurons_ = self.classifier.neurons
        return rtn

    @property
    def thresholds(self):
        return self.thresholds_

    @thresholds.setter
    def thresholds(self, val):
        self.thresholds_ = val
        self.classifier.thresholds = val

    @property
    def neurons(self):
        return self.neurons_

    @neurons.setter
    def neurons(self, val):
        self.neurons_ = val
        self.classifier.neurons = val
