import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .RNN_utils import RNN
from .transformer_utils import Embedding
from .transformer import TransformerDecoder

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

    def forward(self, input, reset_mask=None, chkpt_grad=False, **kwargs):
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

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, all_layers=False, concat_pools=[False] * 3, hidden_warmup=False, residuals=False, get_lm_out=False):
        super(RNNFeaturizer, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn=getattr(RNN, rnn_type)(ninp, nhid, nlayers, dropout=dropout)#, residuals=residuals)
        # self.rnn=getattr(RNN, rnn_type)(ninp, nhid, nlayers, dropout=dropout, residuals=residuals)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.all_layers = all_layers
        self.hidden_warmup = hidden_warmup
        self.aux_lm_loss = get_lm_out
        if self.aux_lm_loss:
            self.decoder = nn.Linear(nhid, ntoken)
        self.concat_max, self.concat_min, self.concat_mean = concat_pools
        self.output_size = self.nhid if not self.all_layers else self.nhid * self.nlayers
        self.output_size *= (1 + sum(concat_pools))

    def forward(self, input, seq_len=None, get_hidden=False, chkpt_grad=False, **kwargs):
        if not self.hidden_warmup:
            self.rnn.reset_hidden(input.size(1))
        if self.aux_lm_loss:
            outs = []
        if seq_len is None:
            for i in range(input.size(0)):
                emb = self.drop(self.encoder(input[i]))
                out, hidden = self.rnn(emb.unsqueeze(0), collect_hidden=True, chkpt_grad=chkpt_grad)
                if self.aux_lm_loss:
                    outs.append(out)
            cell = self.get_features(hidden)
            if self.concat_pools:
                cell = torch.cat((cell, torch.mean(cell, -1), torch.max(cell, -1)))
            if get_hidden:
                cell = (self.get_features(hidden, get_hidden=True), cell)
        else:
            last_cell = last_hidden = 0
            ops = ['max', 'min', 'add']
            maps = {
                k : {'last_c' : 0, 'last_h' : 0, 'c' : None, 'h' : None, 'op' : ops[i]}
                for i, k in enumerate(['concat_max', 'concat_min', 'concat_mean']) if getattr(self, k)
            }
            full_emb = self.drop(self.encoder(input))
            for i in range(input.size(0)):
                emb = full_emb[i]
                out, hidden = self.rnn(emb.unsqueeze(0), collect_hidden=True)
                if self.aux_lm_loss:
                    outs.append(out)
                # print(hidden)                 -> [[[tensor(...)]], [[tensor(...)]]]
                # print(hidden[0][0][0].size()) -> torch.Size([128, 4096])

                cell = self.get_features(hidden)
                if i == 0: # instantiate pools for cell
                    for k, d in maps.items():
                        d['c'] = cell
                if get_hidden:
                    hidden = self.get_features(hidden, get_hidden=True)
                    if i == 0: # instantiate pools for hidden
                        for k, d in maps.items():
                            d['h'] = hidden
                if i > 0:
                    cell = get_valid_outs(i, seq_len, cell, last_cell)
                    for k, d in maps.items():
                        d['c'] = getattr(torch, d['op'])(d['c'], cell)
                        d['c'] = get_valid_outs(i, seq_len, d['c'], d['last_c'])
                    if get_hidden:
                        for k, d in maps.items():
                            d['h'] = getattr(torch, d['op'])(d['h'], hidden)
                            d['h'] = get_valid_outs(i, seq_len, d['h'], d['last_h'])
                        hidden = get_valid_outs(i, seq_len, hidden, last_hidden)
                last_cell = cell
                for k, d in maps.items():
                    d['last_c'] = d['c']
                if get_hidden:
                    last_hidden = hidden
                    for k, d in maps.items():
                        d['last_h'] = d['h']
            # print("Cell dimensions: ", cell.size()) -> torch.Size([128, 4096])
            seq_len = seq_len.view(-1, 1).float()
            if self.concat_mean:
                maps['concat_mean']['c'] /= seq_len
                if get_hidden:
                    maps['concat_mean']['h'] /= seq_len
            for k, d in maps.items():
                cell = torch.cat([cell, d['c']], -1)
                if get_hidden:
                    hidden = torch.cat([hidden, d['h']], -1)
            if get_hidden:
                cell = (hidden, cell)
        if self.aux_lm_loss:
            return cell, self.decoder(torch.cat(outs, 0))
        else:
            return cell, None

    def get_features(self, hidden, get_hidden=False):
        if not get_hidden:
            cell = hidden[1]
        else:
            cell = hidden[0]
        #get cell state from layers
        cell = cell[0]
        if self.all_layers:
            return torch.cat(cell, -1)
        else:
            return cell[-1]

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = {}
        sd['encoder'] = self.encoder.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd['rnn'] = self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd = {'encoder': sd}
        if self.aux_lm_loss:
            sd['decoder'] = self.decoder.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        self.encoder.load_state_dict(state_dict['encoder']['encoder'], strict=strict)
        self.rnn.load_state_dict(state_dict['encoder']['rnn'], strict=strict)
        if self.aux_lm_loss:
            self.decoder.load_state_dict(state_dict['decoder'], strict=strict)

def get_valid_outs(timestep, seq_len, out, last_out):
    invalid_steps = timestep >= seq_len
    if (invalid_steps.long().sum() == 0):
        return out
    return selector_circuit(out, last_out, invalid_steps)

def selector_circuit(val0, val1, selections):
    selections = selections.type_as(val0.data).view(-1, 1).contiguous()
    return (val0*(1-selections)) + (val1*selections)

class TransformerDecoderModel(nn.Module):
    """Base class for encoder-decoder models."""

    def __init__(self, args):
        super().__init__()
        self._is_generation_fast = False
        self.encoder = TransformerDecoder(args, Embedding(args.data_size, args.decoder_embed_dim, padding_idx=args.padding_idx))

    def forward(self, src_tokens, get_attention=True, chkpt_grad=False, **kwargs):
        decoder_out, attn = self.encoder(src_tokens, src_tokens, chkpt_grad=chkpt_grad)
        if get_attention:
            return decoder_out, attn
        return decoder_out

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.encoder.max_positions()

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.encoder.get_normalized_probs(net_output, log_probs, sample)

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.encoder.max_positions()

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters and buffers from state_dict into this module and
        its descendants.

        Overrides the method in nn.Module; compared with that method this
        additionally "upgrades" state_dicts from old checkpoints.
        """
        state_dict = self.upgrade_state_dict(state_dict)
        super().load_state_dict(state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        assert state_dict is not None

        def do_upgrade(m):
            if m != self and hasattr(m, 'upgrade_state_dict'):
                m.upgrade_state_dict(state_dict)

        self.apply(do_upgrade)
        sd = {}
        for k,v in state_dict.items():
            if k.startswith('decoder'):
                k = k.replace('decoder', 'encoder')
            sd[k] = v
        return sd

    def make_generation_fast_(self, **kwargs):
        """Optimize model for faster generation."""
        if self._is_generation_fast:
            return  # only apply once
        self._is_generation_fast = True

        # remove weight norm from all modules in the network
        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(apply_remove_weight_norm)

        def apply_make_generation_fast_(module):
            if module != self and hasattr(module, 'make_generation_fast_'):
                module.make_generation_fast_(**kwargs)

        self.apply(apply_make_generation_fast_)

        def train(mode):
            if mode:
                raise RuntimeError('cannot train after make_generation_fast')

        # this model should no longer be used for training
        self.eval()
        self.train = train

class TransformerFeaturizer(nn.Module):
    def __init__(self, get_lm_out, args):
        super(TransformerFeaturizer, self).__init__()
        args.use_final_embed = True
        self.encoder = TransformerDecoderModel(args)
        self.aux_lm_loss = get_lm_out

    def forward(self, input, seq_len=None, get_hidden=False, chkpt_grad=False, **kwargs):
        encoder_out = self.encoder(input, get_attention=get_hidden, chkpt_grad=chkpt_grad, **kwargs)
        if get_hidden:
            encoder_out = encoder_out[0]
        feats = encoder_out[seq_len.squeeze(), torch.arange(seq_len.size(0))]
        if get_hidden:
            feats = [feats, None]
        lm_out = None
        if self.aux_lm_loss:
            lm_out = F.linear(encoder_out, self.encoder.encoder.embed_out)
        return feats, lm_out

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.encoder.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.encoder.load_state_dict(state_dict, strict=strict)
