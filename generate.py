###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import os
import math

import argparse

import torch
from torch.autograd import Variable

from apex.reparameterization import apply_weight_norm, remove_weight_norm

import model

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style({'font.family': 'monospace'})


parser = argparse.ArgumentParser(description='PyTorch Sentiment Discovery Generation/Visualization')

# Model parameters.
parser.add_argument('--model', type=str, default='mLSTM',
                    help='type of recurrent net (RNNTanh, RNNReLU, LSTM, mLSTM, GRU')
parser.add_argument('--emsize', type=int, default=64,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=4096,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--all_layers', action='store_true',
                    help='if more than one layer is used, extract features from all layers, not just the last layer')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--load_model', type=str, default='model.pt',
                    help='model checkpoint to use')
parser.add_argument('--save', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--gen_length', type=int, default='1000',
                    help='number of tokens to generate')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--fp16', action='store_true',
                    help='run in fp16 mode')
parser.add_argument('--neuron', type=int, default=-1,
                    help='''specifies which neuron to analyze for visualization or overwriting.
                         Defaults to maximally weighted neuron during classification steps''')
parser.add_argument('--visualize', action='store_true',
                    help='generates heatmap of main neuron activation [not working yet]')
parser.add_argument('--overwrite', type=float, default=None,
                    help='Overwrite value of neuron s.t. generated text reads as a +1/-1 classification')
parser.add_argument('--text', default='',
                    help='warm up generation with specified text first')
args = parser.parse_args()

args.data_size = 256

args.cuda = torch.cuda.is_available()

# Set the random seed manually for reproducibility.
if args.seed >= 0:
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

#if args.temperature < 1e-3:
#    parser.error("--temperature has to be greater or equal 1e-3")

model = model.RNNModel(args.model, args.data_size, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    model.cuda()

if args.fp16:
    model.half()
with open(args.load_model, 'rb') as f:
    sd = torch.load(f)
try:
    model.load_state_dict(sd)
except:
    apply_weight_norm(model.rnn)
    model.load_state_dict(sd)
    remove_weight_norm(model)

def get_neuron_and_polarity(sd, neuron):
    """return a +/- 1 indicating the polarity of the specified neuron in the module"""
    if neuron == -1:
        neuron = None
    if 'classifier' in sd:
        sd = sd['classifier']
        if 'weight' in sd:
            weight = sd['weight']
        else:
            return neuron, 1
    else:
        return neuron, 1
    if neuron is None:
        val, neuron = torch.max(torch.abs(weight[0].float()), 0)
        neuron = neuron[0]
    val = weight[0][neuron]
    if val >= 0:
        polarity = 1
    else:
        polarity = -1
    return neuron, polarity

def process_hidden(cell, hidden, neuron, mask=False, mask_value=1, polarity=1):
    feat = cell.data[:, neuron]
    rtn_feat = feat.clone()
    if mask:
#        feat.fill_(mask_value*polarity)
        hidden.data[:, neuron].fill_(mask_value*polarity)
    return rtn_feat[0]

def model_step(model, input, neuron=None, mask=False, mask_value=1, polarity=1):
    out, _ = model(input)
    if neuron is not None:
        hidden = model.rnn.rnns[-1].hidden
        if len(hidden) > 1:
            hidden, cell = hidden
        else:
            hidden = cell = hidden
        feat = process_hidden(cell, hidden, neuron, mask, mask_value, polarity)
        return out, feat
    return out

def sample(out, temperature):
    if temperature == 0:
        char_idx = torch.max(out.squeeze().data, 0)[1][0]
    else:
        word_weights = out.float().squeeze().data.div(args.temperature).exp().cpu()
        char_idx = torch.multinomial(word_weights, 1)[0]
    return char_idx

def process_text(text, model, input, temperature, neuron=None, mask=False, overwrite=1, polarity=1):
    chrs = []
    vals = []
    for c in text:
        input.data.fill_(int(ord(c)))
        if neuron:
            ch, val = model_step(model, input, neuron, mask, overwrite, polarity)
            vals.append(val)
        else:
            ch = model_step(model, input, neuron, mask, overwrite, polarity)
#        ch = sample(ch, temperature)
    input.data.fill_(sample(ch, temperature))
    chrs = list(text)
#    chrs.append(chr(ch))
    return chrs, vals

def generate(gen_length, model, input, temperature, neuron=None, mask=False, overwrite=1, polarity=1):
    chrs = []
    vals = []
    for i in range(gen_length):
        chrs.append(chr(input.data[0]))
        if neuron:
            ch, val = model_step(model, input, neuron, mask, overwrite, polarity)
            vals.append(val)
        else:
            ch = model_step(model, input, neuron, mask, overwrite, polarity)
        ch = sample(ch, temperature)
        input.data.fill_(ch)
#        chrs.append(chr(ch))
#    chrs.pop()
    return chrs, vals

def make_heatmap(text, values, save=None, polarity=1):
    cell_height=.325
    cell_width=.15
    n_limit = 74
    text = list(map(lambda x: x.replace('\n', '\\n'), text))
    num_chars = len(text)
    total_chars = math.ceil(num_chars/float(n_limit))*n_limit
    mask = np.array([0]*num_chars + [1]*(total_chars-num_chars))
    text = np.array(text+[' ']*(total_chars-num_chars))
    values = np.array(values+[0]*(total_chars-num_chars))
    values *= polarity

    values = values.reshape(-1, n_limit)
    text = text.reshape(-1, n_limit)
    mask = mask.reshape(-1, n_limit)
    num_rows = len(values)
    plt.figure(figsize=(cell_width*n_limit, cell_height*num_rows))
    hmap=sns.heatmap(values, annot=text, mask=mask, fmt='', vmin=-1, vmax=1, cmap='RdYlGn',
                     xticklabels=False, yticklabels=False, cbar=False)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    # clear plot for next graph since we returned `hmap`
    plt.clf()
    return hmap


neuron, polarity = get_neuron_and_polarity(sd, args.neuron)
neuron = neuron if args.visualize or args.overwrite is not None else None
mask = args.overwrite is not None
    
model.eval()

hidden = model.rnn.init_hidden(1)
input = Variable(torch.LongTensor([int(ord('\n'))]))
if args.cuda:
    input = input.cuda()
input = input.view(1,1).contiguous()
model_step(model, input, neuron, mask, args.overwrite, polarity)
input.data.fill_(int(ord(' ')))
out = model_step(model, input, neuron, mask, args.overwrite, polarity)
if neuron is not None:
    out = out[0]
input.data.fill_(sample(out, args.temperature))

outchrs = []
outvals = []
#with open(args.save, 'w') as outf:
with torch.no_grad():
    if args.text != '':
        chrs, vals = process_text(args.text, model, input, args.temperature, neuron, mask, args.overwrite, polarity)
        outchrs += chrs
        outvals += vals
    chrs, vals = generate(args.gen_length, model, input, args.temperature, neuron, mask, args.overwrite, polarity)
    outchrs += chrs
    outvals += vals
outstr = ''.join(outchrs)
print(outstr)
with open(args.save, 'w') as f:
    f.write(outstr)

if args.visualize:
    make_heatmap(outchrs, outvals, os.path.splitext(args.save)[0]+'.png', polarity)
