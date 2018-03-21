###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable

from apex.reparameterization import apply_weight_norm, remove_weight_norm

import model

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

# Set the random seed manually for reproducibility.
if args.seed >= 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

#if args.temperature < 1e-3:
#    parser.error("--temperature has to be greater or equal 1e-3")

model = model.RNNModel(args.model, args.data_size, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).cuda()

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
        val, neuron = torch.max(weight[0], 0)
    else:
        val = weight[0][neuron]
    polarity = torch.sign(val)[0]
    neuron = neuron[0]
    return neuron, polarity

def process_hidden(cell, neuron, mask=False, mask_value=1, polarity=1):
    feat = cell.data[-1, :, neuron]
#    rtn_feat = feat.clone()
    if mask:
        feat.fill_(mask_value*polarity)
    return feat[0]

def model_step(model, input, neuron=None, mask=False, mask_value=1, polarity=1):
    out, hidden = model(input)
    if isinstance(hidden, tuple):
        hidden, cell = hidden
    else:
        hidden = cell = hidden
    if neuron is not None:
        feat = process_hidden(cell, neuron, mask, mask_value, polarity)
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
        ch = sample(ch, temperature)
    input.data.fill_(ch)
    chrs = list(text)
    chrs.append(chr(ch))
    return chrs, vals

def generate(gen_length, model, input, temperature, neuron=None, mask=False, overwrite=1, polarity=1):
    chrs = []
    vals = []
    for i in range(gen_length):
        if neuron:
            ch, val = model_step(model, input, neuron, mask, overwrite, polarity)
            vals.append(val)
        else:
            ch = model_step(model, input, neuron, mask, overwrite, polarity)
        ch = sample(ch, temperature)
        input.data.fill_(ch)
        chrs.append(chr(ch))
    return chrs, vals

def make_heatmap(outvals, save=None, polarity=1):
    pass

neuron, polarity = get_neuron_and_polarity(sd, args.neuron)
neuron = neuron if args.visualize or args.overwrite is not None else None
mask = args.overwrite is not None
    
model.eval()

hidden = model.rnn.init_hidden(1)
input = Variable(torch.LongTensor([int(ord('\n'))])).cuda()
input = input.view(1,1).contiguous()
model_step(model, input, neuron, mask, args.overwrite, polarity)
input.data.fill_(int(ord(' ')))
out = model_step(model, input, neuron, mask, args.overwrite, polarity)
if neuron is not None:
    out = out[0]
input.data.fill_(sample(out, args.temperature))

outstr = []
outvals = []
#with open(args.save, 'w') as outf:
with torch.no_grad():
    if args.text != '':
        chrs, vals = process_text(args.text, model, input, args.temperature, neuron, mask, args.overwrite, polarity)
        outstr += chrs
        outvals += vals
    chrs, vals = generate(args.gen_length, model, input, args.temperature, neuron, mask, args.overwrite, polarity)
    outstr += chrs
    outvals += vals
outstr = ''.join(outstr)
print(outstr)
with open(args.save, 'w') as f:
    f.write(outstr)

if args.visualize:
    make_heatmap(outvals, args.save, polarity)
#    for i in range(args.gen_length):
#        output, _ = model(input)
#        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
#        char_idx = torch.multinomial(word_weights, 1)[0]
#        input.data.fill_(char_idx)
#        character = chr(char_idx)
#
#        outf.write(character + ('\n' if i % 20 == 19 else ' '))
#
#        if i % args.log_interval == 0:
#            print('| Generated {}/{} words'.format(i, args.gen_length))
