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
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--fp16', action='store_true',
                    help='run in fp16 mode')
parser.add_argument('--neuron', type=int, default=-1,
                    help='''specifies which neuron to analyze for visualization or overwriting.
                         Defaults to maximally weighted neuron during classification steps [not working yet]''')
parser.add_argument('--visualize', action='store_true',
                    help='generates heatmap of main neuron activation [not working yet]')
parser.add_argument('--overwrite', type=float, default=0.0,
                    help='Overwrite value of neuron s.t. generated text reads as a +1/-1 classification [not working yet]')
parser.add_argument('--text', default='',
                    help='warm up generation with specified text first [not working yet]')
args = parser.parse_args()

args.data_size = 256

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

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

model.eval()

hidden = model.rnn.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(args.data_size).long()).cuda()

with open(args.save, 'w') as outf:
    with torch.no_grad():
        for i in range(args.gen_length):
            output, _ = model(input)
            word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
            char_idx = torch.multinomial(word_weights, 1)[0]
            input.data.fill_(char_idx)
            character = chr(char_idx)

            outf.write(character + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.gen_length))
