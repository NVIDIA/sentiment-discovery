import argparse
import os
import time
import math
import collections

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from apex.reparameterization import apply_weight_norm, remove_weight_norm

from model import SentimentClassifier
from configure_data import configure_data

parser = argparse.ArgumentParser(description='PyTorch Sentiment Discovery Classification')

parser.add_argument('--model', type=str, default='mLSTM',
                    help='type of recurrent net (RNNTanh, RNNReLU, LSTM, mLSTM, GRU')
parser.add_argument('--emsize', type=int, default=64,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=4096,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--all_layers', action='store_true',
                    help='if more than one layer is used, extract features from all layers, not just the last layer')
parser.add_argument('--load_model', type=str,  default='lang_model_transfer/sentiment/classifier.pt', required=True,
                    help='path to save the classification')
parser.add_argument('--save_probs', type=str,  default='clf_results.npy',
                    help='path to save numpy of predicted probabilities')
parser.add_argument('--write_results', default='',
                    help='write results of model on data to specified filepath (csv/json) [no json support currently]')
parser.add_argument('--fp16', action='store_true',
                    help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--neurons', default=-1, type=int,
                    help='number of nenurons to extract as features')

data_config, data_parser = configure_data(parser)

data_parser.set_defaults(split='1.', data='data/binary_sst/train.csv')

args = parser.parse_args()

args.cuda = torch.cuda.is_available()

train_data, val_data, test_data = data_config.apply(args)
ntokens = args.data_size
model = SentimentClassifier(args.model, ntokens, args.emsize, args.nhid, args.nlayers, 0.0, args.all_layers)
if args.cuda:
    model.cuda()

if args.fp16:
    model.half()

with open(args.load_model, 'rb') as f:
    sd = torch.load(f)

try:
    model.load_state_dict(sd)
except:
    apply_weight_norm(model.encoder.rnn)
    model.load_state_dict(sd)
    remove_weight_norm(model)

if args.neurons > 0:
    model.set_neurons(args.neurons)

# uses similar function as transform from transfer.py
def classify(model, text):
    model.eval()
    labels = np.array([])
    first_label = True

    def get_batch(batch):
        '''
        Process batch and return tuple of (text, text label, text length) long tensors.
        Text is returned in column format with (time, batch) dimensions.
        '''
        text = batch['text']
        timesteps = batch['length']
        labels = batch['label']
        text = Variable(text[0]).long()
        timesteps = Variable(timesteps).long()
        labels = Variable(labels).long()
        if args.cuda:
            text, timesteps, labels = text.cuda(), timesteps.cuda(), labels.cuda()
        return text.t(), labels, timesteps-1

    tstart = start = time.time()
    n = 0
    len_ds = len(text)
    with torch.no_grad():
        for i, data in enumerate(text):
            text_batch, labels_batch, length_batch = get_batch(data)
            size = text_batch.size(1)
            n += size
            # get predicted probabilities given transposed text and lengths of text
            probs = model(text_batch, length_batch)
            if first_label:
                first_label = False
                labels = []
            labels.append(probs[:,-1].data.cpu().numpy())

            num_char = float(length_batch.sum().data.cpu()[0])

            end = time.time()
            elapsed_time = end - start
            total_time = end - tstart
            start = end

            s_per_batch = total_time / (i+1)
            timeleft = (len_ds - (i+1)) * s_per_batch
            ch_per_s = num_char / (elapsed_time)
            print('batch {:5d}/{:5d} | ch/s {:.2E} | time {:.2E} | time left {:.2E}'.format(i+1, len_ds, ch_per_s, elapsed_time, timeleft))

    if not first_label:
        labels = (np.concatenate(labels).flatten())
    print('%0.3f seconds to transform %d examples' %
                  (time.time() - tstart, n))
    return labels

ypred = classify(model, train_data)

save_root = args.load_model
save_root.replace('.current', '')
save_root = os.path.dirname(save_root)
save_root = os.path.join(save_root, args.save_probs)

print('saving predicted probabilities to '+save_root)
np.save(save_root, ypred)

if args.write_results == '':
    exit()

def get_writer(probs):
    header = ['predicted proba']
    yield header
    for prob in probs:
        yield [prob]

print('writing results to '+args.write_results)
writer = get_writer(ypred)
train_data.dataset.write(writer, path=args.write_results)
