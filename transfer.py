import argparse
import os
import time
import math
import collections
import pickle as pkl

import torch
import torch.nn as nn
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LogisticRegression

from fp16 import FP16_Module, FP16_Optimizer
from apex.reparameterization import apply_weight_norm, remove_weight_norm

import model
from model import DistributedDataParallel as DDP
from configure_data import configure_data

parser = argparse.ArgumentParser(description='PyTorch Sentiment Discovery Transfer Learning')

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
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to run Logistic Regression')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--load_model', type=str,  default='lang_model.pt',
                    help='path to trained world language model')
parser.add_argument('--save_results', type=str,  default='sentiment',
                    help='path to save intermediate and final results of transfer')
parser.add_argument('--fp16', action='store_true',
                    help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--neurons', default=1, type=int,
                    help='number of nenurons to extract as features')
parser.add_argument('--no_test_eval', action='store_true',
                    help='whether to not evaluate the test model (useful when your test set has no labels)')
parser.add_argument('--write_results', default='',
                    help='write results of model on test (or train if none is specified) data to specified filepath [only supported for csv datasets currently]')
parser.add_argument('--use_cached', action='store_true',
                    help='reuse cached featurizations from a previous from last time')
parser.add_argument('--drop_neurons', action='store_true',
                    help='drop top neurons instead of keeping them')
parser.add_argument('--data-path', type=str, default=None,
                    help='path to singleton dataset (used for first-time tweet datasets)')
parser.add_argument('--get-hidden', action='store_true',
                    help='whether to use the hidden state (as opposed to cell state) as features for classifier')

data_config, data_parser = configure_data(parser)
data_parser.set_defaults(split='1.', data='data/binary_sst/train.csv')
data_parser.set_defaults(valid='data/binary_sst/val.csv', test='data/binary_sst/test.csv')
args = parser.parse_args()

data_parser.set_defaults(split='1.', data='data/binary_sst/train.csv')
data_parser.set_defaults(valid='data/binary_sst/val.csv', test='data/binary_sst/test.csv')

args.cuda = torch.cuda.is_available()

if args.seed is not -1:
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

train_data, val_data, test_data = data_config.apply(args)
ntokens = args.data_size
model = model.RNNFeaturizer(args.model, ntokens, args.emsize, args.nhid, args.nlayers, 0.0, args.all_layers)
if args.cuda:
    model.cuda()

if args.fp16:
    model.half()

# load char embedding and recurrent encoder for featurization
with open(args.load_model, 'rb') as f:
    sd = x = torch.load(f)
    if 'encoder' in sd:
        sd = sd['encoder']

try:
    model.load_state_dict(sd)
except:
    # if state dict has weight normalized parameters apply and remove weight norm to model while loading sd
    apply_weight_norm(model.rnn)
    model.load_state_dict(sd)
    remove_weight_norm(model)

def transform(model, text):
    '''
    Apply featurization `model` to extract features from text in data loader.
    Featurization model should return cell state not hidden state.
    `text` data loader should return tuples of ((text, text length), text label)
    Returns labels and features for samples in text.
    '''
    model.eval()
    features = np.array([])
    labels = np.array([])
    first_feature = True

    def get_batch(batch):
        '''
        Process batch and return tuple of (text, text label, text length) long tensors.
        Text is returned in column format with (time, batch) dimensions.
        '''
        (text, timesteps), labels = batch
        text = Variable(text).long()
        timesteps = Variable(timesteps).long()
        labels = Variable(labels).long()
        if args.cuda:
            text, timesteps, labels = text.cuda(), timesteps.cuda(), labels.cuda()
        return text.t(), labels, timesteps-1

    tstart = start = time.time()
    n = 0
    len_ds = len(text)
    # Use no grad context for improving memory footprint/speed of inference
    with torch.no_grad():
        for i, data in enumerate(text):
            text_batch, labels_batch, length_batch = get_batch(data)
            # get batch size and reset hidden state with appropriate batch size
            batch_size = text_batch.size(1)
            n += batch_size
            model.rnn.reset_hidden(batch_size)
            # extract batch of features from text batch
            cell = model(text_batch, length_batch, args.get_hidden)

            if first_feature:
                features = []
                first_feature = False
                labels = []
            labels.append(labels_batch.data.cpu().numpy())
            features.append(cell.data.cpu().numpy())

            num_char = int(length_batch.sum().cpu().numpy())

            end = time.time()
            elapsed_time = end - start
            total_time = end - tstart
            start = end

            s_per_batch = total_time / (i+1)
            timeleft = (len_ds - (i+1)) * s_per_batch
            if elapsed_time == 0:
                ch_per_s = num_char
            else:
                ch_per_s = num_char / (elapsed_time+1e-8)
            print('batch {:5d}/{:5d} | ch/s {:.2E} | time {:.2E} | time left {:.2E}'.format(i, len_ds, ch_per_s, elapsed_time, timeleft))

    if not first_feature:
        features = (np.concatenate(features))
        labels = (np.concatenate(labels).flatten())
    print('%0.3f seconds to transform %d examples' %
                  (time.time() - tstart, n))
    return features, labels

def score_and_predict(model, X, Y):
    '''
    Given a binary classification model, predict output classification for numpy features `X`
    and evaluate accuracy against labels `Y`. Labels should be numpy array of 0s and 1s.
    Returns (accuracy, numpy array of classification probabilities)
    '''
    probs = model.predict_proba(X)[:, 1]
    clf = probs > .5
    accuracy = (np.squeeze(Y) == np.squeeze(clf)).mean()
    return accuracy, probs

def train_logreg(trX, trY, vaX=None, vaY=None, teX=None, teY=None, penalty='l1', max_iter=100,
        C=2**np.arange(-8, 1).astype(np.float), seed=42, model=None, eval_test=True, neurons=None, drop_neurons=False):
    """
    slightly modified version of openai implementation https://github.com/openai/generating-reviews-discovering-sentiment/blob/master/utils.py
    if model is not None it doesn't train the model before scoring, it just scores the model
    """
    # if only integer is provided for C make it iterable so we can loop over
    if not isinstance(C, collections.Iterable):
        C = list([C])
    # extract features for given neuron indices
    if neurons is not None:
        if drop_neurons:
            all_neurons = set(list(range(trX.shape[-1])))
            neurons = set(list(neurons))
            neurons = list(all_neurons - neurons)
        trX = trX[:, neurons]
        if vaX is not None:
            vaX = vaX[:, neurons]
        if teX is not None:
            teX = teX[:, neurons]

    # Cross validation over C
    scores = []
    if model is None:
        for i, c in enumerate(C):
            model = LogisticRegression(C=c, penalty=penalty, max_iter=max_iter, random_state=seed+i)
            model.fit(trX, trY)
            if vaX is not None:
                score = model.score(vaX, vaY)
            else:
                score = model.score(trX, trY)
            scores.append(score)
            del model
        c = C[np.argmax(scores)]
        model = LogisticRegression(C=c, penalty=penalty, max_iter=max_iter, random_state=seed+len(C))
        model.fit(trX, trY)
    else:
        c = model.C
    # predict probabilities and get accuracy of regression model on train, val, test as appropriate
    # also get number of regression weights that are not zero. (number of features used for modeling)
    nnotzero = np.sum(model.coef_ != 0)
    scores = []
    probs = []
    train_score, train_probs = score_and_predict(model, trX, trY)
    scores.append(train_score*100)
    probs.append(train_probs)
    if vaX is None:
        eval_data = trX
        val_score = train_score
        val_probs = train_probs
    else:
        eval_data = vaX
        val_score, val_probs = score_and_predict(model, vaX, vaY)
    scores.append(val_score*100)
    probs.append(val_probs)
    eval_score = val_score
    eval_probs = val_probs
    if teX is not None and teY is not None:
        if eval_test:
            eval_score, eval_probs = score_and_predict(model, teX, teY)
        else:
            eval_probs = model.predict_proba(teX)[:, 1]
    scores.append(eval_score*100)
    probs.append(eval_probs)
    return model, scores, probs, c, nnotzero

def get_top_k_neuron_weights(model, k=1):
    """
    Get's the indices of the top weights based on the l1 norm contributions of the weights
    based off of https://rakeshchada.github.io/Sentiment-Neuron.html interpretation of
    https://arxiv.org/pdf/1704.01444.pdf (Radford et. al)
    Args:
        weights: numpy arraylike of shape `[d,num_classes]`
        k: integer specifying how many rows of weights to select
    Returns:
        k_indices: numpy arraylike of shape `[k]` specifying indices of the top k rows
    """
    weights = model.coef_.T
    weight_penalties = np.squeeze(np.linalg.norm(weights, ord=1, axis=1))
    if k == 1:
        k_indices = np.array([np.argmax(weight_penalties)])
    elif k >= np.log(len(weight_penalties)):
        # runs O(nlogn)
        k_indices = np.argsort(weight_penalties)[-k:][::-1]
    else:
        # runs O(n+klogk)
        k_indices = np.argpartition(weight_penalties, -k)[-k:]
        k_indices = (k_indices[np.argsort(weight_penalties[k_indices])])[::-1]
    return k_indices

def plot_logits(save_root, X, Y_pred, top_neurons):
    """plot logits and save to appropriate experiment directory"""
    save_root = os.path.join(save_root,'logit_vis')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    print('plotting_logits at', save_root)

    for i, n in enumerate(top_neurons):
        plot_logit_and_save(trXt, trY, n, os.path.join(save_root, str(i)+'_'+str(n)))


def plot_logit_and_save(logits, labels, logit_index, name):
    """
    Plots histogram (wrt to what label it is) of logit corresponding to logit_index.
    Saves plotted histogram to name.

    Args:
        logits:
        labels:
        logit_index:
        name:
"""
    logit = logits[:,logit_index]
    plt.title('Distribution of Logit Values')
    plt.ylabel('# of logits per bin')
    plt.xlabel('Logit Value')
    plt.hist(logit[labels < .5], bins=25, alpha=0.5, label='neg')
    plt.hist(logit[labels >= .5], bins=25, alpha=0.5, label='pos')
    plt.legend()
    plt.savefig(name+'.png')
    plt.clf()

def plot_weight_contribs_and_save(coef, name):
    plt.title('Values of Resulting L1 Penalized Weights')
    plt.tick_params(axis='both', which='major')
    coef = normalize(coef)
    plt.plot(range(len(coef[0])), coef.T)
    plt.xlabel('Neuron (Feature) Index')
    plt.ylabel('Neuron (Feature) weight')
    print('saving weight visualization to', name)
    plt.savefig(name)
    plt.clf()

def normalize(coef):
    norm = np.linalg.norm(coef)
    coef = coef/norm
    return coef

save_root = args.load_model
save_root = save_root.replace('.current', '')
save_root = os.path.splitext(save_root)[0]
save_root += '_transfer'
save_root = os.path.join(save_root, args.save_results)
if not os.path.exists(save_root):
    os.makedirs(save_root)
print('writing results to '+save_root)

# featurize train, val, test or use previously cached features if possible
print('transforming train')
if not (os.path.exists(os.path.join(save_root, 'trXt.npy')) and args.use_cached):
    trXt, trY = transform(model, train_data)
    np.save(os.path.join(save_root, 'trXt'), trXt)
    np.save(os.path.join(save_root, 'trY'), trY)
else:
    trXt = np.load(os.path.join(save_root, 'trXt.npy'))
    trY = np.load(os.path.join(save_root, 'trY.npy'))
vaXt, vaY = None, None
if val_data is not None:
    print('transforming validation')
    if not (os.path.exists(os.path.join(save_root, 'vaXt.npy')) and args.use_cached):
        vaXt, vaY = transform(model, val_data)
        np.save(os.path.join(save_root, 'vaXt'), vaXt)
        np.save(os.path.join(save_root, 'vaY'), vaY)
    else:
        vaXt = np.load(os.path.join(save_root, 'vaXt.npy'))
        vaY = np.load(os.path.join(save_root, 'vaY.npy'))
teXt, teY = None, None
if test_data is not None:
    print('transforming test')
    if not (os.path.exists(os.path.join(save_root, 'teXt.npy')) and args.use_cached):
        teXt, teY = transform(model, test_data)
        np.save(os.path.join(save_root, 'teXt'), teXt)
        np.save(os.path.join(save_root, 'teY'), teY)
    else:
        teXt = np.load(os.path.join(save_root, 'teXt.npy'))
        teY = np.load(os.path.join(save_root, 'teY.npy'))

# train logistic regression model of featurized text against labels
start = time.time()
logreg_model, logreg_scores, logreg_probs, c, nnotzero = train_logreg(trXt, trY, vaXt, vaY, teXt, teY, max_iter=args.epochs, eval_test=not args.no_test_eval, seed=args.seed)
end = time.time()
elapsed_time = end - start

with open(os.path.join(save_root, 'all_neurons_score.txt'), 'w') as f:
    f.write(str(logreg_scores))
with open(os.path.join(save_root, 'all_neurons_probs.pkl'), 'wb') as f:
    pkl.dump(logreg_probs, f)
with open(os.path.join(save_root, 'neurons.pkl'), 'wb') as f:
    pkl.dump(logreg_model.coef_, f)

print('all neuron regression took %s seconds'%(str(elapsed_time)))
print(', '.join([str(score) for score in logreg_scores]), 'train, val, test accuracy for all neuron regression')
print(str(c)+' regularization coefficient used')
print(str(nnotzero) + ' features used in all neuron regression\n')

# save a sentiment classification pytorch model
sd = {}
if not args.fp16:
    clf_sd = {'weight': torch.from_numpy(logreg_model.coef_).float(), 'bias': torch.from_numpy(logreg_model.intercept_).float()}
else:
    clf_sd = {'weight': torch.from_numpy(logreg_model.coef_).half(), 'bias': torch.from_numpy(logreg_model.intercept_).half()}
sd['classifier'] = clf_sd
model.float().cpu()
sd['encoder'] = model.state_dict()
with open(os.path.join(save_root, 'classifier.pt'), 'wb') as f:
    torch.save(sd, f)
model.half()
sd['encoder'] = model.state_dict()
with open(os.path.join(save_root, 'classifier.pt.16'), 'wb') as f:
    torch.save(sd, f)

# extract sentiment neuron indices
sentiment_neurons = get_top_k_neuron_weights(logreg_model, args.neurons)
print('using neuron(s) %s as features for regression'%(', '.join([str(neuron) for neuron in list(sentiment_neurons.reshape(-1))])))

# train logistic regression model of features corresponding to sentiment neuron indices against labels
start = time.time()
logreg_neuron_model, logreg_neuron_scores, logreg_neuron_probs, neuron_c, neuron_nnotzero = train_logreg(trXt, trY, vaXt, vaY, teXt, teY, max_iter=args.epochs, eval_test=not args.no_test_eval, seed=args.seed, neurons=sentiment_neurons, drop_neurons=args.drop_neurons)
end = time.time()

if args.drop_neurons:
    with open(os.path.join(save_root, 'dropped_neurons_score.txt'), 'w') as f:
        f.write(str(logreg_neuron_scores))

    with open(os.path.join(save_root, 'dropped_neurons_probs.pkl'), 'wb') as f:
        pkl.dump(logreg_neuron_probs, f)

    print('%d dropped neuron regression took %s seconds'%(args.neurons, str(end-start)))
    print(', '.join([str(score) for score in logreg_neuron_scores]), 'train, val, test accuracy for %d dropped neuron regression'%(args.neurons))
    print(str(neuron_c)+' regularization coefficient used')

    start = time.time()
    logreg_neuron_model, logreg_neuron_scores, logreg_neuron_probs, neuron_c, neuron_nnotzero = train_logreg(trXt, trY, vaXt, vaY, teXt, teY, max_iter=args.epochs, eval_test=not args.no_test_eval, seed=args.seed, neurons=sentiment_neurons)
    end = time.time()

print('%d neuron regression took %s seconds'%(args.neurons, str(end-start)))
print(', '.join([str(score) for score in logreg_neuron_scores]), 'train, val, test accuracy for %d neuron regression'%(args.neurons))
print(str(neuron_c)+' regularization coefficient used')

# log model accuracies, predicted probabilities, and weight/bias of regression model

with open(os.path.join(save_root, 'all_neurons_score.txt'), 'w') as f:
    f.write(str(logreg_scores))

with open(os.path.join(save_root, 'neurons_score.txt'), 'w') as f:
    f.write(str(logreg_neuron_scores))

with open(os.path.join(save_root, 'all_neurons_probs.pkl'), 'wb') as f:
    pkl.dump(logreg_probs, f)

with open(os.path.join(save_root, 'neurons_probs.pkl'), 'wb') as f:
    pkl.dump(logreg_neuron_probs, f)

with open(os.path.join(save_root, 'neurons.pkl'), 'wb') as f:
    pkl.dump(logreg_model.coef_, f)

with open(os.path.join(save_root, 'neuron_bias.pkl'), 'wb') as f:
    pkl.dump(logreg_model.intercept_, f)

#Plot feats
use_feats, use_labels = teXt, teY
if use_feats is None:
    use_feats, use_labels = vaXt, vaY
if use_feats is None:
    use_feats, use_labels = trXt, trY
try:
    plot_logits(save_root, use_feats, use_labels, sentiment_neurons)
except:
    print('no labels to plot logits for')

plot_weight_contribs_and_save(logreg_model.coef_, os.path.join(save_root, 'weight_vis.png'))


print('results successfully written to ' + save_root)
if args.write_results == '':
    exit()

def get_csv_writer(feats, top_neurons, all_proba, neuron_proba):
    """makes a generator to be used in data_utils.datasets.csv_dataset.write()"""
    header = ['prob w/ all', 'prob w/ %d neuron(s)'%(len(top_neurons),)]
    top_feats = feats[:, top_neurons]
    header += ['neuron %s'%(str(x),) for x in top_neurons]

    yield header

    for i, _ in enumerate(top_feats):
        row = []
        row.append(all_proba[i])
        row.append(neuron_proba[i])
        row.extend(list(top_feats[i].reshape(-1)))
        yield row

data, use_feats = test_data, teXt
if use_feats is None:
    data, use_feats = val_data, vaXt
if use_feats is None:
    data, use_feats = train_data, trXt
csv_writer = get_csv_writer(use_feats, sentiment_neurons, logreg_probs[-1], logreg_neuron_probs[-1])
data.dataset.write(csv_writer, path=args.write_results)
