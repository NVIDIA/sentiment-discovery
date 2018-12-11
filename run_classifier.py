import argparse
import os
import time
import math
import collections
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import pandas as pd

from apex.reparameterization import apply_weight_norm, remove_weight_norm

from model import SentimentClassifier
from configure_data import configure_data
from arguments import add_general_args, add_model_args, add_classifier_model_args, add_run_classifier_args

def get_data_and_args():
    parser = argparse.ArgumentParser(description='PyTorch Sentiment Discovery Classification')
    parser = add_general_args(parser)
    parser = add_model_args(parser)
    parser = add_classifier_model_args(parser)
    data_config, data_parser, run_classifier_parser, parser = add_run_classifier_args(parser)
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    args.shuffle=False

    if args.seed is not -1:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    (train_data, val_data, test_data), tokenizer = data_config.apply(args)
    args.data_size = tokenizer.num_tokens
    args.padding_idx = tokenizer.command_name_map['pad'].Id
    return (train_data, val_data, test_data), tokenizer, args

def get_model(args):

    sd = None
    model_args = args
    if args.load is not None and args.load != '':
        sd = torch.load(args.load)
        if 'args' in sd:
            model_args = sd['args']
        if 'sd' in sd:
            sd = sd['sd']

    ntokens = model_args.data_size
    concat_pools = model_args.concat_max, model_args.concat_min, model_args.concat_mean
    if args.model == 'transformer':
        model = SentimentClassifier(model_args.model, ntokens, None, None, None, model_args.classifier_hidden_layers, model_args.classifier_dropout,
                                      None, concat_pools, False, model_args)
    else:
        model = SentimentClassifier(model_args.model, ntokens, model_args.emsize, model_args.nhid, model_args.nlayers,
                                      model_args.classifier_hidden_layers, model_args.classifier_dropout, model_args.all_layers, concat_pools, False, model_args)
    args.heads_per_class = model_args.heads_per_class
    args.use_softmax = model_args.use_softmax
    try:
        args.classes = list(model_args.classes)
    except:
        args.classes = [args.label_key]

    try:
        args.dual_thresh = model_args.dual_thresh and not model_args.joint_binary_train
    except:
        args.dual_thresh = False

    if args.cuda:
        model.cuda()

    if args.fp16:
        model.half()

    if sd is not None:
        try:
            model.load_state_dict(sd)
        except:
            # if state dict has weight normalized parameters apply and remove weight norm to model while loading sd
            if hasattr(model.lm_encoder, 'rnn'):
                apply_weight_norm(model.lm_encoder.rnn)
            else:
                apply_weight_norm(model.lm_encoder)
            model.lm_encoder.load_state_dict(sd)
            remove_weight_norm(model)

    if args.neurons > 0:
        print('WARNING. Setting neurons %s' % str(args.neurons))
        model.set_neurons(args.neurons)
    return model

# uses similar function as transform from transfer.py
def classify(model, text, args):
    # Make sure to set *both* parts of the model to .eval() mode. 
    model.lm_encoder.eval()
    model.classifier.eval()
    # Initialize data, append results
    stds = np.array([])
    labels = np.array([])
    label_probs = np.array([])
    first_label = True
    heads_per_class = args.heads_per_class

    def get_batch(batch):
        text = batch['text'][0]
        timesteps = batch['length']
        labels = batch['label']
        text = Variable(text).long()
        timesteps = Variable(timesteps).long()
        labels = Variable(labels).long()
        if args.max_seq_len is not None:
            text = text[:, :args.max_seq_len]
            timesteps = torch.clamp(timesteps, max=args.max_seq_len)
        if args.cuda:
            text, timesteps, labels = text.cuda(), timesteps.cuda(), labels.cuda()
        return text.t(), labels, timesteps-1

    def get_outs(text_batch, length_batch):
        if args.model.lower() == 'transformer':
            class_out, (lm_or_encoder_out, state) = model(text_batch, length_batch, args.get_hidden)
        else:
            model.lm_encoder.rnn.reset_hidden(args.batch_size)
            for _ in range(1 + args.num_hidden_warmup):
                class_out, (lm_or_encoder_out, state) = model(text_batch, length_batch, args.get_hidden)
        if args.use_softmax and args.heads_per_class == 1:
            class_out = F.softmax(class_out, -1)
        return class_out, (lm_or_encoder_out, state)


    tstart = start = time.time()
    n = 0
    len_ds = len(text)
    with torch.no_grad():
        for i, data in tqdm(enumerate(text), total=len(text)):
            text_batch, labels_batch, length_batch = get_batch(data)
            size = text_batch.size(1)
            n += size
            # get predicted probabilities given transposed text and lengths of text
            probs, _ = get_outs(text_batch, length_batch)
#            probs = model(text_batch, length_batch)
            if first_label:
                first_label = False
                labels = []
                label_probs = []
                if heads_per_class > 1:
                    stds = []
            # Save variances, and predictions
            # TODO: Handle multi-head [multiple classes out]
            if heads_per_class > 1:
                _, probs, std, preds = probs
                stds.append(std.data.cpu().numpy())
            else:
                probs, preds = probs
                if args.use_softmax:
                    probs = F.softmax(probs, -1)
            labels.append(preds.data.cpu().numpy())
            label_probs.append(probs.data.cpu().numpy())

            num_char = length_batch.sum().item()

            end = time.time()
            elapsed_time = end - start
            total_time = end - tstart
            start = end

            s_per_batch = total_time / (i+1)
            timeleft = (len_ds - (i+1)) * s_per_batch
            ch_per_s = float(num_char) / elapsed_time

    if not first_label:
        labels = (np.concatenate(labels)) #.flatten())
        label_probs = (np.concatenate(label_probs)) #.flatten())
        if heads_per_class > 1:
            stds = (np.concatenate(stds))
        else:
            stds = np.zeros_like(labels)
    print('%0.3f seconds to transform %d examples' %
                  (time.time() - tstart, n))
    return labels, label_probs, stds

def make_header(classes, heads_per_class=1, softmax=False, dual_thresh=False):
    header = []
    if softmax:
        header.append('prediction')
    for cls in classes:
        if not softmax:
            header.append(cls + ' pred')
        header.append(cls + ' prob')
        if heads_per_class > 1:
            header.append(cls + ' std')
    if dual_thresh:
        header.append('neutral pred')
        header.append('neutral prob')
    return header

def get_row(pred, prob, std, classes, heads_per_class=1, softmax=False, dual_thresh=False):
    row = []
    if softmax:
        row.append(pred[0])
    for i in range(len(classes)):
        if not softmax:
            row.append(pred[i])
        row.append(prob[i])
        if heads_per_class > 1:
            row.append(std[i])
    if dual_thresh:
        row.append(pred[2])
        row.append(prob[2])
    return row 

def get_writer(preds, probs, stds, classes, heads_per_class=1, softmax=False, dual_thresh=False):
    header = make_header(classes, heads_per_class, softmax, dual_thresh)
    yield header
    for pred, prob, std in zip(preds, probs, stds):
        yield get_row(pred, prob, std, classes, heads_per_class, softmax, dual_thresh)

def main():
    (train_data, val_data, test_data), tokenizer, args = get_data_and_args()
    model = get_model(args)

    ypred, yprob, ystd = classify(model, train_data, args)

    save_root = ''
    save_root = os.path.join(save_root, args.save_probs)

    print('saving predicted probabilities to '+save_root)
    np.save(save_root, ypred)
    np.save(save_root+'.prob', yprob)
    np.save(save_root+'.std', ystd)

    if args.write_results is None or args.write_results == '':
        exit()

    print('writing results to '+args.write_results)
    writer = get_writer(ypred, yprob, ystd, args.classes, args.heads_per_class, args.use_softmax, args.dual_thresh)
    train_data.dataset.write(writer, path=args.write_results)

if __name__ == '__main__':
    main()
