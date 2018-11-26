import argparse
import os
import time
import math
import collections
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pandas as pd

from apex.reparameterization import apply_weight_norm, remove_weight_norm

from model import SentimentClassifier
from configure_data import configure_data
from arguments import add_general_args, add_model_args, add_classifier_model_args, add_run_classifier_data_args

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

    if args.cuda:
        model.cuda()

    if args.fp16:
        model.half()

    if sd is not None:
        try:
            model.load_state_dict(sd)
        except:
            # if state dict has weight normalized parameters apply and remove weight norm to model while loading sd
            if hasattr(model.encoder, 'rnn'):
                apply_weight_norm(model.encoder.rnn)
            else:
                apply_weight_norm(model.encoder)
            model.encoder.load_state_dict(sd)
            remove_weight_norm(model)

    if args.neurons > 0:
        print('WARNING. Setting neurons %s' % str(args.neurons))
        model.set_neurons(args.neurons)
    return model

# uses similar function as transform from transfer.py
def classify(model, text, args):
    # Make sure to set *both* parts of the model to .eval() mode. 
    model.encoder.eval()
    model.classifier.eval()
    # Initialize data, append results
    vars = np.array([])
    labels = np.array([])
    first_label = True
    heads_per_class = args.heads_per_class

    def get_batch(batch):
        text = batch['text'][0]
        timesteps = batch['length']
        labels = batch['label']
        text = Variable(text).long()
        timesteps = Variable(timesteps).long()
        labels = Variable(labels).long()
        if args.cuda:
            text, timesteps, labels = text.cuda(), timesteps.cuda(), labels.cuda()
        return text.t(), labels, timesteps-1

    def get_outs(text_batch, length_batch):
        if args.model.lower() == 'transformer' or args.model.lower() == 'bert':
            class_out, (lm_or_encoder_out, state) = model(text_batch, length_batch, args.get_hidden)
        else:
            model.encoder.rnn.reset_hidden(args.batch_size)
            for _ in range(1 + args.num_hidden_warmup):
                class_out, (lm_or_encoder_out, state) = model(text_batch, length_batch, args.get_hidden)
        if args.use_softmax:
            class_out = torch.max(class_out,-1)[1].view(-1,1)
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
                if heads_per_class > 1:
                    vars = []
            # Save variances, and predictions
            # TODO: Handle multi-head [multiple classes out]
            if heads_per_class > 1:
                prob_var = probs.view(probs.shape[0],-1,heads_per_class).contiguous().std(2)
                vars.append(prob_var[:,:].data.cpu().numpy())
                probs = probs.view(probs.shape[0],-1,heads_per_class).contiguous().mean(2)
            labels.append(probs[:,:].data.cpu().numpy())

            num_char = length_batch.sum().data[0]

            end = time.time()
            elapsed_time = end - start
            total_time = end - tstart
            start = end

            s_per_batch = total_time / (i+1)
            timeleft = (len_ds - (i+1)) * s_per_batch
            ch_per_s = float(num_char) / elapsed_time

    if not first_label:
        labels = (np.concatenate(labels)) #.flatten())
        if heads_per_class > 1:
            vars = (np.concatenate(vars))
        else:
            vars = np.zeros_like(labels)
    print('%0.3f seconds to transform %d examples' %
                  (time.time() - tstart, n))
    return labels, vars

def main():
    (train_data, val_data, test_data), tokenizer, args = get_data_and_args()
    model = get_model(args)

    ypred, yvar = classify(model, train_data, args)

    save_root = ''
    save_root = os.path.join(save_root, args.save_probs)

    print('saving predicted probabilities to '+save_root)
    np.save(save_root, ypred)

    if args.write_results is None or args.write_results == '':
        exit()

    #TODO: Handle multilabel/softmax properly
    def get_writer(probs):
        header = ['predicted proba'] if not args.use_softmax else ['predicted']
        yield header
        for prob in probs:
            yield prob

    print('writing results to '+args.write_results)
    writer = get_writer(ypred)
    train_data.dataset.write(writer, path=args.write_results)

if __name__ == '__main__':
    main()