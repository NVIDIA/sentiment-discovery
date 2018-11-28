import argparse
import os
import sys
import time
import math
import random
import collections
import pandas as pd
import pickle as pkl
import json
import torch
import torch.nn as nn
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef

from fp16 import FP16_Module, FP16_Optimizer
from apex.reparameterization import apply_weight_norm, remove_weight_norm

import model as M
from tqdm import tqdm
from model import DistributedDataParallel as DDP
from configure_data import configure_data
from learning_rates import AnnealingLR, SlantedTriangularLR, ConstantLR
from arguments import add_general_args, add_model_args, add_classifier_model_args, add_finetune_classifier_args
from metric_utils import update_info_dict, get_metric
from analysis import _binary_threshold, _neutral_threshold_two_output

def get_data_and_args():
    parser = argparse.ArgumentParser(description='PyTorch Sentiment Discovery Transfer Learning')
    parser = add_general_args(parser)
    parser = add_model_args(parser)
    parser = add_classifier_model_args(parser)
    data_config, data_parser, finetune_classifier_parser, parser = add_finetune_classifier_args(parser)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    if args.seed is not -1:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    (train_data, val_data, test_data), tokenizer = data_config.apply(args)
    args.data_size = tokenizer.num_tokens
    args.padding_idx = tokenizer.command_name_map['pad'].Id
    return (train_data, val_data, test_data), tokenizer, args

def get_model_and_optim(args, train_data):
    if args.use_softmax:
        args.report_no_thresholding = True
    ntokens = args.data_size
    concat_pools = args.concat_max, args.concat_min, args.concat_mean
    if args.model == 'transformer':
        model = M.SentimentClassifier(args.model, ntokens, None, None, None, args.classifier_hidden_layers, args.classifier_dropout,
                                      None, concat_pools, args.aux_lm_loss, args)
    else:
        model = M.SentimentClassifier(args.model, ntokens, args.emsize, args.nhid, args.nlayers,
                                      args.classifier_hidden_layers, args.classifier_dropout, args.all_layers, concat_pools, args.aux_lm_loss, args)
    if args.cuda:
        model.cuda()

    if args.fp16:
        model.half()
    # load char embedding and recurrent encoder for featurization
    if args.load is not None and args.load != '':
        with open(args.load, 'rb') as f:
            sd = x = torch.load(f)
            if 'sd' in sd:
                sd = sd['sd']
            if not args.load_finetuned and 'encoder' in sd:
                sd = sd['encoder']

        if not args.load_finetuned:
            try:
                model.encoder.load_state_dict(sd)
            except:
                # if state dict has weight normalized parameters apply and remove weight norm to model while loading sd
                if hasattr(model.encoder, 'rnn'):
                    apply_weight_norm(model.encoder.rnn)
                else:
                    apply_weight_norm(model.encoder)
                model.encoder.load_state_dict(sd)
                remove_weight_norm(model)
        else:
            model.load_state_dict(sd)

    optims = {
        'adam' : 'Adam',
        'sgd'  : 'SGD'
    }

    optim = eval('torch.optim.'+ optims[args.optim.lower()])(model.parameters(), lr=args.lr)
    iters_per_epoch = len(train_data)
    num_iters = iters_per_epoch * args.epochs

    assert not (args.stlr_cut_frac and args.cos_cut_frac)
    if args.stlr_cut_frac is not None:
        LR = SlantedTriangularLR(optim, max_val=args.lr, cut_frac=args.stlr_cut_frac, num_iters=num_iters)
    elif args.cos_cut_frac is not None:
        LR = AnnealingLR(optim, start_lr=args.lr, warmup_iter=int(args.cos_cut_frac * num_iters), num_iters=num_iters, decay_style='cosine')
    elif args.decay_style is not None:
        warmup_iters = int(args.warmup_epochs * iters_per_epoch)
        if args.decay_epochs == -1:
            decay_iters = int(args.epochs * iters_per_epoch)
        else:
            decay_iters = int(args.decay_epochs * iters_per_epoch)
        if args.decay_style == 'constant':
            #TODO: implement
            LR = AnnealingLR(optim, start_lr=args.lr, warmup_iter=warmup_iters, num_iters=decay_iters+warmup_iters, decay_style=args.decay_style)
        elif args.decay_style == 'linear':
            #TODO: implement
            LR = AnnealingLR(optim, start_lr=args.lr, warmup_iter=warmup_iters, num_iters=decay_iters+warmup_iters, decay_style=args.decay_style)
        elif args.decay_style == 'cosine':
            LR = AnnealingLR(optim, start_lr=args.lr, warmup_iter=warmup_iters, num_iters=decay_iters+warmup_iters, decay_style=args.decay_style)
        elif args.decay_style == 'exponential':
            #TODO: implement
            LR = ConstantLR(optim, lr=args.lr)
        else:
            LR = ConstantLR(optim, lr=args.lr)
    else:
        LR = ConstantLR(optim, lr=args.lr)
    return model, optim, LR

def get_supervised_batch(batch, use_cuda, model, ids=False, args=None, save_outputs=False,  heads_per_class=1):
    '''
    Process batch and return tuple of (text, text label, text length) long tensors.
    Text is returned in column format with (time, batch) dimensions.
    '''
    text = batch['text'][0]
    timesteps = batch['length']
    labels = batch['label']
    text = Variable(text).long()
    timesteps = Variable(timesteps).long()
    labels = Variable(labels)
    if args.use_softmax:
        labels = Variable(labels).view(-1).long()
    elif heads_per_class > 1:
        labels = labels.view(labels.size(0), -1, 1)
        labels = labels.expand(-1,-1,heads_per_class)
        labels = Variable(labels).view(-1, heads_per_class).float()
    else:
        labels = labels.view(-1, model.out_dim).float()
    if use_cuda:
        text, timesteps, labels = text.cuda(), timesteps.cuda(), labels.cuda()
    return text.t(), labels, timesteps-1

def transform(model, text_batch, labels_batch, length_batch, args, LR=None):
    batch_size = text_batch.size(1)

    def get_outs():
        if args.model.lower() == 'transformer' or args.model.lower() == 'bert':
            class_out, (lm_or_encoder_out, state) = model(text_batch, length_batch, args.get_hidden)
        else:
            model.encoder.rnn.reset_hidden(args.batch_size)
            for _ in range(1 + args.num_hidden_warmup):
                class_out, (lm_or_encoder_out, state) = model(text_batch, length_batch, args.get_hidden)
        if args.use_softmax:
            class_out = torch.max(class_out,-1)[1].view(-1,1)
        return class_out, (lm_or_encoder_out, state)

    if LR is not None and not args.use_logreg:
        # doing true finetuning
        class_out, lm_or_encoder_out = get_outs()
    else:
        with torch.no_grad():
            class_out, lm_or_encoder_out = get_outs()

    class_out = class_out.float().view(-1, model.out_dim)
    return class_out, lm_or_encoder_out

def train_logreg(args, trX, trY, vaX=None, vaY=None, teX=None, teY=None, penalty='l1', max_iter=100,
        C=2**np.arange(-8, 1).astype(np.float), seed=42, model=None, eval_test=True, neurons=None, drop_neurons=False):
    """
    slightly modified version of openai implementation https://github.com/openai/generating-reviews-discovering-sentiment/blob/master/utils.py
    if model is not None it doesn't train the model before scoring, it just scores the model
    """
    # if only integer is provided for C make it iterable so we can loop over
    if not isinstance(C, collections.Iterable):
        C = list([C])
    # Cross validation over C
    n_classes = 1
    if len(trY.shape)>1:
        n_classes = trY.shape[-1]
    scores = []
    if model is None:
        for i, c in enumerate(C):
            if n_classes <= 1:
                model = LogisticRegression(C=c, penalty=penalty, max_iter=max_iter, random_state=seed)
                model.fit(trX, trY)
                blank_info_dict = {'fp' : 0, 'tp' : 0, 'fn' : 0, 'tn' : 0, 'std' : 0.,
                                   'metric' : args.metric, 'micro' : args.micro}
                if vaX is not None:
                    info_dict = update_info_dict(blank_info_dict.copy(), vaY, model.predict_proba(vaX)[:, -1])
                else:
                    info_dict = update_info_dict(blank_info_dict.copy(), trY, model.predict_proba(trX)[:, -1])
                scores.append(get_metric(info_dict))
                print(scores[-1])
                del model
            else:
                info_dicts = []
                model = []
                for cls in range(n_classes):
                    _model = LogisticRegression(C=c, penalty=penalty, max_iter=max_iter, random_state=seed)
                    _model.fit(trX, trY[:, cls])
                    blank_info_dict = {'fp' : 0, 'tp' : 0, 'fn' : 0, 'tn' : 0, 'std' : 0.,
                                       'metric' : args.metric, 'micro' : args.micro}
                    if vaX is not None:
                        info_dict = update_info_dict(blank_info_dict.copy(), vaY[:, cls], _model.predict_proba(vaX)[:, -1])
                    else:
                        info_dict = update_info_dict(blank_info_dict.copy(), trY[:, cls], _model.predict_proba(trX)[:, -1])
                    info_dicts.append(info_dict)
                    model.append(_model)
                scores.append(get_metric(info_dicts))
                print(scores[-1])
                del model
        c = C[np.argmax(scores)]
        if n_classes <= 1:
            model = LogisticRegression(C=c, penalty=penalty, max_iter=max_iter, random_state=seed)
            model.fit(trX, trY)
        else:
            model = []
            for cls in range(n_classes):
                _model = LogisticRegression(C=c, penalty=penalty, max_iter=max_iter, random_state=seed)
                _model.fit(trX, trY[:, cls])
                model.append(_model)
    else:
        c = model.C
    # predict probabilities and get accuracy of regression model on train, val, test as appropriate
    # also get number of regression weights that are not zero. (number of features used for modeling)
    scores = []
    if n_classes == 1:
        nnotzero = np.sum(model.coef_ != 0)
        preds = model.predict_proba(trX)[:, -1]
        train_score = get_metric(update_info_dict(blank_info_dict.copy(), trY, preds))
    else:
        nnotzero = 0
        preds = []
        info_dicts = []
        for cls in range(n_classes):
            nnotzero += np.sum(model[cls].coef_ != 0)
            _preds = model[cls].predict_proba(trX)[:, -1]
            info_dicts.append(update_info_dict(blank_info_dict.copy(), trY[:, cls], _preds))
            preds.append(_preds)
        nnotzero/=n_classes
        train_score = get_metric(info_dicts)
        preds = np.concatenate([p.reshape((-1, 1)) for p in preds], axis=1)
    scores.append(train_score * 100)
    if vaX is None:
        eval_data = trX
        eval_labels = trY
        val_score = train_score
    else:
        eval_data = vaX
        eval_labels = vaY
        if n_classes == 1:
            preds = model.predict_proba(vaX)[:, -1]
            val_score = get_metric(update_info_dict(blank_info_dict.copy(), vaY, preds))
        else:
            preds = []
            info_dicts = []
            for cls in range(n_classes):
                _preds = model[cls].predict_proba(vaX)[:, -1]
                info_dicts.append(update_info_dict(blank_info_dict.copy(), vaY[:, cls], _preds))
                preds.append(_preds)
            val_score = get_metric(info_dicts)
            preds = np.concatenate([p.reshape((-1, 1)) for p in preds], axis=1)
    val_preds = preds
    val_labels = eval_labels
    scores.append(val_score * 100)
    eval_score = val_score
    threshold = .5
    if args.automatic_thresholding:
        _, threshold, _, _ = _binary_threshold(preds.reshape(-1,1), eval_labels.reshape(-1,1), args.metric, args.micro)
        threshold = float(threshold.squeeze())
    if teX is not None and teY is not None and eval_test:
        eval_data = teX
        eval_labels = teY
        if n_classes == 1:
            preds = model.predict_proba(eval_data)[:, -1]
        else:
            preds = []
            for cls in range(n_classes):
                _preds = model[cls].predict_proba(eval_data)[:, -1]
                preds.append(_preds)
            preds = np.concatenate([p.reshape((-1, 1)) for p in preds], axis=1)
    if n_classes == 1:
        eval_score = get_metric(update_info_dict(blank_info_dict.copy(), eval_labels, preds, threshold=threshold))
    else:
        info_dicts = []
        for cls in range(n_classes):
            info_dicts.append(update_info_dict(blank_info_dict.copy(), eval_labels[:, cls], preds[:, cls]))
        eval_score = get_metric(info_dicts)

    scores.append(eval_score * 100)
    return model, scores, c, nnotzero

def finetune(model, text, args, val_data=None, LR=None, reg_loss=None, tqdm_desc='nvidia', save_outputs=False,
    heads_per_class=1, default_threshold=0.5, last_thresholds=[], threshold_validation=True, debug=False):
    '''
    Apply featurization `model` to extract features from text in data loader.
    Featurization model should return cell state not hidden state.
    `text` data loader should return tuples of ((text, text length), text label)
    Returns labels and features for samples in text.
    '''
    # NOTE: If in training mode, do not run in .eval() mode. Bug fixed.
    if LR is None:
        model.encoder.eval()
        model.classifier.eval()
    else:
        # Very important to reset back to train mode for future epochs!
        model.encoder.train()
        model.classifier.train()

    # Optionally, freeze language model (train MLP only)
    # NOTE: un-freeze gradients if they every need to be tweaked in future iterations
    if args.freeze_lm:
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Choose which losses to implement
    if args.use_softmax:
        binary_loss_fn = torch.nn.CrossEntropyLoss()
    else:
        binary_loss_fn = torch.nn.BCELoss()
    if args.aux_lm_loss:
        aux_loss_fn = torch.nn.CrossEntropyLoss(reduce=False)
    else:
        aux_loss_fn = None

    if args.thresh_test_preds:
        thresholds = pd.read_csv(args.thresh_test_preds, header=None).values.squeeze()
    elif len(last_thresholds) > 0:
        # Re-use previous thresholds, if provided.
        # Why? More accurate reporting, and not that slow. Don't compute thresholds on training, for example -- but can recycle val threshold
        thresholds = last_thresholds
    else:
        # Default thresholds -- faster, but less accurate
        thresholds = np.array([default_threshold for _ in range(model.out_dim)])

    total_loss = 0
    total_classifier_loss = 0
    total_lm_loss = 0
    total_multihead_variance_loss = 0
    class_accuracies = torch.zeros(model.out_dim).cuda()
    if model.out_dim/heads_per_class > 1 and not args.use_softmax:
        keys = args.non_binary_cols
    elif args.use_softmax:
        keys = [str(m) for m in range(model.out_dim)]
    else:
        keys = ['']
    info_dicts = [{'fp' : 0, 'tp' : 0, 'fn' : 0, 'tn' : 0, 'std' : 0,
                   'metric' : args.metric, 'micro' : args.micro} for k in keys]

    # Sanity check -- should do this sooner. Does #classes match expected output?
    assert model.out_dim == len(keys) * heads_per_class, "model.out_dim does not match keys (%s) x heads_per_class (%d)" % (keys, heads_per_class)

    batch_adjustment = 1. / len(text)
    # Save all outputs *IF* small enough, and requested for thresholding -- basically, on validation
    #if threshold_validation and LR is not None:
    all_batches = []
    all_labels = []
    print('Running %d batches' % len(text))
    for i, data in tqdm(enumerate(text), total=len(text), unit="batch", desc=tqdm_desc, position=1, ncols=100):
        text_batch, labels_batch, length_batch = get_supervised_batch(data, args.cuda, model, args.ids, args, heads_per_class=args.heads_per_class)
        class_out, (lm_out, _) = transform(model, text_batch, labels_batch, length_batch, args, LR)
        classifier_loss = binary_loss_fn(class_out, labels_batch)
        loss = classifier_loss
        classifier_loss = classifier_loss.clone() # save for reporting
        # Also compute multihead variance loss -- from classifier [divide by output size since it scales linearly]
        if args.aux_head_variance_loss_weight > 0.:
            multihead_variance_loss = model.classifier.get_last_layer_variance() / model.out_dim
            loss = loss + multihead_variance_loss * args.aux_head_variance_loss_weight
        # Divide by # batches? Since we're looking at the parameters here, and should be batch independent.
        # multihead_variance_loss *= batch_adjustment

        if args.aux_lm_loss:
            lm_labels = text_batch[1:]
            lm_losses = aux_loss_fn(lm_out[:-1].view(-1, lm_out.size(2)).contiguous().float(),
                                      lm_labels.contiguous().view(-1))

            padding_mask = (torch.arange(lm_labels.size(0)).unsqueeze(1).cuda() > length_batch).float()
            portion_unpadded = padding_mask.sum() / padding_mask.size(0)
            lm_loss = portion_unpadded * torch.mean(lm_losses * (padding_mask.view(-1).float()))

            # Scale LM loss -- since it's so big
            if args.aux_lm_loss_weight > 0.:
                loss = loss + lm_loss * args.aux_lm_loss_weight

        # Training
        if LR is not None:
            LR.optimizer.zero_grad()
            loss.backward()
            LR.optimizer.step()
            LR.step()

        # Remove loss from CUDA -- kill gradients and save memory.
        total_loss += loss.detach().cpu().numpy()
        if args.use_softmax:
            labels_batch = onehot(labels_batch.squeeze(), model.out_dim)
            class_out = onehot(torch.max(class_out, -1)[1].squeeze(), model.out_dim/heads_per_class)
        total_classifier_loss += classifier_loss.detach().cpu().numpy()
        if args.aux_lm_loss:
            total_lm_loss += lm_loss.detach().cpu().numpy()
        if args.aux_head_variance_loss_weight > 0:
            total_multihead_variance_loss += multihead_variance_loss.detach().cpu().numpy()
        for j in range(model.out_dim):
            info_dicts[math.floor(j/heads_per_class)] = update_info_dict(info_dicts[math.floor(j/heads_per_class)], labels_batch[:, j], class_out[:, j], thresholds[j])
        # Save, for overall thresholding (not on training)
        if threshold_validation and LR is None:
            all_labels.append(labels_batch.detach().cpu().numpy())
            all_batches.append(class_out.detach().cpu().numpy())

    if threshold_validation and LR is None:
        all_batches = np.concatenate(all_batches)
        all_labels = np.concatenate(all_labels)
        # Compute new thresholds -- per class
        _, thresholds, _, _ =  _binary_threshold(all_batches, all_labels, args.threshold_metric, True, global_tweaks=0, heads_per_class=heads_per_class, class_single_threshold=False)
        info_dicts = [{'fp' : 0, 'tp' : 0, 'fn' : 0, 'tn' : 0, 'std' : 0.,
               'metric' : args.metric, 'micro' : args.micro} for k in keys]
        # In multihead case, look at class averages? Why? More predictive. Works especially well when we force single per-class threshold.
        if heads_per_class > 1 and args.use_class_multihead_average:
            for j in range(int(model.out_dim/heads_per_class)):
                # Pass long std/variances per class, as well!
                info_dicts[j] = update_info_dict(info_dicts[j],
                    all_labels[:, j*heads_per_class:(j+1)*heads_per_class].sum(1)/heads_per_class,
                    all_batches[:, j*heads_per_class:(j+1)*heads_per_class].sum(1)/heads_per_class,
                    threshold=thresholds[j*heads_per_class:(j+1)*heads_per_class].sum()/heads_per_class,
                    std=all_batches[:, j*heads_per_class:(j+1)*heads_per_class].std(1))
        else:
            for j in range(model.out_dim):
                info_dicts[math.floor(j/heads_per_class)] = update_info_dict(info_dicts[math.floor(j/heads_per_class)], all_labels[:, j], all_batches[:, j], thresholds[j])

    # Metrics for all items -- with current best thresholds
    total_metrics, class_metric_strs = get_metric_report(info_dicts, args, keys, LR)

    # Show losses
    if debug:
        print('losses -- total / classifier / LM / multihead_variance')
        print(total_loss * batch_adjustment)
        print(total_classifier_loss * batch_adjustment)
        print(total_lm_loss * batch_adjustment)
        print(total_multihead_variance_loss * batch_adjustment)

    # class_metric_strs, total_metrics = [], []
    # report_metrics = ['mcc', 'f1'] if args.all_metrics else [args.metric]
    # for m in report_metrics:
    #     for d in info_dicts:
    #         d.update({'metric' : m})
    #     class_metrics = [get_metric(d) for d in info_dicts]
    #     total_metrics.append(get_metric(info_dicts))

    #     if LR is not None:
    #         delim = '-'
    #     else:
    #         delim = {'mcc' : '#', 'f1' : '+', 'jacc' : '=', 'acc' : '>'}[m]
    #     class_metric_strs.append(", ".join('{} {} {:5.2f}'.format(k, delim, f * 100) for k, f in zip(keys, class_metrics)))

    return total_loss.item() / (i + 1), total_metrics, class_metric_strs, thresholds

def onehot(sparse, nclasses):
    rows = len(sparse)
    rtn = torch.zeros(rows, math.floor(nclasses))
    rtn[torch.arange(rows), sparse.squeeze().cpu()] = 1
    return rtn

def get_metric_report(info_dicts, args, keys=['-'], LR=None):
    class_metric_strs, total_metrics = [], []
    # report_metrics = ['jacc', 'acc', 'mcc', 'f1', 'var'] if args.all_metrics else [args.metric]
    report_metrics = ['jacc', 'acc', 'mcc', 'f1', 'recall', 'precision', 'var'] if args.all_metrics else [args.metric]
    for m in report_metrics:
        for d in info_dicts:
            d.update({'metric' : m})
        class_metrics = [get_metric(d) for d in info_dicts]
        total_metrics.append(get_metric(info_dicts))

        if LR is not None:
            delim = '-'
        else:
            delim = {'mcc' : '#', 'f1' : '+', 'jacc' : '=', 'acc' : '>', 'var' : '%', 'recall': '<', 'precision':'~'}[m]
        class_metric_strs.append(", ".join('{} {} {:5.2f}'.format(k, delim, f * 100) for k, f in zip(keys, class_metrics)))

    return total_metrics, class_metric_strs

def generate_outputs(model, text, args, thresholds=None, debug=False):
    model.eval()
    collected_outputs = torch.zeros(1, model.out_dim).cuda()
    # Average across heads in a category, if requested
    if args.use_class_multihead_average and thresholds is not None:
        collected_outputs = torch.zeros(1, int(model.out_dim/args.heads_per_class)).cuda()
    collected_labels = collected_outputs.clone()
    # Collect category standard deviations, across multiple heads
    collected_outputs_std = collected_outputs.clone()
    if args.thresh_test_preds:
        thresholds = pd.read_csv(args.thresh_test_preds, header=None).values.squeeze()
        assert len(thresholds) == model.out_dim or args.double_thresh
        collected_outputs = collected_outputs.long()
    elif thresholds is not None:
        thresholds = thresholds.squeeze()
        if args.use_class_multihead_average:
            thresholds_average = thresholds.reshape(-1,args.heads_per_class).sum(1)/args.heads_per_class
            if debug:
                print(thresholds_average)
            thresholds = thresholds_average

    for i, data in tqdm(enumerate(text), total=len(text), unit='batch', desc='predictions', position=1, ncols=100):
        text_batch, labels_batch, length_batch = get_supervised_batch(data, args.cuda, model, args.ids, args, save_outputs=True, heads_per_class=args.heads_per_class)
        class_out, (lm_out, _) = transform(model, text_batch, labels_batch, length_batch, args)
        # Take the average per-category if requested
        if args.use_class_multihead_average and thresholds is not None:
            class_average = class_out.view(class_out.shape[0],-1,args.heads_per_class).contiguous().mean(2)
            labels_average = labels_batch.view(labels_batch.shape[0],-1,args.heads_per_class).contiguous().mean(2)
            # Also compute variance per item, per category [for reporting, and for active learning]
            class_std = class_out.view(class_out.shape[0],-1,args.heads_per_class).contiguous().std(2)
            class_out = class_average
            labels_batch = labels_average
        else:
            class_std = torch.zeros(class_out.shape)

        if args.thresh_test_preds or thresholds is not None:
            if args.double_thresh:
                def double_thresh(x):
                    if x > max(thresholds):
                        return 1
                    elif x > min(thresholds):
                        return 0.5
                    else:
                        return 0
                class_out = pd.Series(class_out.squeeze().cpu().numpy()).apply(double_thresh).values.astype(int).reshape(-1, 1)
            else:
                class_out = (class_out > torch.tensor(thresholds).cuda().float()).long()

        if args.use_softmax:
            labels_batch = onehot(labels_batch.squeeze(), model.out_dim).cuda()
            class_out = onehot(torch.max(class_out, -1)[1].squeeze(), model.out_dim)

        collected_outputs = torch.cat((collected_outputs, torch.tensor(class_out).cuda().float()), 0)
        collected_labels = torch.cat((collected_labels, labels_batch), 0)
        collected_outputs_std = torch.cat((collected_outputs_std, torch.tensor(class_std).cuda().float()), 0)

    if not args.automatic_thresholding and not args.use_softmax:
        write_results(np.array(collected_outputs[1:]), np.array(collected_labels[1:]), args.save_test_preds)
        # np.savetxt(args.save_test_preds, np.array(collected_outputs[1:]), delimiter=',')
        # labels_file = os.path.splitext(args.save_test_preds)[0] + '_labels.txt'
        # np.savetxt(labels_file, np.array(collected_labels[1:]), delimiter=',')
    else:
        return collected_outputs[1:], collected_labels[1:], collected_outputs_std[1:]

def write_results(preds, labels, save):
    labels_file = os.path.splitext(save)[0] + '_labels.txt'
    # HACK -- handle both tensors and numpy arrays here:
    if isinstance(preds, np.ndarray):
        np.savetxt(save, preds.astype(int), delimiter=',')
        np.savetxt(labels_file, labels.astype(int), delimiter=',')
    else:
        np.savetxt(save, preds.cpu().numpy().astype(int), delimiter=',')
        np.savetxt(labels_file, labels.cpu().numpy().astype(int), delimiter=',')

def main():
    (train_data, val_data, test_data), tokenizer, args = get_data_and_args()
    # Print args for logging & reproduction. Need to know, including default args
    if test_data is None:
        test_data = val_data
    model, optim, LR = get_model_and_optim(args, train_data)

    # save_root = '' if args.load is None else args.load
    # save_root = save_root.replace('.current', '')
    # save_root = os.path.splitext(save_root)[0]
    # save_root += '_transfer'
    # save_root = os.path.join(save_root, args.model_version_name)
    # if not os.path.exists(save_root):
    #     os.makedirs(save_root)
    # print('writing results to '+save_root)

    def clf_reg_loss(reg_penalty=.125, order=1):
        loss = 0
        for p in model.classifier.parameters():
            loss += torch.abs(p).sum()*reg_penalty
        return loss
    reg_loss = clf_reg_loss
    init_params = list(model.encoder.parameters())

    if args.use_logreg:
        def transform_for_logreg(model, data, args, desc='train'):
            if data is None:
                return None, None

            X_out = []
            Y_out = []
            for i, batch in tqdm(enumerate(data), total=len(data), unit="batch", desc=desc, position=0, ncols=100):
                text_batch, labels_batch, length_batch = get_supervised_batch(batch, args.cuda, model, args.ids, args, heads_per_class=args.heads_per_class)
                # if args.non_binary_cols:
                #     labels_batch = labels_batch[:,0]-labels_batch[:,1]+1
                _, (_, state) = transform(model, text_batch, labels_batch, length_batch, args)
                X_out.append(state.cpu().numpy())
                Y_out.append(labels_batch.cpu().numpy())
            X_out = np.concatenate(X_out)
            Y_out = np.concatenate(Y_out)
            return X_out, Y_out

        model.eval()
        trX, trY = transform_for_logreg(model, train_data, args, desc='train')
        vaX, vaY = transform_for_logreg(model, val_data, args, desc='val')
        teX, teY = transform_for_logreg(model, test_data, args, desc='test')

        logreg_model, logreg_scores, c, nnotzero = train_logreg(args, trX, trY, vaX, vaY, teX, teY, eval_test=not args.no_test_eval)
        print(', '.join([str(score) for score in logreg_scores]), 'train, val, test accuracy for all neuron regression')
        print(str(c)+' regularization coefficient used')
        print(str(nnotzero) + ' features used in all neuron regression\n')
    else:
        best_vaY = 0
        vaT = [] # Current "best thresholds" so we can get reasonable estimates on training set
        for e in tqdm(range(args.epochs), unit="epoch", desc="epochs", position=0, ncols=100):
            if args.use_softmax:
                vaT = []
            save_outputs = False
            report_metrics = ['jacc', 'acc','mcc', 'f1', 'recall', 'precision', 'var'] if args.all_metrics else [args.metric]
            print_str = ""
            trXt, trY, trC, _ = finetune(model, train_data, args, val_data=val_data, LR=LR, reg_loss=reg_loss, tqdm_desc='train', heads_per_class=args.heads_per_class, last_thresholds=vaT)
            data_str_base = "Train Loss: {:4.2f} Train {:5s} (All): {:5.2f}, Train Class {:5s}: {}"
            for idx, m in enumerate(report_metrics):
                data_str = data_str_base.format(trXt, m, trY[idx] * 100, m, trC[idx])
                print_str += data_str + " " * max(0, 110 - len(data_str)) + "\n"

            vaXt, vaY = None, None
            if val_data is not None:
                vaXt, vaY, vaC, vaT = finetune(model, val_data, args, tqdm_desc='val', heads_per_class=args.heads_per_class, last_thresholds=vaT)
                # Take command line, for metric for which to measure best performance against.
                # NOTE: F1, MCC, Jaccard are good measures. Accuracy is not -- since so skewed.
                selection_metric = ['jacc', 'acc','mcc', 'f1', 'recall', 'precision', 'var'].index(args.threshold_metric)
                avg_Y = vaY[selection_metric]
                tqdm.write('avg_metric '+str(avg_Y))
                if avg_Y > best_vaY:
                    save_outputs = True
                    best_vaY = avg_Y
                data_str_base = "Val   Loss: {:4.2f} Val   {:5s} (All): {:5.2f}, Val   Class {:5s}: {}"
                for idx, m in enumerate(report_metrics):
                    data_str = data_str_base.format(vaXt, m, vaY[idx] * 100, m, vaC[idx])
                    print_str += data_str + " " * max(0, 110 - len(data_str)) + "\n"
                tqdm.write(print_str[:-1])
            teXt, teY = None, None
            if test_data is not None:
                # Hardcode -- enable to always save outputs [regardless of metrics]
                # save_outputs = True
                if save_outputs:
                    tqdm.write('saving outputs')
                    try:
                        with torch.no_grad():
                            if args.automatic_thresholding or args.report_no_thresholding:
                                auto_thresholds = None
                                dual_thresholds = None
                                # NOTE -- we manually threshold to F1 [not necessarily good]
                                if not args.report_no_thresholding:
                                    V_pred, V_label, V_std = generate_outputs(model, val_data, args)
                                    if args.dual_threshold:
                                        # get dual threshold (do not call auto thresholds)
                                        # TODO: Handle multiple heads per class
                                        _, dual_thresholds = _neutral_threshold_two_output(V_pred.cpu().numpy(), V_label.cpu().numpy())
                                    else:
                                        # Use args.threshold_metric to choose which category to threshold on. F1 and Jaccard are good options
                                        # NOTE: For multiple heads per class, can threshold each head (default) or single threshold. Little difference once model converges.
                                        _, auto_thresholds, _, _ = _binary_threshold(V_pred.view(-1, model.out_dim).contiguous(), V_label.view(-1, model.out_dim).contiguous(),
                                            args.threshold_metric, args.micro, global_tweaks=args.global_tweaks, heads_per_class=args.heads_per_class, class_single_threshold=args.class_single_threshold)
                                T_pred, T_label, T_std = generate_outputs(model, test_data, args, auto_thresholds)
                                if not args.use_softmax and model.out_dim / args.heads_per_class > 1:
                                    keys = args.non_binary_cols
                                    if args.dual_threshold:
                                        if len(keys) == len(dual_thresholds):
                                            print('Dual thresholds: %s' % str(list(zip(keys, dual_thresholds))))
                                    else:
                                        print('Class thresholds: %s' % str(list(zip(keys, auto_thresholds))))
                                    #add neutral key to list of keys
                                    if args.dual_threshold:
                                        keys += ['neutral']
                                elif args.use_softmax:
                                    keys = [str(m) for m in range(model.out_dim)]
                                else:
                                    print('Class threshold: %s' % str([args.label_key, auto_thresholds[0]]))
                                    keys = ['']
                                info_dicts = [{'fp' : 0, 'tp' : 0, 'fn' : 0, 'tn' : 0, 'std' : 0.,
                                               'metric' : args.metric, 'micro' : True} for k in keys]
                                #perform dual threshold here, adding the neutral labels to T_label, thresholding existing predictions and adding neutral preds to T_Pred
                                if args.dual_threshold:
                                    if dual_thresholds is None:
                                        dual_thresholds = [.5, .5]
                                    def make_onehot_w_neutral(label):
                                        rtn = [0]*3
                                        rtn[label] = 1
                                        return rtn
                                    def get_label(pos_neg):
                                        thresholded = [pos_neg[0]>=dual_thresholds[0], pos_neg[1]>=dual_thresholds[1]]
                                        if thresholded[0] == thresholded[1]:
                                            return 2
                                        return thresholded.index(1)
                                    new_labels = []
                                    new_preds = []
                                    for j, lab in  enumerate(T_label):
                                        pred = T_pred[j]
                                        new_preds.append(make_onehot_w_neutral(get_label(pred)))
                                        new_labels.append(make_onehot_w_neutral(get_label(lab)))
                                    T_pred = np.array(new_preds)
                                    T_label = np.array(new_labels)

                                # HACK: If dual threshold, hardcoded -- assume positive, negative and neutral -- in that order
                                # It's ok to train with other categories (after positive, neutral) as auxilary loss -- but won't calculate in test
                                if args.dual_threshold:
                                    print(keys)
                                    keys = ['positive', 'negative', 'neutral']
                                    info_dicts = [{'fp' : 0, 'tp' : 0, 'fn' : 0, 'tn' : 0, 'std' : 0.,
                                                   'metric' : args.metric, 'micro' : True} for k in keys]
                                    print(keys)
                                # In the case of multiple heads per class, group here -- as we iterate over keys
                                for j, k in enumerate(keys):
                                    # Unless we already grouped average predictions.
                                    heads = 1 if args.use_class_multihead_average else args.heads_per_class
                                    for offset in range(heads):
                                        j_off = j * heads + offset
                                        # Debug, remove
                                        #print('%d/%d -- %s' % (j_off, len(keys) * args.heads_per_class, k))
                                        # If neutrals, derive neutral "STD" from neg, pos average
                                        if args.dual_threshold:
                                            neutral_std = torch.cat((T_std[:,0].unsqueeze(1), T_std[:,1].unsqueeze(1)), 1)
                                            neutral_std = neutral_std.mean(1)
                                            T_std = torch.cat((T_std, neutral_std.unsqueeze(1)), 1)
                                            # HACK: For dual threshold, mismatch, get tensor on CPU
                                            T_std = T_std.cpu()
                                        p, l, v = T_pred[:,j_off], T_label[:,j_off], T_std[:,j_off]
                                        update_info_dict(info_dicts[j], p, l, std=v)
                                tqdm.write(str(info_dicts))
                                total_metrics, metric_strings = get_metric_report(info_dicts, args, keys)
                                tqdm.write(str(total_metrics))
                                tqdm.write('; '.join(metric_strings))
                                # TODO: Write (per item) standard deviations, along with results!
                                print('Saving prediction results %s to %s' % (str(T_pred.shape), args.save_test_preds))
                                write_results(T_pred, T_label, args.save_test_preds)
                            else:
                                test_preds_path = args.save_test_preds
                                args.save_test_preds = args.save_test_preds[:-4]+'.val.txt'
                                tqdm.write(args.save_test_preds)
                                generate_outputs(model, val_data, args)
                                args.save_test_preds = test_preds_path
                                generate_outputs(model, test_data, args)
                    except KeyboardInterrupt:
                        pass
                else:
                    pass
            # Save the model, upon request
            if args.save_finetune and save_outputs:
                # Save model if best so far. Note epoch number, and also keys [what is it predicting], as well as optional version number
                # TODO: Add key string to handle multiple runs?
                if args.non_binary_cols:
                    keys = args.non_binary_cols
                else:
                    keys = [args.label_key]
                model_save_path = os.path.splitext(args.save_test_preds)[0]+'_model_'+args.model_version_name+'_'+'_'.join(keys)+'_ep'+str(e)+'.pt'
                print('Saving model to %s' % model_save_path)
                torch.save(model.state_dict(), model_save_path)
                # Also save args
                args_save_path = os.path.splitext(args.save_test_preds)[0]+'_args_'+args.model_version_name+'_'+'_'.join(keys)+'_ep'+str(e)+'.txt'
                print('Args to %s' % args_save_path)
                with open(args_save_path, 'w') as f:
                    f.write('\n'.join(sys.argv[1:]))
                # Also save thresholds
                thresh_save_path = os.path.splitext(args.save_test_preds)[0]+'_thresholds_'+args.model_version_name+'_'+'_'.join(keys)+'_ep'+str(e)+'.npy'
                print('Thresh to %s' % thresh_save_path)
                # add thresholds to arguments for easy reloading of model config
                if not args.report_no_thresholding:
                    if args.dual_threshold:
                        np.save(thresh_save_path, list(zip(keys, dual_thresholds)))
                        args.thresholds = list(zip(keys, dual_thresholds))
                    else:
                        np.save(thresh_save_path, list(zip(keys, auto_thresholds)))
                        args.thresholds = list(zip(keys, auto_thresholds))
                else:
                    args.thresholds = None
                args.classes = keys
                clf_save_path = os.path.splitext(args.save_test_preds)[0]+'_model_'+args.model_version_name+'_'+'_'.join(keys)+'_ep'+str(e)+'.clf'
                print('Saving full classifier to %s' % clf_save_path)
                torch.save({'sd': model.state_dict(), 'args': args}, clf_save_path)


if __name__ == "__main__":
    main()


# python3 finetune.py --data csvs/SemEval-7k-processed-IDs.train.csv --valid csvs/SemEval-7k-processed-IDs.val.csv --test csvs/SemEval-7k-processed-IDs.test.csv --epochs 5 --text_key 32k-ids --ids --optim adam --data_size 32000 --aux-lm-loss --label_key label --all-metrics --automatic-thresholding --batch_size 24 --lr 1.73e-5 --model transformer --decoder-embed-dim 768 --decoder-ffn-embed-dim 3072 --decoder-layers 12  --load /home/adlr-sent.cosmos433/chkpts/tf-768emb-3072ffn-12x8head-learnedpos-32000parts-2cos-300/e170000.pt --decoder-learned-pos --use-final-embed --classifier-hidden-layers 8 --non-binary-cols csvs/cols/plutchik-cols.json  --save-finetune
