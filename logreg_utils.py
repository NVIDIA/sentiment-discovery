###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#   
# Copyright (c) 2017, openai. All rights reserved.
###############################################################################
"""
Modified version of openai implementation https://github.com/openai/generating-reviews-discovering-sentiment/blob/master/utils.py
Modified to handle multiple classes, different metrics, thresholding, and dropping neurons.
"""
import collections

import numpy as np
from sklearn.linear_model import LogisticRegression

from metric_utils import update_info_dict, get_metric
from threshold import _binary_threshold, _neutral_threshold_two_output

def train_logreg(trX, trY, vaX=None, vaY=None, teX=None, teY=None, penalty='l1', max_iter=100,
        C=2**np.arange(-8, 1).astype(np.float), seed=42, model=None, eval_test=True, neurons=None,
        drop_neurons=False, report_metric='acc', automatic_thresholding=False, threshold_metric='acc', micro=False):
    
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
                                   'metric' : threshold_metric, 'micro' : micro}
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
                                       'metric' : threshold_metric, 'micro' : micro}
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
        train_score = get_metric(update_info_dict(blank_info_dict.copy(), trY, preds), report_metric)
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
        train_score = get_metric(info_dicts, report_metric)
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
            val_score = get_metric(update_info_dict(blank_info_dict.copy(), vaY, preds), report_metric)
        else:
            preds = []
            info_dicts = []
            for cls in range(n_classes):
                _preds = model[cls].predict_proba(vaX)[:, -1]
                info_dicts.append(update_info_dict(blank_info_dict.copy(), vaY[:, cls], _preds))
                preds.append(_preds)
            val_score = get_metric(info_dicts, report_metric)
            preds = np.concatenate([p.reshape((-1, 1)) for p in preds], axis=1)
    val_preds = preds
    val_labels = eval_labels
    scores.append(val_score * 100)
    eval_score = val_score
    threshold = np.array([.5]*n_classes)
    if automatic_thresholding:
        _, threshold, _, _ = _binary_threshold(preds.reshape(-1, n_classes), eval_labels.reshape(-1, n_classes), threshold_metric, micro)
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
        threshold = float(threshold.squeeze())
        eval_score = get_metric(update_info_dict(blank_info_dict.copy(), eval_labels, preds, threshold=threshold), report_metric)
    else:
        info_dicts = []
        for cls in range(n_classes):
            info_dicts.append(update_info_dict(blank_info_dict.copy(), eval_labels[:, cls], preds[:, cls], threshold=threshold[cls]))
        eval_score = get_metric(info_dicts, report_metric)

    scores.append(eval_score * 100)
    return model, scores, preds, c, nnotzero