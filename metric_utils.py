import torch
import itertools

# At pain of messing up a good thing, also collect standard deviation (total) -- divided by total items for average
def update_info_dict(info_dict, labels, preds, threshold=0.5, std=None):
    preds = (torch.tensor(preds) > threshold).long()
    labels = (torch.tensor(labels) > threshold).long()
    # For backward compatibility -- if no std, assume it's zero -- and put it on CUDA if needed
    if std is not None:
        info_dict['std'] += torch.sum(torch.tensor(std)).float()
    else:
        info_dict['std'] += torch.sum((preds == 1) & (preds == 0)).float()
    info_dict['tp'] += torch.sum((preds == 1) & (labels == 1)).float()
    info_dict['tn'] += torch.sum((preds == 0) & (labels == 0)).float()
    info_dict['fp'] += torch.sum((preds == 1) & (labels == 0)).float()
    info_dict['fn'] += torch.sum((preds == 0) & (labels == 1)).float()
    return info_dict

# Mis-nomer -- returns standard deviation per class.
def get_variance(tp, tn, fp, fn, std):
    total = tp + tn + fp + fn
    return std / total

# TODO: Also return variance per class (in multihead sense) as a metric
def get_metric(infos, metric=None, micro=False):
    """Essentially a case-switch for getting a metric"""
    metrics = {
        'acc'  : get_accuracy,
        'jacc' : get_jaccard_index,
        'f1'   : get_f1,
        'mcc'  : get_mcc,
        'recall': get_recall,
        'precision': get_precision,
        'var'  : get_variance
    }
    tp = tn = fp = fn = std = 0
    if isinstance(infos, dict):
        infos = [infos]
    metric = metrics[infos[0].get('metric') or metric]
    micro = infos[0].get('micro') or micro
    stats = ['tp', 'tn', 'fp', 'fn', 'std']

    if micro:
        # micro averaging computes the metric after aggregating
        # all of the parameters from sets being averaged
        for info in infos:
            tp += info['tp']
            tn += info['tn']
            fp += info['fp']
            fn += info['fn']
            std += info['std']
        return metric(tp, tn, fp, fn, std)
    else:
        # macro averaging computes the metric on each set
        # and averages the metrics afterward
        individual_metrics = []
        for info in infos:
            individual_metrics.append(metric(*[info[s].item() for s in stats]))
        return sum(individual_metrics) / len(individual_metrics)

# Metrics as functions of true positive, true negative,
# false positive, false negative, standard deviation
def get_precision(tp, tn, fp, fn, std):
    if tp == 0:
        return 0
    return tp / (tp + fp)

def get_recall(tp, tn, fp, fn, std):
    if tp == 0:
        return 0
    return tp / (tp + fn)

def get_jaccard_index(tp, tn, fp, fn, std):
    if tp == 0:
        return 0
    return (tp) / (tp + fp + fn)

def get_accuracy(tp, tn, fp, fn, std):
    return (tp + tn) / (tp + tn + fp + fn)

def get_f1(tp, tn, fp, fn, std):
    if tp == 0:
        return 0
    return 2.0 * tp / (2 * tp + fp + fn)

def get_mcc(tp, tn, fp, fn, std):
    total = (tp + tn + fp + fn)
    for v in tp, tn, fp, fn:
        v /= total
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    denom = denom if denom > 1e-8 else 1
    return (tp * tn - fp * fn) / denom

