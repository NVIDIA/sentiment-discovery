from sklearn import metrics
import itertools
import argparse
import torch
import numpy as np
import pandas as pd
from train_utils import update_info_dict, get_metric
from collections import defaultdict
from tqdm import tqdm

def binary_threshold(args, labels=None):
    preds = pd.read_csv(args.preds_file, header=None).values
    labels = pd.read_csv(args.labels_file, header=None).values
    avg_metric, best_thresholds, category_metrics, category_best_info_dicts = _binary_threshold(preds, labels, args.metric, args.micro)
    print(avg_metric / preds.shape[1])
    np.savetxt('best_binary_thresholds_{}_{}.txt'.format('micro' if args.micro else 'macro', args.metric), (best_thresholds))


def _binary_threshold(preds, labels, metric, micro, global_tweaks=1000, debug=False):
    avg_metric = 0
    best_thresholds = []
    info_dicts = []
    category_metrics = []
    for category in range(preds.shape[1]):
        category_best_threshold = category_best_metric = 0
        for threshold in np.linspace(0.005, 1, 200):
            info_dict = update_info_dict(defaultdict(int), labels[:, category], preds[:, category], threshold=threshold)
            metric_score = get_metric(info_dict, metric, micro)
            if metric_score > category_best_metric or category_best_metric==0:
                category_best_metric, category_best_threshold, category_best_info_dict = metric_score, threshold, info_dict
        info_dicts.append(category_best_info_dict)
        category_metrics.append(category_best_metric)
        best_thresholds.append(category_best_threshold)

    # HACK -- use micro average here, even if not elsewhere
    micro = True
    best_metric = get_metric(info_dicts, metric, micro)
    # HACK: Attempt to tune thresholds simultaneously... for overall micro average
    num_categories = preds.shape[1]
    if num_categories < 2:
        global_tweaks = 0
    if debug and global_tweaks > 0:
        print('best after invididual thresholds (micro %s)' % micro)
        print(best_thresholds)
        print(get_metric(info_dicts, metric, micro))
    for i in range(global_tweaks):
        # Choose random category
        category = np.random.randint(num_categories)
        curr_threshold = best_thresholds[category]
        # tweak randomly
        new_threshold = curr_threshold + (0.08 * (np.random.random() - 0.5))
        info_dict = update_info_dict(defaultdict(int), labels[:, category], preds[:, category], threshold=new_threshold)
        old_dict = info_dicts[category]
        info_dicts[category] = info_dict
        # compute *global* metrics
        metric_score = get_metric(info_dicts, metric, micro)
        # save new threshold if global metrics improve
        if metric_score > best_metric:
            # print('Better threshold %.3f for category %d' % (new_threshold, category))
            best_thresholds[category] = round(new_threshold, 3)
            best_metric = metric_score
        else:
            info_dicts[category] = old_dict
    if debug and global_tweaks > 0:
        print('final thresholds')
        print(best_thresholds)
        print(get_metric(info_dicts, metric, micro))
    return get_metric(info_dicts, metric, micro), np.array(best_thresholds), category_metrics, info_dicts

def get_auc(args):
    preds = pd.read_csv(args.preds_file, header=None).values
    labels = pd.read_csv(args.labels_file, header=None).values.astype(int)

    aucs = []
    for category in range(preds.shape[1]):
        fpr, tpr, thresholds = metrics.roc_curve(labels[:, category], preds[:, category], pos_label=1)
        aucs.append(metrics.auc(fpr, tpr))

    for idx, auc in enumerate(aucs):
        print('{}: {}\n'.format(idx, auc))


def neutral_threshold_scalar_output(args):
    preds = pd.read_csv(args.preds_file, header=None, names=['preds'])
    labels = pd.read_csv(args.labels_file, header=None, names=['labels'])
    assert preds.shape[1] == labels.shape[1] == 1, "Neutral thresholding only available for single category labels"

    labels['positive'] = labels['labels'].apply(lambda s: int(s == 1))
    labels['negative'] = labels['labels'].apply(lambda s: int(s == 0))
    labels['neutral'] = ((labels['positive'] == labels['negative']).sum() == 2).astype(int)
    labels_vals = labels[['positive', 'negative', 'neutral']].values

    best_pos = best_neg = best_acc = 0
    for pos, neg in tqdm(itertools.product(np.linspace(0.005, 1, 200), repeat=2), total=200 ** 2, unit='setting'):
        if neg > pos:
            continue
        new_df = pd.DataFrame()
        new_df['pos'] = preds['preds'].apply(lambda s: int(s > pos))
        new_df['neg'] = preds['preds'].apply(lambda s: int(s < neg))
        new_df['neutral'] = ((new_df['pos'] == new_df['neg']).sum() == 2).astype(int)

        new_df_vals = new_df.values
        acc = 0
        for new_row, label_row in zip(new_df_vals, labels_vals):
            acc += int((new_row == label_row).sum() == 3)
        acc /= float(labels.shape[0])
        if acc > best_acc:
            best_pos, best_neg, best_acc = pos, neg, acc

    print("Best acc:", best_acc, "Best pos:", best_pos, "Best neg:", best_neg)
    np.savetxt('best_neutral_thresholds.txt', np.array([best_pos, best_neg]))


def neutral_threshold_two_output(args):
    preds = pd.read_csv(args.preds_file, header=None, names=['positive', 'negative']) # ordered positive, negative
    labels = pd.read_csv(args.labels_file, header=None, names=['positive', 'negative'])
    labels['neutral'] = labels['positive'] == labels['negative']
    labels_vals = labels.values

    best_pos = best_neg = best_acc = 0
    for pos, neg in tqdm(itertools.product(np.linspace(0.005, 1, 200), repeat=2), total=200 ** 2, unit='setting'):
        new_df = pd.DataFrame()
        new_df['pos'] = preds['positive'].apply(lambda s: int(s > pos))
        new_df['neg'] = preds['negative'].apply(lambda s: int(s > neg))
        new_df['neutral'] = (new_df['pos'] == new_df['neg']).astype(int)

        new_df_vals = new_df.values
        acc = 0
        for new_row, label_row in zip(new_df_vals, labels_vals):
            if new_row[0] == new_row[1] == 1:
                new_row[0] = new_row[1] = 0
            acc += int((new_row == label_row).sum() == 3)
        acc /= float(labels.shape[0])
        if acc > best_acc:
            best_pos, best_neg, best_acc = pos, neg, acc

    print("Best acc:", best_acc, "Best pos:", best_pos, "Best neg:", best_neg)
    np.savetxt('best_neutral_thresholds.txt', np.array([best_pos, best_neg]))

def neutral_threshold_two_output(args):
    preds = pd.read_csv(args.preds_file, header=None, names=['positive', 'negative']) # ordered positive, negative
    labels = pd.read_csv(args.labels_file, header=None, names=['positive', 'negative'])

    best_acc, (best_pos, best_neg) = _neutral_threshold_two_output(preds.values, labels.values)

    print("Best acc:", best_acc, "Best pos:", best_pos, "Best neg:", best_neg)
    np.savetxt('best_neutral_thresholds.txt', np.array([best_pos, best_neg]))


def _neutral_threshold_two_output(preds, labels, threshold_granularity=50):
    neutral_labels = (labels[:,0] == labels[:,1]).astype(int).reshape(-1, 1)
    labels_vals = np.concatenate([labels[:,:2], neutral_labels], axis=1)

    best_0 = best_1 = best_acc = 0 

    for t0, t1 in tqdm(itertools.product(np.linspace(0.005, 1, threshold_granularity), repeat=2), total=threshold_granularity ** 2, unit='setting'):
        new_df = pd.DataFrame()
        new_df['0'] = (preds[:,0]>t0).astype(int)
        new_df['1'] = (preds[:,1]>t1).astype(int)
        new_df['neutral'] = (new_df['0'] == new_df['1']).astype(int)

        new_df_vals = new_df[['0','1','neutral']].values
        acc = 0
        for new_row, label_row in zip(new_df_vals, labels_vals):
            if new_row[0] == new_row[1] == 1:
                new_row[0] = new_row[1] = 0
            acc += int((new_row == label_row).sum() == 3)
        acc /= labels_vals.shape[0]
        if acc > best_acc:
            best_0, best_1, best_acc = t0, t1, acc

    return best_acc, (best_0, best_1)


def consolidate_google_results(args):
    stem_list = [
        'p/4k-nikolai-bin',
        'p/5k1-nikolai-bin',
        'p/5k2-nikolai-bin'
    ]
    csv_df = pd.concat([pd.read_csv(s + '.csv') for s in stem_list], axis=0).reset_index(drop=True)
    print(csv_df.shape)
    npys = pd.DataFrame(np.concatenate([np.load(s + '.npy') for s in stem_list], axis=1).transpose(), columns=['google_labels', 'google_scores'])
    print(npys.shape)
    all_df = pd.concat([csv_df, npys], axis=1)
    all_df['google_scores'] = (all_df['google_scores'].astype(float) + 1) / 2
    print(all_df.isnull().sum())
    all_df = all_df.sample(frac=1)
    size = all_df.shape[0]
    train, thresh, val = all_df.iloc[:int(size * 0.7)], all_df.iloc[int(size * 0.7):int(size * 0.8)], all_df.iloc[int(size * 0.8):]
    for s in ['train', 'thresh', 'val']:
        print(eval(s).isnull().sum(), eval(s).shape)
        eval(s).to_csv('p/all-nvidia-bin-{}.csv'.format(s), index=False)

def main():
    task_dict = {
        'auc' : get_auc,
        'binary' : binary_threshold,
        'neutral' : neutral_threshold_two_output,
        'scalar' : neutral_threshold_scalar_output,
        'google' : consolidate_google_results
    }
    parser = argparse.ArgumentParser("Tools for optimizing outputs through ROC/AUC analysis")
    parser.add_argument('--task', type=str, required=True, help='what do you want to do?')
    parser.add_argument('--preds-file', type=str, help='path to predictions file')
    parser.add_argument('--labels-file', type=str, help='path to labels file')
    parser.add_argument('--metric', type=str, default='f1', help='which metric to analyze/optimize')
    parser.add_argument('--micro', action='store_true', help='whether to micro-average metric')

    args = parser.parse_args()
    task_dict[args.task](args)


if __name__ == '__main__':
    main()
