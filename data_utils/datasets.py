import os
import time
from operator import itemgetter
from bisect import bisect_left, bisect_right
import json
from itertools import accumulate
import csv
import collections

import torch
from torch.utils import data
import pandas as pd
import numpy as np

from .preprocess import process_str, binarize_labels
from .lazy_loader import lazy_array_loader, exists_lazy, make_lazy
from .cache import array_cache
from .tokenization import Tokenization

PERSIST_ALL = -1
PERSIST_SHARD = 1
RESET_STATE = 0

def get_processed_path(path, text_key='text', label_key='label'):
    filepath, ext = os.path.splitext(path)
    return filepath+'.%s.%s'%(text_key, label_key)+ext

def get_load_path_and_should_process(path, text_key='text', label_key='label'):
    processed_path = get_processed_path(path, text_key, label_key)
    exists = os.path.exists(processed_path)
    if not exists:
        return path, True
    return processed_path, False

def save_preprocessed(ds, text_key='text', label_key='label'):
    processed_path = get_processed_path(ds.path, text_key, label_key)
    if not torch.distributed._initialized or torch.distributed.get_rank() == 0:
        ds.write(path=processed_path)
    return processed_path

class ConcatDataset(data.Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated.
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, **kwargs):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        self._X = None
        self._Y = None

    def SetTokenizer(self, tokenizer):
        for ds in self.datasets:
            ds.SetTokenizer(tokenizer)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def X(self):
        if self._X is None:
            self._X = []
            for data in self.datasets:
                self._X.extend(data.X)
        return self._X

    @property
    def Y(self):
        if self._Y is None:
            self._Y = []
            for data in self.datasets:
                self._Y.extend(list(data.Y))
            self._Y = np.array(self._Y)
        return self._Y

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

class SplitDataset(data.Dataset):
    """
    Dataset wrapper to access a subset of another dataset.
    Purpose: useful to index into existing datasets, possibly
    large-scale datasets as the subindexing operation is done in an
    on-the-fly manner.
    Arguments:
        ds (Dataset or array-like): List of datasets to be subindexed
        split_inds (1D array-like): List of indices part of subset
    """
    def __init__(self, ds, split_inds, **kwargs):
        self.split_inds = list(split_inds)
        self.wrapped_data = ds
        self.is_lazy = isinstance(ds, lazy_array_loader)
        if self.is_lazy:
            self.lens = itemgetter(*self.split_inds)(list(self.wrapped_data.lens))
        self._X = None
        self._Y = None

    def __len__(self):
        return len(self.split_inds)

    def __getitem__(self, index):
        return self.wrapped_data[self.split_inds[index]]

    @property
    def X(self):
        if self._X is None:
            self._X = itemgetter(*self.split_inds)(self.wrapped_data.X)
        return self._X

    @property
    def Y(self):
        if self._Y is None:
            self._Y = np.array(itemgetter(*self.split_inds)(self.wrapped_data.Y))
        return self._Y

    def __iter__(self):
        for idx in self.split_inds:
            yield self.wrapped_data[idx]

def split_ds(ds, split=[.8,.2,.0], shuffle=True):
    """
    Split a dataset into subsets given proportions of how
    much to allocate per split. If a split is 0% returns None for that split.
    Purpose: Useful for creating train/val/test splits
    Arguments:
        ds (Dataset or array-like): Data to be split.
        split (1D array-like): proportions to split `ds`. `sum(splits) != 0`
        shuffle (boolean): Randomly split dataset. Default: True
    """
    split_sum = sum(split)
    if split_sum == 0:
        raise Exception('Split cannot sum to 0.')
    split = np.array(split)
    split /= split_sum
    ds_len = len(ds)
    inds = np.arange(ds_len)
    if shuffle:
        np.random.shuffle(inds)
    start_idx = 0
    residual_idx = 0
    rtn_ds = [None]*len(split)
    for i, f in enumerate(split):
        if f != 0:
            proportion = ds_len*split[i]
            residual_idx += proportion % 1
            split_ = int(int(proportion) + residual_idx)
            split_inds = inds[start_idx:start_idx+max(split_, 1)]
            rtn_ds[i] = SplitDataset(ds, split_inds)
            start_idx += split_
            residual_idx %= 1
    return rtn_ds

class csv_dataset(data.Dataset):
    """
    Class for loading datasets from csv files.
    Purpose: Useful for loading data for unsupervised modeling or transfer tasks
    Arguments:
        path (str): Path to csv file with dataset.
        tokenizer (data_utils.Tokenizer): Tokenizer to use when processing text. Default: None
        preprocess_fn (callable): Callable that process a string into desired format.
        delim (str): delimiter for csv. Default: ','
        binarize_sent (bool): binarize label values to 0 or 1 if they\'re on a different scale. Default: False
        drop_unlabeled (bool): drop rows with unlabelled sentiment values. Always fills remaining empty
            columns with -1 (regardless if rows are dropped based on sentiment value) Default: False
        text_key (str): key to get text from csv. Default: 'sentence'
        label_key (str): key to get label from json dictionary. Default: 'label'
    Attributes:
        X (list): all strings from the csv file
        Y (np.ndarray): labels to train against
    """
    def __init__(self, path, tokenizer=None, preprocess_fn=None, delim=',',
                binarize_sent=False, drop_unlabeled=False, text_key='sentence', label_key='label',
                **kwargs):
        self.preprocess_fn = preprocess_fn
        self.tokenizer = self.SetTokenizer(tokenizer)
        self.path = path
        self.delim = delim
        self.text_key = text_key
        self.label_key = label_key
        self.drop_unlabeled = drop_unlabeled

        if '.tsv' in self.path:
            self.delim = '\t'


        self.X = []
        self.Y = []
        try:
            cols = [text_key]
            if isinstance(label_key, list):
                cols += label_key
            else:
                cols += [label_key]
            data = pd.read_csv(self.path, sep=self.delim, usecols=cols, encoding='latin-1')
        except:
            data = pd.read_csv(self.path, sep=self.delim, usecols=[text_key], encoding='latin-1')

        data = data.dropna(axis=0)

        self.X = data[text_key].values.tolist()
        try:
            self.Y = data[label_key].values
        except Exception as e:
            self.Y = np.ones(len(self.X))*-1

        if binarize_sent:
            self.Y = binarize_labels(self.Y, hard=binarize_sent)

    def SetTokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """process string and return string,label,and stringlen"""
        x = self.X[index]
        if self.tokenizer is not None:
            x = self.tokenizer.EncodeAsIds(x, self.preprocess_fn)
        elif self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
        y = self.Y[index]
        return {'text': x, 'length': len(x), 'label': y}

    def write(self, writer_gen=None, path=None, skip_header=False):
        """
        given a generator of metrics for each of the data points X_i,
            write the metrics, text, and labels to a csv file
        """
        if path is None:
            path = self.path+'.results'
        print('generating csv at ' + path)
        with open(path, 'w') as csvfile:
            c = csv.writer(csvfile, delimiter=self.delim)
            if writer_gen is not None:
                #if first item of generator is a header of what the metrics mean then write header to csv file
                if not skip_header:
                    header = (self.label_key,)+tuple(next(writer_gen))+(self.text_key,)
                    c.writerow(header)
                for i, row in enumerate(writer_gen):
                    row = (self.Y[i],)+tuple(row)+(self.X[i],)
                    c.writerow(row)
            else:
                c.writerow([self.label_key, self.text_key])
                for row in zip(self.Y, self.X):
                    c.writerow(row)

class json_dataset(data.Dataset):
    """
    Class for loading datasets from a json dump.
    Purpose: Useful for loading data for unsupervised modeling or transfer tasks
    Arguments:
        path (str): path to json file with dataset.
        tokenizer (data_utils.Tokenizer): Tokenizer to use when processing text. Default: None
        preprocess_fn (callable): callable function that process a string into desired format.
            Takes string, maxlen=None, encode=None as arguments. Default: process_str
        text_key (str): key to get text from json dictionary. Default: 'sentence'
        label_key (str): key to get label from json dictionary. Default: 'label'
    Attributes:
        all_strs (list): list of all strings from the dataset
        all_labels (list): list of all labels from the dataset (if they have it)
    """
    def __init__(self, path, tokenizer=None, preprocess_fn=process_str, binarize_sent=False,
                text_key='sentence', label_key='label', loose_json=False, **kwargs):
        self.preprocess_fn = preprocess_fn
        self.path = path
        self.tokenizer = self.SetTokenizer(tokenizer)
        self.X = []
        self.Y = []
        self.text_key = text_key
        self.label_key = label_key
        self.loose_json = loose_json

        for j in self.load_json_stream(self.path):
            s = j[text_key]
            self.X.append(s)
            self.Y.append(j[label_key])

        if binarize_sent:
            self.Y = binarize_labels(self.Y, hard=binarize_sent)

    def SetTokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        """gets the index'th string from the dataset"""
        x = self.X[index]
        if self.tokenizer is not None:
            x = self.tokenizer.EncodeAsIds(x, self.preprocess_fn)
        elif self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
        y = self.Y[index]
        return {'text': x, 'length': len(x), 'label': y}

    def __len__(self):
        return len(self.X)

    def write(self, writer_gen=None, path=None, skip_header=False):
        """
        given a generator of metrics for each of the data points X_i,
            write the metrics, text, and labels to a json file
        """
        if path is None:
            path = self.path+'.results'

        jsons = []

        if writer_gen is not None:
            #if first item of generator is a header of what the metrics mean then write header to csv file
            def gen_helper():
                keys = {}
                keys[0] = self.label_key
                if not skip_header:
                    for idx, k in enumerate(tuple(next(writer_gen))):
                        keys[idx+1] = k
                for i, row in enumerate(writer_gen):
                    if i == 0 and skip_header:
                        for idx, _ in enumerate(row):
                            keys[idx+1] = 'metric_%d'%(idx,)
                    j = {}
                    for idx, v in enumerate((self.Y[i],)+tuple(row)):
                        k = keys[idx]
                        j[k] = v
                    yield j
        else:
            def gen_helper():
                for y in self.Y:
                    j = {}
                    j[self.label_key] = y
                    yield j

        def out_stream():
            for i, j in enumerate(gen_helper()):
                j[self.text_key] = self.X[i]
                yield j

        self.save_json_stream(path, out_stream())

    def save_json_stream(self, save_path, json_stream):
        if self.loose_json:
            with open(save_path, 'w') as f:
                for i, j in enumerate(json_stream):
                    write_string = ''
                    if i != 0:
                        write_string = '\n'
                    write_string += json.dumps(j)
                    f.write(write_string)
        else:
            jsons = [j for j in json_stream]
            json.dump(jsons, open(save_path, 'w'), separators=(',', ':'))

    def load_json_stream(self, load_path):
        if not self.loose_json:
            jsons = json.load(open(load_path, 'r'))
            generator = iter(jsons)
        else:
            def gen_helper():
                with open(load_path, 'r') as f:
                    for row in f:
                        yield json.loads(row)
            generator = gen_helper()

        for j in generator:
            if self.label_key not in j:
                j[self.label_key] = -1
            yield j

class data_shard(object):
    """
    Data Shard of multiple tokenizations.
    Purpose: Useful in L2R unsupervised learning. It's stateful and on consecutive
    calls to `get` it returns the next sequence of tokens following the last 
    sequence of tokens returned.
    Arguments:
        data (Tokenization or list): data comprising the data shard. Either a Tokenization or list of Tokenizations.
        seq_len (int): sequence length to sample from shard
        persist_state (int): one of -1,0,1 specifying whether to never reset state,
            reset after every sentence, or at end of every shard. Default: 0
    Attributes:
        all_seq (list): list of all tokenizations
        seq_ends (list): cummulative lengths of `all_strs` if they were all concat'd to gether.
            `itertools.accumulate([len(s) for s in all_strs])
        total_toks (int): `seq_ends[-1]`
        num_seq (int): `len(all_seq)`
    """
    def __init__(self, data, seq_len=-1, persist_state=0, **kwargs):
        self.seq_len = seq_len
        self.persist_state = persist_state

        if isinstance(data, Tokenization):
            self.num_seq = 1
            self.all_seq = [data]
            self.seq_ends = [len(data)]
        else:
            self.num_seq = len(data)
            self.all_seq = data
            self.seq_ends = [len(self.all_seq[0])]
            for i in range(1, self.num_seq):
                s = self.all_seq[i]
                self.seq_ends.append(len(s)+self.seq_ends[-1])

        self.pad = self.all_seq[-1].pad
        self.total_toks = self.seq_ends[-1]
        self.counter = 0
        self.seq_counter = 0
        self.intra_seq_counter = 0

    def set_seq_len(self, val):
        self.seq_len = val

    def _get(self, seq_len):
        """
        Get next sequence and reset mask of `seq_len` length.
        """
        rtn_mask = []
        rtn = []
        if seq_len <= 0:
            self.counter = self.total_toks
            rtn = []
            for seq in self.all_seq:
                s = seq[:]
                rtn.extend(s)
                rtn_mask.extend(self.get_string_mask(s))
                self.seq_counter += 1
        else:
            rtn = []
            #add one to the sequence length because we need [0:seq_len] as inputs and [1:seq_len+1] as targets
            seq_len += 1
            while self.seq_counter < self.num_seq and not len(rtn) >= seq_len:
                tokenization = self.all_seq[self.seq_counter]
                num_chars = seq_len - len(rtn)
                start = self.intra_seq_counter
                end = start + num_chars
                seq = list(tokenization[start:end])
                rtn.extend(seq)
                rtn_mask.extend(self.get_string_mask(seq))
                seq_complete = len(rtn) == seq_len
                self.intra_seq_counter += len(seq)
                if self.intra_seq_counter >= len(tokenization):
                    if seq_complete:
                        # if sampled seq_len+1 tokens ends on the last token of an example do not advance intra_seq_counter as the last token will be needed for input during next sample
                        self.intra_seq_counter -= 1
                    else:
                        self.seq_counter += 1
                        self.intra_seq_counter = 0
                else:
                    self.intra_seq_counter -= 1
        return rtn, rtn_mask

    def get_string_mask(self, s):
        """
        Get hidden state reset mask for string being currently sampled.
        """
        start_mask = 0
        if self.persist_state == PERSIST_SHARD:
            start_mask = (self.seq_counter == 0 and self.intra_seq_counter == 0)
        elif self.persist_state == RESET_STATE:
            start_mask = self.intra_seq_counter == 0
        return [start_mask] + [0] * (len(s)-1)


    def get(self, seq_len=None):
        """
        Get the next sequence from the data shard as well as state reset and padding/loss masks.
        Returns a sequence of seq_len+1 so that i`nputs, targets = sequence[:-1], sequence[1:]`
        """
        if seq_len is None:
            seq_len = self.seq_len
        rtn, rtn_mask = self._get(seq_len)
        rtn_len = len(rtn)
        # returned sequence should be seq_len+1 length since it needs to contain inputs and targets
        num_padding = seq_len - (rtn_len-1)
        if num_padding > 0:
            rtn.extend([self.pad] * num_padding)
            rtn_mask.extend([0] * num_padding)
        if seq_len > 0:
            self.counter += seq_len
            # mask all padding + the last valid target token to 0 since they won't be used in loss computation
            loss_mask = [1]*(rtn_len-1) + [0]*(num_padding+1)
        else:
            self.counter = self.total_toks
            loss_mask = [1]*rtn_len
        return np.array(rtn), np.array(rtn_mask), np.array(loss_mask)

    def is_last(self):
        return self.counter >= self.total_toks-self.seq_len -1
        
    def is_done(self):
        return self.counter >= self.total_toks-1

    def __iter__(self):
        self.counter = 0
        while self.counter < self.total_toks:
            yield self.get(self.seq_len)