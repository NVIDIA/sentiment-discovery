import os
import math

from .samplers import BatchSampler, DistributedBatchSampler, TransposedSampler, RandomShardSampler, BatchShardSampler, DistributedBatchShardSampler
from .loaders import DataLoader, ShardLoader
from .preprocess import tokenize_str_batch, binarize_labels
from .datasets import json_dataset, csv_dataset, split_ds, get_processed_path, ConcatDataset, SplitDataset, data_shard
from .lazy_loader import exists_lazy, make_lazy, lazy_array_loader
from .tokenization import Tokenization, CommandToken, Tokenizer, CharacterLevelTokenizer

TRAIN_DATA = 0
VAL_DATA = 1
TEST_DATA = 2

def should_split(split):
    return max(split)/sum(split) != 1.

def get_ext(path):
    return os.path.splitext(path)[1]

def get_dataset(path, **kwargs):
    """gets dataset object based on keyword args and file at `path`"""
    ext = get_ext(path)
    if ext =='.json':
        text = json_dataset(path, **kwargs)
    elif ext in ['.csv', '.tsv']:
        text = csv_dataset(path, **kwargs)
    else:
        raise NotImplementedError('data file type %s is not supported'%(ext))
    return text

def make_dataset(path, seq_length, text_key, label_key, lazy=False, preprocess=False, split=[1.],
                persist_state=0, delim=',', loose=False, binarize_sent=False, drop_unlabeled=False,
                ds_type='supervised', num_shards=1002):
    tokenizer = CharacterLevelTokenizer()
    def get_dataset_from_path(path_):
        if lazy:
            if not exists_lazy(path_, data_type='data'):
                text = get_dataset(path_, text_key=text_key, label_key=label_key, binarize_sent=binarize_sent,
                    delim=delim, drop_unlabeled=drop_unlabeled, loose_json=loose)
                make_lazy(path_, text.X, data_type='data')
            text = lazy_array_loader(path_, data_type='data', map_fn=tokenizer)
        else:
            text = get_dataset(path_, text_key=text_key, label_key=label_key, binarize_sent=binarize_sent,
                    delim=delim, drop_unlabeled=drop_unlabeled, loose_json=loose, tokenizer=tokenizer)
        return text
    datasets = [get_dataset_from_path(p) for p in path]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
