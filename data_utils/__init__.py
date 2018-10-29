import os
import math

from .samplers import BatchSampler, DistributedBatchSampler, TransposedSampler, RandomShardSampler, BatchShardSampler, DistributedBatchShardSampler
from .loaders import DataLoader, ShardLoader
from .preprocess import tokenize_str_batch, binarize_labels
from .datasets import unsupervised_dataset, json_dataset, csv_dataset, split_ds, get_processed_path
from .lazy_loader import exists_lazy, make_lazy, lazy_array_loader
from .tokenization import Tokenization, CommandToken, Tokenizer, CharacterLevelTokenizer

TRAIN_DATA = 0
VAL_DATA = 1
TEST_DATA = 2

def get_split_name(split_type=-1, proportion=1):
    name = ''
    if split_type == TRAIN_DATA:
        name += 'train'
    elif split_type == VAL_DATA:
        name += 'val'
    elif split_type == TEST_DATA:
        name += 'test'
    else:
        name += 'data'
    name += '.'+str(proportion)
    return name

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

def handle(path, text_key, label_key, preprocess=False, split=[1.], loose=False,
                binarize_sent=False, delim=',', drop_unlabeled=False, lazy=False):
    """gets a dataset and handles splitting it into train/val/test if necessary"""
    tokenizer = CharacterLevelTokenizer()
    if lazy:
        if not exists_lazy(path, data_type='data'):
            text = get_dataset(path, text_key=text_key, label_key=label_key, binarize_sent=binarize_sent,
                delim=delim, drop_unlabeled=drop_unlabeled, loose_json=loose)
            make_lazy(path, text.X, data_type='data')
        text = lazy_array_loader(path, data_type='data', map_fn=tokenizer)
    else:
        text = get_dataset(path, text_key=text_key, label_key=label_key, binarize_sent=binarize_sent,
                delim=delim, drop_unlabeled=drop_unlabeled, loose_json=loose, tokenizer=tokenizer)
    if should_split(split):
        return split_ds(text, split)
    return text


def post_process_ds(ds, seq_length, ds_type='supervised', shard_split=1., num_shards=1002, persist_state=0):
    """add caching on top of dataset. Add unsupervised wrapper last"""
    # if ds_type == 'unsupervised':
    #     shards = math.ceil(shard_split*num_shards)
    #     ds = unsupervised_dataset(ds, seq_length, persist_state=persist_state, num_shards=shards)
    return ds

def make_dataset(path, seq_length, text_key, label_key, lazy=False, preprocess=False, split=[1.],
                persist_state=0, delim=',', loose=False, binarize_sent=False, drop_unlabeled=False,
                ds_type='supervised', num_shards=1002):
    """returns dataset. returns train/val/test datasets if split is specified. returns None if split proportion is 0."""
    #if lazy:
    #    ds = handle_lazy(path, preprocess=preprocess, text_key=text_key, label_key=label_key, split=split,
    #            binarize_sent=binarize_sent, delim=delim, drop_unlabeled=drop_unlabeled, loose=loose)
    #else:
    #    ds = handle(path, preprocess=preprocess, text_key=text_key, label_key=label_key, split=split,
    #                binarize_sent=binarize_sent, delim=delim, drop_unlabeled=drop_unlabeled, loose=loose)
    ds = handle(path, preprocess=preprocess, text_key=text_key, label_key=label_key, split=split,
                binarize_sent=binarize_sent, delim=delim, drop_unlabeled=drop_unlabeled, loose=loose, lazy=lazy)

    if should_split(split):
        datasets = []
        for i, s in enumerate(split):
            d = post_process_ds(ds[i], seq_length, ds_type=ds_type, persist_state=persist_state, shard_split=s,
                        num_shards=num_shards)
            datasets.append(d)
        ds = datasets
    else:
        ds = post_process_ds(ds, seq_length, ds_type=ds_type, persist_state=persist_state, num_shards=num_shards)

    return ds
