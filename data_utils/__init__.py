import os
import math

from .samplers import BatchSampler, DistributedBatchSampler, TransposedSampler, RandomShardSampler, BatchShardSampler, DistributedBatchShardSampler
from .loaders import DataLoader, ShardLoader
from .preprocess import tokenize_str_batch, binarize_labels, process_str, process_tweet, batch_tokens
from .datasets import json_dataset, csv_dataset, split_ds, get_processed_path, ConcatDataset, SplitDataset, data_shard
from .lazy_loader import exists_lazy, make_lazy, lazy_array_loader
from .tokenization import Tokenization, CommandToken, Tokenizer, CharacterLevelTokenizer, make_tokenizer

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

def make_dataset(path, seq_length, text_key, label_key, lazy=False, process_fn=process_str, split=[1.],
                delim=',', loose=False, binarize_sent=False, drop_unlabeled=False, tokenizer=None,
                tokenizer_type='CharacterLevelTokenizer', tokenizer_model_path=None, vocab_size=None,
                model_type='bpe', pad_token=0, character_converage=1.0, non_binary_cols=None, **kwargs):
    if isinstance(process_fn, str):
        process_fn = eval(process_fn)
    if non_binary_cols is not None:
        label_key = non_binary_cols
    def get_dataset_from_path(path_):
        if lazy:
            if not exists_lazy(path_, data_type='data'):
                text = get_dataset(path_, text_key=text_key, label_key=label_key, binarize_sent=binarize_sent,
                    delim=delim, drop_unlabeled=drop_unlabeled, loose_json=loose)
                make_lazy(path_, text.X, data_type='data')
            text = lazy_array_loader(path_, data_type='data', map_fn=process_fn)
        else:
            text = get_dataset(path_, text_key=text_key, label_key=label_key, binarize_sent=binarize_sent,
                    delim=delim, drop_unlabeled=drop_unlabeled, loose_json=loose, preprocess_fn=process_fn)
        return text
    if isinstance(path, str):
        path = [path]
    datasets = [get_dataset_from_path(p) for p in path]
    if len(datasets) == 1:
        ds = datasets[0]
    else:
        ds = ConcatDataset(datasets)
    if tokenizer is None:
        tokenizer = make_tokenizer(tokenizer_type, ds, tokenizer_model_path, vocab_size, model_type, 
                                    pad_token, character_converage)
    ds.SetTokenizer(tokenizer)
    if should_split(split):
        ds = split_ds(ds, split)
    return ds, tokenizer
