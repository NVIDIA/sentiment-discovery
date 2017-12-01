import os

from .samplers import BatchSampler, DistributedBatchSampler, TransposedSampler
from .loaders import DataLoader
from .preprocess import process_str, tokenize_str_batch, tokenize_text_file
from .datasets import unsupervised_dataset, json_dataset, csv_dataset, split_ds
from .cache import array_cache
from .lazy_loader import exists_lazy, make_lazy, lazy_array_loader

def make_dataset(path, seq_length, text_key, label_key, lazy=False, preprocess=False,
				persist_state=0, cache=False, batch_size=1, delim=',', binarize_sent=False,
				drop_unlabeled=False, dataset='train', ds_type=None):

	if ds_type == 'unsupervised':
		text = unsupervised_dataset(
					path, seq_length, lazy=lazy, preprocess=preprocess, use_cache=cache,
					cache_size=batch_size, delim=delim, persist_state=persist_state, text_key=text_key,
					label_key=label_key, shard_type=dataset)

	elif ds_type == 'json' or os.path.splitext(path)[1] == '.json':
		text = json_dataset(
					path, preprocess=preprocess, text_key=text_key, label_key=label_key)

	elif ds_type == 'csv' or os.path.splitext(path)[1] == '.csv':
		text = csv_dataset(
					path, preprocess=preprocess, binarize_sent=binarize_sent, text_key=text_key,
					label_key=label_key, delim=delim, drop_unlabeled=drop_unlabeled)

	else:
		raise NotImplementedError('data_set_type %s is deprecated'%(ds_type))
	return text
