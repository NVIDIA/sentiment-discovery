import os

from .samplers import BatchSampler, DistributedBatchSampler, TransposedSampler
from .loaders import DataLoader
from .preprocess import process_str, tokenize_str_batch, binarize_labels
from .datasets import unsupervised_dataset, json_dataset, csv_dataset, split_ds
from .cache import array_cache
from .lazy_loader import exists_lazy, make_lazy, lazy_array_loader

def get_ext(path):
	return os.path.splitext(path)[1]

def get_processed_path(path):
	pass

def get_dataset(path, **kwargs):
	ext = get_ext(path)
	if ext =='.json':
		text = json_dataset(path, **kwargs)
	elif ext == '.csv':
		text = csv_dataset(path, **kwargs)
	else:
		raise NotImplementedError('data file type %s is not supported'%(ext))
	return text

def make_dataset(path, seq_length, text_key, label_key, lazy=False, preprocess=False,
				persist_state=0, cache=False, batch_size=1, delim=',', binarize_sent=False,
				drop_unlabeled=False, dataset='train', ds_type=None):

	if ds_type == 'unsupervised':
		text = unsupervised_dataset(
					path, seq_length, lazy=lazy, preprocess=preprocess, use_cache=cache,
					cache_size=batch_size, delim=delim, persist_state=persist_state, text_key=text_key,
					label_key=label_key, shard_type=dataset)
	else:
		text = get_dataset(path, preprocess=preprocess, text_key=text_key, label_key=label_key,
					binarize_sent=binarize_sent, delim=delim, drop_unlabeled=drop_unlabeled)
	return text
