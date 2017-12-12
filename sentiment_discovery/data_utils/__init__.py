import os
import math

from .samplers import BatchSampler, DistributedBatchSampler, TransposedSampler
from .loaders import DataLoader
from .preprocess import tokenize_str_batch, binarize_labels
from .datasets import unsupervised_dataset, json_dataset, csv_dataset, split_ds, get_processed_path
from .cache import array_cache
from .lazy_loader import exists_lazy, make_lazy, lazy_array_loader, get_lazy_path

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
	elif ext == '.csv':
		text = csv_dataset(path, **kwargs)
	else:
		raise NotImplementedError('data file type %s is not supported'%(ext))
	return text

def get_lazy(path, text_key, label_key, preprocess=False, data_shard='data', loose=False,
				binarize_sent=False, delim=',', drop_unlabeled=False, ds=None):
	"""gets lazy evaluated dataset associated with file at `path`. If lazy dataset is not already
		 saved, loads `path` dataset into memory and saves it for future use"""
	processed_path = get_processed_path(path, text_key, label_key)
	if not exists_lazy(processed_path, data_type=data_shard):
		if ds is None:
			ds = get_dataset(path, preprocess=preprocess, text_key=text_key, label_key=label_key,
						binarize_sent=binarize_sent, delim=delim, drop_unlabeled=drop_unlabeled, loose_json=loose)
		make_lazy(processed_path, ds.X, data_type=data_shard)
		del ds
	return lazy_array_loader(processed_path, data_type=data_shard)

def handle(path, text_key, label_key, preprocess=False, split=[1.], loose=False,
				binarize_sent=False, delim=',', drop_unlabeled=False):
	"""gets a dataset and handles splitting it into train/val/test if necessary"""
	text = get_dataset(path, preprocess=preprocess, text_key=text_key, label_key=label_key,
				binarize_sent=binarize_sent, delim=delim, drop_unlabeled=drop_unlabeled, loose_json=loose)
	if should_split(split):
		return split_ds(text, split)
	return text

def handle_lazy(path, text_key, label_key, preprocess=False, split=[1.], loose=False,
				binarize_sent=False, delim=',', drop_unlabeled=False):
	"""
	returns lazy datset, and splits it if appropriate.
	Saves split and reloads it exactly as is.
	If all parts of split do not exist when reloading then the entire dataset is reprocessed, split, and saved
	"""
	if not should_split(split):
		return get_lazy(path, text_key, label_key, preprocess=preprocess, data_shard='data',
				binarize_sent=binarize_sent, delim=delim, drop_unlabeled=drop_unlabeled, loose=loose)
	else:
		missing_split = False
		processed_path = get_processed_path(path, text_key, label_key)
		lazy_dir = get_lazy_path(processed_path)
		split_names = []
		for i, s in enumerate(split):
			data_shard = get_split_name(i, s)
			split_names.append(data_shard)
			if s != 0:
				missing_split = missing_split or not exists_lazy(processed_path, data_type=data_shard)

		rtn_ds = []
		if missing_split:
			files_to_remove = [os.path.join(lazy_dir, f) for f in os.listdir()]
			for f, filename in enumerate(files_to_remove):
				try:
					os.remove(filename)
				except:
					pass
			ds = handle(path, preprocess=preprocess, text_key=text_key, label_key=label_key, split=split,
						binarize_sent=binarize_sent, delim=delim, drop_unlabeled=drop_unlabeled, loose=loose)
			for d, data_set in enumerate(ds):
				lazy_ds = None
				if data_set is not None:
					data_shard = split_names[d]
					lazy_ds = get_lazy(path, text_key, label_key, preprocess=preprocess, data_shard=data_shard,
						binarize_sent=binarize_sent, delim=delim, drop_unlabeled=drop_unlabeled, ds=data_set)
				rtn_ds.append(lazy_ds)
		else:
			for i, data_shard in enumerate(split_names):
				p = split[i]
				lazy_ds = None
				if p != 0:
					data_shard = split_names[i]
					lazy_ds = get_lazy(path, text_key, label_key, preprocess=preprocess, data_shard=data_shard,
						binarize_sent=binarize_sent, delim=delim, drop_unlabeled=drop_unlabeled)
				rtn_ds.append(lazy_ds)
		return rtn_ds


def post_process_ds(ds, seq_length, cache=False, cache_size=1, cache_block_size=64,
					ds_type='supervised', shard_split=1., num_shards=1002, persist_state=0):
	"""add caching on top of dataset. Add unsupervised wrapper last"""
	if cache:
		ds = array_cache(ds, cache_block_size=cache_block_size, cache_size=cache_size)
	if ds_type == 'unsupervised':
		shards = math.ceil(shard_split*num_shards)
		ds = unsupervised_dataset(ds, seq_length, persist_state=persist_state, num_shards=shards)
	return ds

def make_dataset(path, seq_length, text_key, label_key, lazy=False, preprocess=False, split=[1.],
				persist_state=0, cache=False, cache_size=1, cache_block_size=64, delim=',', loose=False,
				binarize_sent=False, drop_unlabeled=False, ds_type='supervised', num_shards=1002):
	"""returns dataset. returns train/val/test datasets if split is specified. returns None if split proportion is 0."""
	if lazy:
		ds = handle_lazy(path, preprocess=preprocess, text_key=text_key, label_key=label_key, split=split,
				binarize_sent=binarize_sent, delim=delim, drop_unlabeled=drop_unlabeled, loose=loose)
	else:
		ds = handle(path, preprocess=preprocess, text_key=text_key, label_key=label_key, split=split,
					binarize_sent=binarize_sent, delim=delim, drop_unlabeled=drop_unlabeled, loose=loose)

	if should_split(split):
		datasets = []
		for i, s in enumerate(split):
			d = post_process_ds(ds[i], seq_length, ds_type=ds_type, cache=cache, cache_size=cache_size,
						cache_block_size=cache_block_size, persist_state=persist_state, shard_split=s,
						num_shards=num_shards)
			datasets.append(d)
		ds = datasets
	else:
		ds = post_process_ds(ds, seq_length, ds_type=ds_type, cache=cache, cache_size=cache_size,
						cache_block_size=cache_block_size, persist_state=persist_state, num_shards=num_shards)

	return ds
