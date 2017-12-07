import os
from operator import itemgetter
from bisect import bisect_left
import json
from itertools import accumulate
import csv

import torch
from torch.utils import data
import pandas as pd
import numpy as np

from .preprocess import process_str
from .lazy_loader import lazy_array_loader, exists_lazy, make_lazy
from .cache import array_cache


PERSIST_ALL = -1
PERSIST_SHARD = 1
RESET_STATE = 0

NUM_TRAIN_SHARDS = 1000
NUM_VAL_SHARDS = 1
NUM_TEST_SHARDS = 1
NUM_SHARDS = NUM_TRAIN_SHARDS+NUM_VAL_SHARDS+NUM_TEST_SHARDS

def get_shard_indices(strs, shard_type='train'):
	"""get indices of shard starts for the strings"""
	num_shards = NUM_TEST_SHARDS
	if shard_type == 'train':
		num_shards = NUM_TRAIN_SHARDS
	elif shard_type == 'val':
		num_shards = NUM_VAL_SHARDS
	shard_size = len(strs)//num_shards
	inds = list(np.arange(num_shards)*shard_size)
	return set(inds)

def get_shard(strs, shard='train'):
	"""gets portion of dataset corresponding to train/test/val/etc."""
	if shard == 'train':
		return train_shard(strs)
	elif shard == 'val':
		return val_shard(strs)
	return test_shard(strs)

def train_shard(strs):
	"""gets training shards in dataset"""
	num_strs = len(strs)
	shard_size = num_strs//NUM_SHARDS
	return strs[:NUM_TRAIN_SHARDS*shard_size]

def val_shard(strs):
	"""gets validation shards in dataset"""
	num_strs = len(strs)
	shard_size = num_strs//NUM_SHARDS
	return strs[NUM_TRAIN_SHARDS*shard_size:(NUM_TRAIN_SHARDS+NUM_VAL_SHARDS)*shard_size]

def test_shard(strs):
	"""gets test shards in dataset"""
	num_strs = len(strs)
	shard_size = num_strs//NUM_SHARDS
	return strs[(NUM_TRAIN_SHARDS+NUM_VAL_SHARDS)*shard_size:NUM_SHARDS*shard_size]

class unsupervised_dataset(data.Dataset):
	"""
	class for loading dataset for unsupervised text reconstruction
	Args:
		path (str): path to json file with dataset.
		seq_len (int): path to json file with dataset.
		preprocess (bool): whether to call preprocess_fn on strings in dataset. Default: False
		preprocess_fn (callable): callable function that process a string into desired format.
			Takes string, maxlen=None, encode=None as arguments. Default: process_str
		use_cache (bool): whether or not to use an array_cache while loading the dataset.
			Useful when using Transposed Sampling to avoid cache paging. Default: False
		cache_block_size (int): number of strings to cache in one cache block. Default: 64
		cache_size (int): number of caches blocks to store before removing (based on LRU removal). Default: 32
		lazy (bool): whether to lazy evaluate dataset from disk or keep in memory. Default: False
		mem_map (bool): whether to mem_map file of lazy evaluation. Default: False
		persist_state (int): one of -1,0,1 specifying whether to never reset state,
			reset after every sentence, or at end of every shard. Default: 0
		shuffle (bool): whether to shuffle strings in dataset. Default: False
		data_file_type (str): json is the only supported type of data set rn.
		text_key (str): key to get text from json dictionary. Default: 'sentence'
		label_key (str): key to get label from json dictionary. Default: 'label'
	Attributes:
		all_strs (list): all strings from the data file
		str_ends (list): cummulative lengths of `all_strs` if they were all concat'd to gether.
			`itertools.accumulate([len(s) for s in all_strs])`
		total_chars (int): str_ends[-1]
		num_strs (int): len(all_strs)
		shard_starts (set): set containing indices of what strings
	"""
	def __init__(self, path, seq_len, preprocess=False, preprocess_fn=process_str, use_cache=False,
				cache_block_size=64, cache_size=32, lazy=False, mem_map=False, persist_state=0,
				shuffle=True, delim=',', text_key='sentence', label_key='label', shard_type='train'):
		self.path = path
		self.persist_state = persist_state
		self.lazy = lazy
		self.seq_len = seq_len
		self.all_strs = []
		self.str_ends = []
		self.total_chars = 0
		self.num_strs = 0
		self.shuffle = shuffle
		self.cache_block_size = cache_block_size
		self.cache_size = cache_size
		self.use_cache = use_cache
		self.shard_type = shard_type

		# no need to reprocess data if already processed while making training shards
		if shard_type != 'train':
			preprocess = not self.unprocessed_exists()

		#if we need to preprocess dataset, if lazy version doesn't exist then load the dataset
		if not self.lazy or not exists_lazy(self.path, self.shard_type) or preprocess:
			print('making ds')
			if os.path.splitext(path)[1] == '.json':
				self.data_file_type = 'json'
				ds = json_dataset(self.get_path(preprocess), preprocess, preprocess_fn, text_key, label_key)
			elif os.path.splitext(path)[1] == '.csv':
				self.data_file_type = 'csv'
				ds = csv_dataset(self.get_path(preprocess), preprocess, preprocess_fn,
								delim=delim, text_key=text_key, label_key=label_key)
			else:
				raise NotImplementedError('No support for other file types for unsupervised learning.')

			self.all_strs = ds.X
			if preprocess:
				self.save_processed(ds)
			# get all datashards corresponding to training set
			self.all_strs = get_shard(self.all_strs, self.shard_type)
			#shuffle strings
			if self.shuffle:
				shuffle_inds = np.arange(len(self.all_strs))
				np.random.shuffle(shuffle_inds)
				self.all_strs = [self.all_strs[idx] for idx in shuffle_inds]
			#get string endings
			self.str_ends = list(accumulate(map(len, self.all_strs)))
			self.total_chars = self.str_ends[-1]
			self.num_strs = len(self.all_strs)
			if self.lazy:
				#make lazy evaluation file
				make_lazy(self.path, self.all_strs, self.str_ends, self.shard_type)
				#free memory after lazy
				del self.all_strs
				del self.str_ends

		if self.lazy:
			#make lazy loader from lazy evaluation file
			self.all_strs = lazy_array_loader(path, self.shard_type, mem_map=mem_map)
			self.str_ends = self.all_strs.ends
			self.total_chars = self.str_ends[-1]
			self.num_strs = len(self.str_ends)

		#get indices of shard starts for the strings
		self.shard_starts = get_shard_indices(self.all_strs, self.shard_type)

		if self.use_cache:
			self.all_strs = array_cache(self.all_strs, self.cache_block_size, self.cache_size)

		if self.seq_len == -1:
			self.seq_len = self.total_chars

	def save_processed(self, ds):
		"""save processed data so that we don't have to keep reprocessing it"""
		if not torch.distributed._initialized or torch.distributed.get_rank() == 0:
			if not self.unprocessed_exists():
				os.rename(self.path, self.path+'.original')
			if self.data_file_type == 'json':
				json.dump(ds.data, open(self.path, 'w'))
			if self.data_file_type == 'csv':
				ds.write(path=self.path)
			else:
				raise NotImplementedError('Support for file types other \
					than json not implemented for unsupervised learning.')

	def get_path(self, preprocess=False):
		"""get path of the unprocessed dataset"""
		if preprocess and self.unprocessed_exists():
			return self.path+'.original'
		return self.path

	def unprocessed_exists(self):
		"""check if we've already saved a processed version of our dataset"""
		return os.path.exists(self.path+'.original')

	def __len__(self):
		if self.seq_len == self.total_chars:
			return 1
		return (self.total_chars-self.seq_len-1)//self.seq_len

	def __getitem__(self, index):
		"""
		Concatenates srings together into sequences of seq_len.
		Gets the index'th such concatenated sequence.
		"""
		#search for what string corresponds to the start of a sequence index
		str_ind = self.binary_search_strings(index)%self.num_strs
		#find what index to start from in start string
		#first find what character the previous string ended at/our string starts at
		if str_ind != 0:
			str_start_ind = self.str_ends[str_ind-1]
		else:
			str_start_ind = 0
		#subtract string start index from sequence start index
		start_char = index*self.seq_len-str_start_ind

		if self.seq_len == self.total_chars:
			rtn_strings = self.all_strs[:]
		else:
			rtn_strings = []
			#get first string of sequence
			rtn = self.all_strs[str_ind]
			rtn_strings.append(rtn)
			rtn_length = len(rtn)-start_char

			#concatenate additional strings so that this sequence reaches the maximum sequence length+1
			i = 0
			while rtn_length < self.seq_len+1:
				#increment string index
				other_str_idx = (str_ind+i+1)%self.num_strs
				s = self.all_strs[other_str_idx]
				s = s[:self.seq_len+1-rtn_length]

				rtn_strings.append(s)
				rtn_length += len(s)
				i += 1

		rtn_masks = []
		for i, s in enumerate(rtn_strings):
			rtn_masks.append(self.get_str_mask(s, str_ind+i))

		rtn_strings[0] = rtn_strings[0][start_char:start_char+self.seq_len+1]
		rtn_masks[0] = rtn_masks[0][start_char:start_char+self.seq_len+1]

		rtn = ''.join(rtn_strings)
		rtn_mask = [bit for mask in rtn_masks for bit in mask]

		return (rtn, np.array(rtn_mask))

	def binary_search_strings(self, sequence_index):
		"""binary search string endings to find which string corresponds to a specific sequence index"""
		return bisect_left(self.str_ends, sequence_index*self.seq_len)

	def get_str_mask(self, string, str_ind):
		"""get mask for when to reset state in a given sequence"""
		if self.persist_state == PERSIST_SHARD:
			rtn_mask = [int(str_ind not in self.shard_starts)]+[1]*(len(string)-1)
		elif self.persist_state == RESET_STATE:
			rtn_mask = [0]+[1]*(len(string)-1)
		else:
			rtn_mask = [1]*(len(string))
		return rtn_mask


class json_dataset(data.Dataset):
	"""
	class for loading a dataset from a json dump
	Args:
		path (str): path to json file with dataset.
		preprocess (bool): whether to call preprocess_fn on strings in dataset. Default: False
		preprocess_fn (callable): callable function that process a string into desired format.
			Takes string, maxlen=None, encode=None as arguments. Default: process_str
		text_key (str): key to get text from json dictionary. Default: 'sentence'
		label_key (str): key to get label from json dictionary. Default: 'label'
	Attributes:
		all_strs (list): list of all strings from the dataset
		all_labels (list): list of all labels from the dataset (if they have it)
	"""
	def __init__(self, path, preprocess=False, preprocess_fn=process_str,
				text_key='sentence', label_key='label'):
		self.path = path
		self.preprocess = preprocess
		self.preprocess_fn = preprocess_fn
		jsons = json.load(open(path, 'r'))
		self.X = []
		self.Y = []
		for j in jsons:
			s = j[text_key]
			if self.preprocess:
				s = self.preprocess_fn(s, maxlen=None, encode=None)
				j[text_key] = s
			self.X.append(s)
			if label_key in j:
				self.Y.append(j[label_key])
			else:
				self.Y.append(-1)
		self.data = jsons

	def __getitem__(self, index):
		"""gets the index'th string from the dataset"""
		x = self.X[index]
		y = self.Y[index] if (len(self.Y) == len(self.X)) else -1
		return self.X[index], y, len(x)

	def __len__(self):
		return len(self.X)

class csv_dataset(data.Dataset):
	"""
	class for loading dataset for sentiment transfer
	Args:
		path (str): path to csv file with dataset.
		preprocess_fn (callable): callable function that process a string into desired format.
			Takes string, maxlen=None, encode=None as arguments. Default: process_str
		delim (str): delimiter for csv. Default: False
		binarize_sent (bool): binarize sentiment values to 0 or 1 if they\'re on a different scale. Default: False
		drop_unlabeled (bool): drop rows with unlabelled sentiment values. Always fills remaining empty
			columns with 0 (regardless if rows are dropped based on sentiment value) Default: False
		text_key (str): key to get text from json dictionary. Default: 'sentence'
		label_key (str): key to get label from json dictionary. Default: 'label'
	Attributes:
		X (list): all strings from the csv file
		Y (np.ndarray): labels to train against
	"""
	def __init__(self, path, preprocess=False, preprocess_fn=process_str, delim=',',
				binarize_sent=False, drop_unlabeled=False, text_key='sentence', label_key='label'):
		self.path = path
		self.delim = delim
		self.text_key = text_key
		self.label_key = label_key
		self.drop_unlabeled = drop_unlabeled

		if drop_unlabeled:
			data = pd.read_csv(path, sep=delim, usecols=['Sentiment', text_key, label_key],
				encoding='unicode_escape')
			self.Sentiment = data['Sentiment'].values
			data = data.dropna(axis=0, subset=['Sentiment'])
		else:
			data = pd.read_csv(path, sep=delim, usecols=[text_key, label_key])

		data = data.fillna(value=0)

		self.X = data[text_key].values.tolist()
		if preprocess:
			self.X = [preprocess_fn(s, maxlen=None, encode=None) for s in self.X]
		if label_key in data:
			self.Y = data[label_key].values
			if binarize_sent:
				self.Y = ((self.Y/np.max(self.Y)) > .5).astype(int)
		else:
			self.Y = np.ones(len(self.X))*-1

	def __len__(self):
		return len(self.X)

	def __getitem__(self, index):
		"""process string and return string,label,and stringlen"""
		x = self.X[index]
		y = self.Y[index]
		return x, int(y), len(x)

	def write(self, writer_gen=None, path=None, skip_header=False):
		"""
		given a generator of metrics for each of the data points X_i, write the metrics, sentence,
			and labels to a similarly named csv file
		"""
		if path is None:
			path = self.path+'.results'
		with open(path, 'w') as csvfile:
			c = csv.writer(csvfile, delimiter=self.delim)
			if writer_gen is not None:
				#if first item of generator is a header of what the metrics mean then write header to csv file
				if not skip_header:
					header = tuple(next(writer_gen))
					header = (self.label_key,)+header+(self.text_key,)
					c.writerow(header)
				for i, row in enumerate(writer_gen):
					row = tuple(row)
					if self.drop_unlabeled:
						row = (self.Sentiment[i],)+row
					row = (self.Y[i],)+row+(self.X[i],)
					c.writerow(row)
			else:
				c.writerow([self.label_key, self.text_key])
				for row in zip(self.Y, self.X):
					c.writerow(row)

class train_val_test_ds_wrapper(data.Dataset):
	def __init__(self, ds, split_inds):
		split_inds = list(split_inds)
		self.X = itemgetter(*split_inds)(ds.X)
		self.Y = np.array(itemgetter(*split_inds)(ds.Y))

	def __len__(self):
		return len(self.X)
	def __getitem__(self, index):
		processed = self.X[index]
		return processed, int(self.Y[index]), len(processed)

def split_ds(ds, split=[.8,.2,.0]):
	"""randomly split a dataset into train/val/test given a percentage of how much to allocate to training"""
	split = np.array(split)
	split /= np.sum(split)
	ds_len = len(ds)
	inds = np.arange(ds_len)
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
			rtn_ds[i] = train_val_ds_wrapper(ds, split_inds)
			start_idx += split_	
			residual_idx %= 1
	return train_ds, val_ds
