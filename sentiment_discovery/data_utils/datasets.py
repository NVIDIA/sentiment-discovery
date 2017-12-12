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

from .preprocess import process_str, binarize_labels
from .lazy_loader import lazy_array_loader, exists_lazy, make_lazy
from .cache import array_cache


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

class train_val_test_wrapper(data.Dataset):
	def __init__(self, ds, split_inds):
		self.split_inds = list(split_inds)
		self.wrapped_data = ds
		self.X = itemgetter(*split_inds)(ds.X)
		self.Y = np.array(itemgetter(*split_inds)(ds.Y))

	def __len__(self):
		return len(self.X)
	def __getitem__(self, index):
		processed = self.X[index]
		return processed, int(self.Y[index]), len(processed)

def split_ds(ds, split=[.8,.2,.0], shuffle=True):
	"""randomly split a dataset into train/val/test given a percentage of how much to allocate to training"""
	split = np.array(split)
	split /= np.sum(split)
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
			rtn_ds[i] = train_val_test_wrapper(ds, split_inds)
			start_idx += split_	
			residual_idx %= 1
	return rtn_ds

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
				binarize_sent=False, drop_unlabeled=False, text_key='sentence', label_key='label',
				**kwargs):
		self.processed_path = self.path = path
		self.path = path
		self.delim = delim
		self.text_key = text_key
		self.label_key = label_key
		self.drop_unlabeled = drop_unlabeled

		load_path, should_process = get_load_path_and_should_process(self.path, text_key, label_key)
		should_process = should_process or preprocess

		if drop_unlabeled:
			data = pd.read_csv(load_path, sep=delim, usecols=['Sentiment', text_key, label_key],
				encoding='unicode_escape')
			self.Sentiment = data['Sentiment'].values
			data = data.dropna(axis=0, subset=['Sentiment'])
		else:
			data = pd.read_csv(load_path, sep=delim, usecols=[text_key, label_key])

		data = data.fillna(value=-1)

		self.X = data[text_key].values.tolist()
		if should_process:
			self.X = [preprocess_fn(s, maxlen=None, encode=None) for s in self.X]
		if label_key in data:
			self.Y = data[label_key].values
		else:
			self.Y = np.ones(len(self.X))*-1

		if should_process:
			self.processed_path = save_preprocessed(self, text_key=text_key, label_key=label_key)
		else:
			self.processed_path = load_path

		if binarize_sent:
			self.Y = binarize_labels(self.Y, hard=binarize_sent)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, index):
		"""process string and return string,label,and stringlen"""
		x = self.X[index]
		y = self.Y[index]
		return x, int(y), len(x)

	def write(self, writer_gen=None, path=None, skip_header=False):
		"""
		given a generator of metrics for each of the data points X_i,
			write the metrics, text, and labels to a csv file
		"""
		if path is None:
			path = self.path+'.results'
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
	def __init__(self, path, preprocess=False, preprocess_fn=process_str, binarize_sent=False,
				text_key='sentence', label_key='label', loose_json=False, **kwargs):
		self.processed_path = self.path = path
		self.preprocess_fn = preprocess_fn
		self.X = []
		self.Y = []
		self.text_key = text_key
		self.label_key = label_key
		self.loose_json = loose_json

		load_path, should_process = get_load_path_and_should_process(self.path, text_key, label_key)
		should_process = should_process or preprocess

		for j in self.load_json_stream(load_path):
			s = j[text_key]
			if should_process:
				s = self.preprocess_fn(s, maxlen=None, encode=None)
				j[text_key] = s
			self.X.append(s)
			self.Y.append(j[label_key])

		if should_process:
			self.processed_path = save_preprocessed(self, text_key=text_key, label_key=label_key)
		else:
			self.processed_path = load_path

		if binarize_sent:
			self.Y = binarize_labels(self.Y, hard=binarize_sent)

	def __getitem__(self, index):
		"""gets the index'th string from the dataset"""
		x = self.X[index]
		y = self.Y[index]
		return self.X[index], y, len(x)

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

def get_shard_indices(num_strs, num_shards=1000):
	shard_size = num_strs//num_shards
	inds = list(np.arange(num_shards)*shard_size)
	return set(inds)

class unsupervised_dataset(data.Dataset):
	"""
	class for loading dataset for unsupervised text reconstruction
	Args:
		path (str): instance of a dataset.
		seq_len (int): path to json file with dataset.
		persist_state (int): one of -1,0,1 specifying whether to never reset state,
			reset after every sentence, or at end of every shard. Default: 0
		num_shards (int): number of shards to split dataset into.
	Attributes:
		all_strs (list): all strings from the data file
		str_ends (list): cummulative lengths of `all_strs` if they were all concat'd to gether.
			`itertools.accumulate([len(s) for s in all_strs])`
		total_chars (int): str_ends[-1]
		num_strs (int): len(all_strs)
		shard_starts (set): set containing indices of what strings
	"""
	def __init__(self, ds, seq_len=256, persist_state=PERSIST_SHARD, num_shards=1):
		# self.path = path
		self.persist_state = persist_state
		# self.lazy = lazy
		if isinstance(ds, lazy_array_loader):
			self.all_strs = ds
			self.str_ends = ds.ends
		else:
			self.all_strs = ds.X
			self.str_ends = list(accumulate(map(len, self.all_strs)))
		self.total_chars = self.str_ends[-1]
		self.num_strs = len(self.all_strs)

		#get indices of shard starts for the strings
		self.set_num_shards(num_shards)

		self.set_seq_len(seq_len)

	def set_seq_len(self, seq_len):
		self.seq_len = seq_len
		assert self.seq_len != 0
		if self.seq_len < 0:
			self.seq_len = (self.total_chars-1)//(-self.seq_len)

	def set_num_shards(self, num_shards):
		self.shard_starts = get_shard_indices(self.num_strs, num_shards)

	def __len__(self):
		if self.seq_len >= self.total_chars-1:
			return 1
		return (self.total_chars-1)//self.seq_len

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

		if self.seq_len >= self.total_chars-1:
			rtn_strings = list(self.all_strs)
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
