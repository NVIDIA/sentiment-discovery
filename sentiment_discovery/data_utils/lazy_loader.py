import os
import mmap
import pickle as pkl
import time
from itertools import accumulate

import torch

def get_lazy_path(path):
	"""gets path where lazy evaluation file are stored"""
	return os.path.splitext(path)[0]+'.lazy'

def exists_lazy(path, data_type='data'):
	"""check if we've already made a lazy version of this file"""
	if not os.path.exists(get_lazy_path(path)):
		return False
	contents = os.listdir(get_lazy_path(path))
	if data_type not in contents:
		return False
	if data_type+'.len.pkl' not in contents:
		return False
	return True

def make_lazy(path, strs, data_type='data'):
	"""make lazy version of file"""
	lazypath = get_lazy_path(path)
	if not os.path.exists(lazypath):
		os.makedirs(lazypath)
	datapath = os.path.join(lazypath, data_type)
	lenpath = os.path.join(lazypath, data_type+'.len.pkl')
	if not torch.distributed._initialized or torch.distributed.get_rank() == 0:
		with open(datapath, 'w') as f:
			f.write(''.join(strs))
		str_ends = list(accumulate(map(len, strs)))
		pkl.dump(str_ends, open(lenpath, 'wb'))
	else:
		while not os.path.exists(lenpath):
			time.sleep(1)

def split_strings(strings, start, chr_lens):
	"""split strings based on string lengths and given start"""
	return [strings[i-start:j-start] for i, j in zip([start]+chr_lens[:-1], chr_lens)]

class lazy_array_loader(object):
	"""
	Arguments:
		path: path to directory where array entries are concatenated into one big string file
			and the .len file are located
		data_type: one of 'train', 'val', 'test'
		mem_map: boolean specifying whether to memory map file `path`
	"""
	def __init__(self, path, data_type='data', mem_map=False):
		lazypath = get_lazy_path(path)
		datapath = os.path.join(lazypath, data_type)
		#get file where array entries are concatenated into one big string
		self._file = open(datapath, 'r')
		self.file = self._file
		#memory map file if necessary
		self.mem_map = mem_map
		if self.mem_map:
			self.file = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
		lenpath = os.path.join(lazypath, data_type+'.len.pkl')
		self.ends = pkl.load(open(lenpath, 'rb'))

	def __getitem__(self, index):
		"""read file and splice strings based on string ending array `ends` """
		if not isinstance(index, slice):
			if index == 0:
				start = 0
			else:
				start = self.ends[index-1]
			end = self.ends[index]
			return self.file_read(start, end)
		else:
			chr_lens = self.ends[index]
			if index.start == 0 or index.start is None:
				start = 0
			else:
				start = self.ends[index.start-1]
			stop = chr_lens[-1]
			strings = self.file_read(start, stop)
			return split_strings(strings, start, chr_lens)

	def __len__(self):
		return len(self.ends)

	def file_read(self, start=0, end=None):
		"""read specified portion of file"""
		#Seek to start of file read
		self.file.seek(start)
		#read to end of file if no end point provided
		if end is None:
			rtn = self.file.read()
		#else read amount needed to reach end point
		else:
			rtn = self.file.read(end-start)
		#TODO: @raulp figure out mem map byte string bug
		#if mem map'd need to decode byte string to string
		if self.mem_map:
			rtn = rtn.decode('unicode_escape')
		return rtn
