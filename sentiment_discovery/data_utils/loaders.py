import collections
import sys
if sys.version_info[0] == 2:
	import Queue as queue
	string_classes = basestring
else:
	import queue
	string_classes = (str, bytes)
import threading
import traceback

import torch
from torch.utils import data
import torch.multiprocessing as multiprocessing

from .preprocess import tokenize_str_batch
from .samplers import DistributedBatchSampler, BatchSampler, TransposedSampler

_use_shared_memory = False
"""Whether to use shared memory in default_collate"""

numpy_type_map = {
	'float64': torch.DoubleTensor,
	'float32': torch.FloatTensor,
	'float16': torch.HalfTensor,
	'int64': torch.LongTensor,
	'int32': torch.IntTensor,
	'int16': torch.ShortTensor,
	'int8': torch.CharTensor,
	'uint8': torch.ByteTensor,
}

def default_collate(batch, maxlen=None, process=False):
	"""
	normal default collate except for string classes we use our own tokenize_str_batch
		function to batch strings
	"""
	"Puts each data field into a tensor with outer dimension batch size"
	if torch.is_tensor(batch[0]):
		out = None
		if _use_shared_memory:
			numel = sum([x.numel() for x in batch])
			storage = batch[0].storage()._new_shared(numel)
			out = batch[0].new(storage)
		return torch.stack(batch, 0, out=out)
	elif type(batch[0]).__module__ == 'numpy':
		elem = batch[0]
		if type(elem).__name__ == 'ndarray':
			return torch.stack([torch.from_numpy(b) for b in batch], 0)
		if elem.shape == ():
			py_type = float if elem.dtype.name.startswith('float') else int
			return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
	elif isinstance(batch[0], int):
		return torch.LongTensor(batch)
	elif isinstance(batch[0], float):
		return torch.DoubleTensor(batch)
	elif isinstance(batch[0], string_classes):
		return tokenize_str_batch(batch, rtn_maxlen=None, process=process, maxlen=maxlen)
	elif isinstance(batch[0], collections.Mapping):
		return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
	elif isinstance(batch[0], collections.Sequence):
		transposed = zip(*batch)
		return [default_collate(samples) for samples in transposed]

	raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
						.format(type(batch[0]))))

class DataLoader(data.DataLoader):
	"""normal data loader except with options for distributed data batch sampling + wrap around"""
	def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
				 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
				 transpose=False, world_size=2, rank=-1, distributed=False, wrap_last=True,
				 timeout=0, worker_init_fn=None):
		self.dataset = dataset
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.collate_fn = collate_fn
		self.pin_memory = pin_memory
		self.drop_last = drop_last
		self.timeout = timeout
		self.worker_init_fn = worker_init_fn
		if timeout < 0:
			raise ValueError('timeout option should be non-negative')

		if batch_sampler is not None:
			if batch_size > 1 or shuffle or sampler is not None or drop_last:
				raise ValueError('batch_sampler is mutually exclusive with \
									batch_size, shuffle, sampler, and drop_last')

		if sampler is not None and shuffle:
			raise ValueError('sampler is mutually exclusive with shuffle')

		if self.num_workers < 0:
			raise ValueError('num_workers cannot be negative; '
							 'use num_workers=0 to disable multiprocessing.')

		if batch_sampler is None:
			if sampler is None:
				if shuffle:
					sampler = data.sampler.RandomSampler(dataset)
				else:
					if transpose:
						sampler = TransposedSampler(dataset, batch_size)
					else:
						sampler = data.sampler.SequentialSampler(dataset)
			if distributed:
				batch_sampler = DistributedBatchSampler(sampler, batch_size, drop_last,
														world_size=world_size, rank=rank, wrap_last=wrap_last)
			else:
				batch_sampler = BatchSampler(sampler, batch_size, drop_last, wrap_last=wrap_last)

		self.sampler = sampler
		self.batch_sampler = batch_sampler
		self.last_iter = None
