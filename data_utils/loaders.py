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
import math
import time

import torch
from torch.utils import data
import torch.multiprocessing as multiprocessing

import numpy as np

from .preprocess import tokenize_str_batch, batch_tokens
from .samplers import DistributedBatchSampler, BatchSampler, TransposedSampler, RandomShardSampler, DistributedBatchShardSampler, BatchShardSampler
from .tokenization import Tokenization

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

samples = []

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
    elif isinstance(batch[0], Tokenization):
        pad = batch[0].pad
        tokenization, text, original_text = zip(*([(tokenization.tokenization, tokenization.text, tokenization.original_text) for tokenization in batch]))
        return [batch_tokens(tokenization, fill_value=pad)[0], text, original_text]
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

def pin_memory_batch(batch):
    if isinstance(batch, torch.Tensor):
        return batch.pin_memory()
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: pin_memory_batch(sample) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [pin_memory_batch(sample) for sample in batch]
    else:
        return batch

class DataLoader(data.DataLoader):
    """normal data loader except with options for distributed data batch sampling + wrap around"""
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 transpose=False, world_size=2, rank=-1, distributed=False, wrap_last=False,
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

class ShardLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 transpose=False, world_size=2, rank=-1, distributed=False, wrap_last=False,
                 timeout=0, worker_init_fn=None, seq_len=-1, persist_state=0, samples_per_shard=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.persist_state = persist_state
        self.samples_per_shard = samples_per_shard
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

        self.distributed=distributed
        self.world_size=world_size
        self.rank=rank
        if self.distributed:
            self.batch_size = math.ceil(self.batch_size/self.world_size)

        if batch_sampler is None:
            if sampler is None:
                sampler = RandomShardSampler(self.dataset, self.samples_per_shard, self.seq_len, self.persist_state)
            if self.distributed:
                batch_sampler = DistributedBatchShardSampler(sampler, self.batch_size, self.drop_last, world_size=self.world_size, rank=self.rank)
            else:
                batch_sampler = BatchShardSampler(sampler, self.batch_size, self.drop_last)
        else:
            sampler = batch_sampler.sampler 

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.last_iter = None


    def set_seq_len(self, seq_len):
        self.seq_len = seq_len
        self.batch_sampler.set_seq_len(seq_len)

    def set_samples_per_shard(self, samples_per_shard):
        self.samples_per_shard = samples_per_shard
        self.batch_sampler.set_samples_per_shard(samples_per_shard)

    def set_persist_state(self, persist_state):
        self.persist_state = persist_state
        self.batch_sampler.set_persist_state(persist_state)

    def __len__(self):
        return len(self.batch_sampler)/self.batch_size

    def __iter__(self):
        return _ShardLoaderIter(self)

class _ShardLoaderIter(object):
    def __init__(self, shardloader):
        self.shardloader = shardloader
        self.num_workers = self.shardloader.num_workers
        self.batch_sampler = self.shardloader.batch_sampler
        self.collate_fn = self.shardloader.collate_fn
        self.pin_memory = self.shardloader.pin_memory
        self.batch_size = self.batch_sampler.batch_size
        self.timeout = self.shardloader.timeout
        if self.num_workers == 0:
            self.queue_manager = (q for q in self.batch_sampler.manage_queues())
        else:
            self.queue_manager = _ShardLoaderManager(self.batch_sampler, self.num_workers, self.collate_fn, self.pin_memory, self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_workers == 0:
            return self.collate_fn(next(self.queue_manager))
        else:
            return next(self.queue_manager)

MP_STATUS_CHECK_INTERVAL = 5.0

class _ShardLoaderManager(object):
    def __init__(self, batch_sampler, num_workers, collate_fn, pin_memory=False, timeout=False):
        self.batch_sampler = batch_sampler
        self.batch_size = self.batch_sampler.batch_size
        self.num_workers = num_workers
        self.queue_size = num_workers*2
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.timeout = timeout

        self.data_queues = []
        self.workers = []

        indices_per_worker = self.batch_size // self.num_workers
        all_indices = list(range(self.batch_size))
        for i in range(num_workers):
            data_queue = multiprocessing.Queue(self.queue_size)
            self.data_queues.append(data_queue)
            batch_indices = all_indices[indices_per_worker*i:indices_per_worker*(i+1)]
            w = multiprocessing.Process(target=self.batch_sampler.manage_queues_multiproc,
                                        args=(batch_indices, data_queue))
            w.daemon = True
            w.start()
            self.workers.append(w)

        self.output_queue = queue.Queue(self.queue_size)
        cur_device = -1
        if torch.cuda.is_available():
            cur_device = torch.cuda.current_device()
        self.output_thread = threading.Thread(target=_shardloader_pin_memory_loop,
                                              args=(self.output_queue, self.data_queues,
                                                    self.collate_fn, self.pin_memory,
                                                    cur_device))
        self.output_thread.daemon = True
        self.output_thread.start()

    def __iter__(self):
        return self

    def _get_batch(self):
        # In the non-timeout case, worker exit is covered by SIGCHLD handler.
        # But if `pin_memory=True`, we still need account for the possibility
        # that `pin_memory_thread` dies.
        if self.timeout > 0:
            try:
                return self.output_queue.get(timeout=self.timeout)
            except queue.Empty:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self.timeout))
        elif self.pin_memory:
            while self.output_thread.is_alive():
                try:
                    return self.output_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
                except queue.Empty:
                    continue
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError('Pin memory thread exited unexpectedly')
            # In this case, `self.data_queue` is a `queue.Queue`,. But we don't
            # need to call `.task_done()` because we don't use `.join()`.
        else:
            return self.output_queue.get(block=True)

    def __next__(self):
        return self._get_batch()


def _shardloader_pin_memory_loop(output_queue, data_queues, collate_fn, pin_memory=False, device_id=-1, timeout=0):
    queue_results = [list() for _ in data_queues]
    output_queue_len = output_queue.maxsize
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    while True:
        for i, data_queue in enumerate(data_queues):
            try:
                res = data_queue.get_nowait()
                queue_results[i].append(res)
            except queue.Empty:
                continue
        if sum(len(q)>=1 for q in queue_results) >= len(data_queues):
            batch = []
            for q in queue_results:
                batch.extend(q.pop(0))
            batch = collate_fn(batch)
            if pin_memory:
                batch = pin_memory_batch(batch)
            output_queue.put(batch, block=True)
