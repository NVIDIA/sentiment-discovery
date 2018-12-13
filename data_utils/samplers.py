import math
import os
import sys

import torch
from torch.utils import data
import numpy as np

from .datasets import data_shard

class DistributedBatchSampler(data.sampler.BatchSampler):
    """
    similar to normal implementation of distributed batch sampler, except if sampler is transposed sampler
    has option to wrap around instead of not dropping last half batch. This is useful for persisting state
    """
    def __init__(self, sampler, batch_size, drop_last, rank=-1, world_size=2, wrap_last=False):
        super(DistributedBatchSampler, self).__init__(sampler, batch_size, drop_last)
        if rank == -1:
            rank = torch.distributed.get_rank()
        self.rank = rank
        self.world_size = world_size
        self.sampler.wrap_around = 0
        self.wrap_around = 0
        self.wrap_last = wrap_last
        self.start_iter = 0

    def __iter__(self):
        batch = []
        last_batch = None
        i = 0
        for idx in self.data_iterator(self.sampler, wrap_around=False):
            batch.append(idx)
            if len(batch) == self.batch_size:
                tbatch = self._batch(batch)
                if i >= self.start_iter:
                    yield tbatch
                    self.start_iter = 0
                i += 1
                last_batch = np.array(list(tbatch))
                batch = []
        batch_len = len(batch)
        if batch_len > 0 and not self.drop_last:
            if self.wrap_last:
                self.sampler.wrap_around -= (self.batch_size)
                self.wrap_around += (len(batch))
                self.wrap_around %= self.batch_size
                if isinstance(self.sampler, TransposedSampler):
                    for i, idx in enumerate(self.data_iterator(self.sampler, wrap_around=True)):
                        if i == 0:
                            continue
                        batch.append(idx)
                        new_batch_len = len(batch)
                        if len(batch) == self.batch_size:
                            break
            yield self._batch(batch)
        if self.wrap_last:
            self.sampler.wrap_around += self.batch_size

    def data_iterator(self, _iter, wrap_around=False):
        """iterates through data and handles wrap around"""
        for i, idx in enumerate(_iter):
            if i < self.wrap_around%self.batch_size:
                continue
            if wrap_around:
                self.wrap_around += 1
                self.wrap_around %= self.batch_size
            yield idx

    def _batch(self, batch):
        """extracts samples only pertaining to this worker's batch"""
        start = self.rank*self.batch_size//self.world_size
        end = (self.rank+1)*self.batch_size//self.world_size
        return batch[start:end]

class BatchSampler(data.sampler.BatchSampler):
    """
    Normal implementation of batch sampler, except if sampler is transposed sampler it
    has option to wrap around instead of not dropping last half batch.
    Useful for persisting state.
    """
    def __init__(self, sampler, batch_size, drop_last, wrap_last=False):
        super(BatchSampler, self).__init__(sampler, batch_size, drop_last)
        self.wrap_around = 0
        self.sampler.wrap_around = 0
        self.wrap_last = wrap_last
        self.start_iter = 0

    def __iter__(self):
        batch = []
        last_batch = None
        i = 0
        for idx in self.data_iterator(self.sampler, wrap_around=False):
            batch.append(idx)
            new_batch_len = len(batch)
            if new_batch_len == self.batch_size:
                if i >= self.start_iter:
                    yield batch
                    self.start_iter = 0
                i += 1
                last_batch = np.array(list(batch))
                batch = []

        if len(batch) > 0 and (self.wrap_last or not self.drop_last):
            if self.wrap_last:
                self.sampler.wrap_around -= (self.batch_size)
                self.wrap_around += (len(batch))
                self.wrap_around %= self.batch_size
                if isinstance(self.sampler, TransposedSampler):
                    for i, idx in enumerate(self.data_iterator(self.sampler, wrap_around=True)):
                        if i == 0:
                            continue
                        batch.append(idx)
                        if len(batch) == self.batch_size:
                            break
            yield batch
        if self.wrap_last:
            self.sampler.wrap_around += self.batch_size

    def data_iterator(self, _iter, wrap_around=False):
        """iterates through data and handles wrap around"""
        for i, idx in enumerate(_iter):
            if i < self.wrap_around%self.batch_size:
                continue

            if wrap_around:
                self.wrap_around += 1
                self.wrap_around %= self.batch_size
            yield idx

class TransposedSampler(data.sampler.Sampler):
    """
    Instead of performing sequential sampling, samples array in a transposed fashion given the
    batch size to sampled. Instead of generating the following indices for a batch size of 2
        1 3 5
        2 4 6
    It will generate
        1 2 3
        4 5 6
    """
    def __init__(self, data_source, batch_size, data_sampler=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.len_ds = len(data_source)
        self.strat_width = self.len_ds//batch_size
        #self.strat_width = math.ceil(self.len_ds/batch_size)
        self.data_sampler = data_sampler
        self.wrap_around = 0

    def transpose_helper(self, x):
        """computes index corrseponding to transpose of index x"""
        return ((x%self.batch_size)*self.strat_width+(x//self.batch_size))%self.len_ds
        x += self.wrap_around
        return ((x%self.batch_size)*self.strat_width+(x//self.batch_size))%self.len_ds

    def __iter__(self):
        if self.data_sampler is None:
            return iter(map(self.transpose_helper, range(len(self))))
        return iter(map(self.transpose_helper, iter(self.data_sampler)))

    def __len__(self):
        #return self.len_ds
        return self.strat_width*self.batch_size


class RandomShardSampler(object):
    """
    Sampler for data shards.
    Purpose: Samples data shards used for L2R unsupervised modeling from the `data_source`.
    Arguments:
        data_source (Dataset or array-like): Dataset of tokenizations to sample data from.
        samples_per_shard (int): Number of samples per shard to gather from `data_source`.
        seq_len (int): seq_len value to use when creating a data shard. Can be reset later with
            `set_seq_len`.
        persist_state (int): persist_state value to use when creating a data shard. See 
            data_utils.data_shard documentation for valid values. Can be reset later with 
            `set_persist_state`.
        random_state (np.RandomState): Random number generator state to use for sampling data. If
            no value is supplied it uses numpy's default random state (not thread safe).
    """

    def __init__(self, data_source, samples_per_shard, seq_len=-1, persist_state=0):
        self.data_source = data_source
        self.source_size = len(data_source)
        self.samples_per_shard = samples_per_shard
        self.seq_len = seq_len
        self.persist_state = persist_state

    def set_seq_len(self, seq_len):
        self.seq_len = seq_len

    def set_samples_per_shard(self, samples_per_shard):
        self.samples_per_shard = samples_per_shard

    def set_persist_state(self, persist_state):
        self.persist_state = persist_state

    def get(self, random_state, samples_per_shard=None):
        """
        Uses either supplied random state or default random state to sample data from 
        the data source, create a datashard, and return it.
        """
        if samples_per_shard is None:
            samples_per_shard = self.samples_per_shard
        sample_ids = random_state.randint(self.source_size, size=samples_per_shard)
        samples = [self.data_source[i] for i in sample_ids]
        samples = [sample['text'] if isinstance(sample, dict) else sample for sample in samples]
        return data_shard(samples, self.seq_len, self.persist_state)


    def __len__(self):
        return self.source_size


class BatchShardSampler(object):
    """
    Class to manage the random state of and sample a batch of active shards.
    Uses one random state per batch index to control sampling of data shards for that batch index.
    Purpose: Intended for use with data_utils.ShardLoader to perform L2R unsupervised Learning.
    Arguments: 
        shard_sampler (RandomShardSampler): shard sampler used to sample data shards.
        batch_size (int): Batch size to sample.
        drop_last (boolean): Pretty much useless. Used to give a fake length.
        random_batch (list): List of random states to use.
    Attributes:
        batch (list): Batch of shard queues (a list that contains shards). Call `.get` and 
            `.isdone()` on `shard_queue[0]` to get next batch and check if shard is done.
    """
    def __init__(self, shard_sampler, batch_size, drop_last, random_batch=None):
        self.shard_sampler = shard_sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        # self.batch = None
        self.random_batch = random_batch
        if self.random_batch is None:
            self.random_batch = [np.random.RandomState(seed) for seed in np.random.randint(batch_size*999, size=batch_size)]

    def set_seq_len(self, seq_len):
        self.seq_len = seq_len
        self.shard_sampler.set_seq_len(seq_len)

    def set_samples_per_shard(self, samples_per_shard):
        self.samples_per_shard = samples_per_shard
        self.shard_sampler.set_samples_per_shard(samples_per_shard)

    def set_persist_state(self, persist_state):
        self.persist_state = persist_state
        self.shard_sampler.set_persist_state(persist_state)

    def get_shard(self, b):
        return self.shard_sampler.get(random_state=self.random_batch[b])

    def iter_queue(self, b):
        live_shard = self.get_shard(b)
        while True:
            if live_shard.is_done():
                live_shard = self.get_shard(b)
            yield live_shard.get()

    def manage_queues(self):
        queues = [self.iter_queue(b) for b in range(self.batch_size)]
        while True:
            yield [next(q) for q in queues]

    def manage_queues_multiproc(self, queue_indices=None, output_queue=None):
        assert output_queue is not None
        if queue_indices is None:
            queue_indices = list(range(self.batch_size))
        queues = [self.iter_queue(b) for b in queue_indices]

        while True:
            output_queue.put([next(q) for q in queues], block=True)

    def __iter__(self):
        return self.manage_queues()

    def __len__(self):
        if self.drop_last:
            return len(self.shard_sampler) // self.batch_size
        else:
            return (len(self.shard_sampler) + self.batch_size - 1) // self.batch_size

class DistributedBatchShardSampler(BatchShardSampler):
    """
    Coordinates random states so that shard sampling for distributed training can be coordinated
    without any communication between distributed processes. This is possible since random numbers
    are pseudo-deterministic, so if the random states of the global batch are known data loading
    can be coordinated without communication with other processes.
    Purpose: For use with distributed training of L2R modeling.
    Arguments: 
        shard_sampler (RandomShardSampler): Shard sampler used to sample data shards.
        local_batch_size (int): Local batch size to sample.
        drop_last (boolean): Pretty much useless. Used to give a fake length.
        local_random_batch (list): List of random states to use locally for this worker.
        world_size (int): Number of workers in distributed training.
        rank (int): Rank of this distributed worker.
        batch (list): Batch of shard queues (a list that contains shards). Call `.get` and 
            `.isdone()` on `shard_queue[0]` to get next batch and check if shard is done.
    """
    def __init__(self, shard_sampler, local_batch_size, drop_last, local_random_batch=None, world_size=1, rank=0):
        self.global_batch_size = int(local_batch_size*world_size)
        if local_random_batch is None:
            local_random_batch = [np.random.RandomState(seed) for seed in np.random.randint(self.global_batch_size*999, size=self.global_batch_size)]
            local_random_batch = local_random_batch[local_batch_size*rank:local_batch_size*(rank+1)]
        super(DistributedBatchShardSampler, self).__init__(shard_sampler, local_batch_size, drop_last, local_random_batch)

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.global_batch_size
        else:
            return (len(self.sampler) + self.global_batch_size - 1) // self.global_batch_size