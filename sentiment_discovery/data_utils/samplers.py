import math

import torch
from torch.utils import data
import numpy as np

class DistributedBatchSampler(data.sampler.BatchSampler):
	"""
	similar to normal implementation of distributed batch sampler, except if sampler is transposed sampler
	has option to wrap around instead of not dropping last half batch. This is useful for persisting state
	"""
	def __init__(self, sampler, batch_size, drop_last, rank=-1, world_size=2, wrap_last=True):
		super(DistributedBatchSampler, self).__init__(sampler, batch_size, drop_last)
		if rank == -1:
			rank = torch.distributed.get_rank()
		self.rank = rank
		self.world_size = world_size
		self.sampler.wrap_around = 0
		self.wrap_around = 0
		self.wrap_last = wrap_last

	def __iter__(self):
		batch = []
		for idx in self.data_iterator(self.sampler, wrap_around=False):
			batch.append(idx)
			if len(batch) == self.batch_size:
				yield self._batch(batch)
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
	def __init__(self, sampler, batch_size, drop_last, wrap_last=True):
		super(BatchSampler, self).__init__(sampler, batch_size, drop_last)
		self.wrap_around = 0
		self.sampler.wrap_around = 0
		self.wrap_last = wrap_last

	def __iter__(self):
		batch = []
		for idx in self.data_iterator(self.sampler, wrap_around=False):
			batch.append(idx)
			new_batch_len = len(batch)
			if new_batch_len == self.batch_size:
				yield batch
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
		self.strat_width = math.ceil(self.len_ds/batch_size)
		self.data_sampler = data_sampler
		self.wrap_around = 0

	def transpose_helper(self, x):
		"""computes index corrseponding to transpose of index x"""
		x += self.wrap_around
		return ((x%self.batch_size)*self.strat_width+(x//self.batch_size))%self.len_ds

	def __iter__(self):
		if self.data_sampler is None:
			return iter(map(self.transpose_helper, range(len(self))))
		return iter(map(self.transpose_helper, iter(self.data_sampler)))

	def __len__(self):
		return self.len_ds

# class ShardedSampler(data.sampler.Sampler):
# 	def __init__(self,data_source,batch_size,data_sampler=None,num_shards=1000):
# 		"""data_sampler depricated"""
# 		self.data_source=data_source
# 		self.batch_size=batch_size
# 		len_ds=len(data_source)
# 		num_strs=data_source.num_strs
# 		self.data_sampler=data_sampler
# 		self.wrap_around=0
# 		self.num_shards=num_shards
# 		self.shard_size=len_ds//num_shards
# 		# self.shards=list(np.arange(num_shards)*shard_size)
# 		# self.shards=self.get_all_shards()
# 		self.shards=self.prepared_shards(self.get_all_shards())
# 		self.len_ds=self.get_len()
# 		self.active_shards=[]
# 		self.shards_started=0

# 	def strat_helper(self,x):
# 		active_shard_ind=x%self.batch_size
# 		if len(active_shards)<self.batch_size:
# 			self.active_shards.append()

# 	def __iter__(self):
# 		if self.data_sampler is None:
# 			# return iter(map(self.strat_helper,range(len(self))))
# 			return iter(map(self.shard_helper,self.loop_shards(self.shards)))
# 	def shardhelper(self):

# 	def __len__(self):
# 		return self.len_ds

# 	def get_shard_length(self,shard_start_ind):
# 		return self.data_source.get_shard_length(shard_start_ind,self.shard_size)

# 	def get_all_shards(self):
# 		shard_start_inds=list(np.arange(self.num_shards)*self.shard_size)
# 		shard_lens=[self.get_shard_length(shard_start_ind) for x in shard_start_inds]
# 		shards=list(zip(shard_start_inds,shard_lens))
# 		return shards

# 	def prepare_shards(self,shards):
# 		prepared_shards=sorted(shards,key=lambda x:x[1],reverse=True)
# 		return prepared_shards

# 	def get_len(self,shards):
# 		# shard_copy=shards.copy()
# 		# ctr=0
# 		# active_shards=[shard_copy.pop(0) for x in range(self.batch_size)]
# 		# while True:
# 		# 	i=0
# 		# 	should_break=False
# 		# 	while i<len(active_shards):
# 		# 		s=active_shards[i]
# 		# 		s[1]-=1
# 		# 		if s[1]==0:
# 		# 			if len(shard_copy)>0:
# 		# 				should_break=True
# 		# 				break
# 		# 			active_shards[i]=shard_copy.pop(0)
# 		# 		i+=1
# 		# 	if should_break:
# 		# 		break
# 		# 	ctr+=1
# 		ctr=0
# 		for idx in self.loop_shards(shards):
# 			ctr+=1
# 		return ctr/self.batch_size

# 	def loop_shards(self,shards):
# 		shard_copy=shards.copy()
# 		active_shards=[shard_copy.pop(0) for x in range(self.batch_size)]
# 		shard_ctr=[0]*len(active_shards)
# 		while True:
# 			i=0
# 			should_break=False
# 			while i<len(active_shards):
# 				s=active_shards[i]
# 				s[1]-=1
# 				cnt=shard_ctr[i]
# 				shard_ctr[i]+=1
# 				if s[1]==0:
# 					if len(shard_copy)>0:
# 						should_break=True
# 						break
# 					active_shards[i]=shard_copy.pop(0)
# 					shard_ctr[i]=0
# 				yield (s[0],cnt)
# 				i+=1
# 			if should_break:
# 				break
