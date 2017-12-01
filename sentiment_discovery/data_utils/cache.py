class array_cache(object):
	"""
	Arguments:
		cache_strs (list-like): List like object with __len__ and __getitem__
		cache_block_size (int): number of strings to cache in one cache block. Default: 64
		cache_size (int): number of caches blocks to store before removing (LRU). Default: 32
	Attributes:
		num_strs (int): len(cache_strs)
		cache (dict): holds cache blocks
		cache_blocks (list): list of keys for blocks stored in caches
	"""
	def __init__(self, cache_strs, cache_block_size=64, cache_size=32):
		super(array_cache, self).__init__()
		self.cache_size = cache_size
		self.cache_block_size = cache_block_size
		self.cache_strs = cache_strs
		self.num_strs = len(self.cache_strs)
		self.cache = {}
		self.cache_blocks = []

	def __getitem__(self, index):
		#get index of cache block of size cache_block_size
		block_ind = index//self.cache_block_size
		if block_ind not in self.cache:
			self.clean_out_cache()
			cache_block = self.cache_strs[index:min(index+self.cache_block_size, self.num_strs)]
			#store cache block in cache
			self.cache[block_ind] = (cache_block)
			#append key to cache block list
			self.cache_blocks.append(block_ind)
		else:
			cache_block = self.cache[block_ind]
		#get a strings index inside of a cache block
		block_ind_ind = index%self.cache_size

		return cache_block[block_ind_ind]

	def __len__(self):
		return len(self.cache_strs)

	def clean_out_cache(self):
		"""gets index of oldest cache block. and removes the block from cache and removes the index"""
		if len(self.cache_blocks) >= self.cache_size:
			block_ind = self.cache_blocks.pop(0)
			del self.cache[block_ind]
