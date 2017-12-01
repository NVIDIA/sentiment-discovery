import threading

import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.cuda.comm as comm
from torch.nn import Module
import torch.distributed as dist
from torch._utils import _unflatten_tensors
from torch.cuda import nccl

class DataParallel(Module):
	"""Implements data parallelism at the module level.
	This container parallelizes the application of the given module by
	splitting the input across the specified devices by chunking in the batch
	dimension. In the forward pass, the module is replicated on each device,
	and each replica handles a portion of the input. During the backwards
	pass, gradients from each replica are summed into the original module.
	The batch size should be larger than the number of GPUs used. It should
	also be an integer multiple of the number of GPUs so that each chunk is the
	same size (so that each GPU processes the same number of samples).
	See also: :ref:`cuda-nn-dataparallel-instead`
	Arbitrary positional and keyword inputs are allowed to be passed into
	DataParallel EXCEPT Tensors. All variables will be scattered on dim
	specified (default 0). Primitive types will be broadcasted, but all
	other types will be a shallow copy and can be corrupted if written to in
	the model's forward pass.
	Args:
		module: module to be parallelized
		device_ids: CUDA devices (default: all devices)
		output_device: device location of output (default: device_ids[0])
	Example::
		>>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
		>>> output = net(input_var)
	"""

	# TODO: update notes/cuda.rst when this class handles 8+ GPUs well

	def __init__(self, module, device_ids=None, output_device=None, dim=0):
		super(DataParallel, self).__init__()
		if device_ids is None:
			device_ids = list(range(torch.cuda.device_count()))
		if output_device is None:
			output_device = device_ids[0]
		self.dim = dim
		self.module = module
		self.device_ids = device_ids
		self.output_device = output_device
		if len(self.device_ids) == 1:
			self.module.cuda(device_ids[0])

	def forward(self, *inputs, **kwargs):
		inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
		if len(self.device_ids) == 1:
			return self.module(*inputs[0], **kwargs[0])
		replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
		outputs = self.parallel_apply(replicas, inputs, kwargs)
		return self.gather(outputs, self.output_device)

	def replicate(self, module, device_ids):
		return replicate(module, device_ids)

	def scatter(self, inputs, kwargs, device_ids):
		return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

	def parallel_apply(self, replicas, inputs, kwargs):
		return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

	def gather(self, outputs, output_device):
		return gather(outputs, output_device, dim=self.dim)

def replicate(network, devices):
	devices = tuple(devices)
	num_replicas = len(devices)

	params = list(network.parameters())
	param_indices = {param: idx for idx, param in enumerate(params)}
	param_copies = Broadcast.apply(devices, *params)
	if len(params) > 0:
		param_copies = [param_copies[i:i + len(params)]
						for i in range(0, len(param_copies), len(params))]

	buffers = list(network._all_buffers())
	buffer_indices = {buf: idx for idx, buf in enumerate(buffers)}
	buffer_copies = comm.broadcast_coalesced(buffers, devices)

	modules = list(network.modules())
	module_copies = [[] for device in devices]
	module_indices = {}

	for i, module in enumerate(modules):
		module_indices[module] = i
		for j in range(num_replicas):
			replica = module.__new__(type(module))
			replica.__dict__ = module.__dict__.copy()
			replica._parameters = replica._parameters.copy()
			replica._buffers = replica._buffers.copy()
			replica._modules = replica._modules.copy()
			module_copies[j].append(replica)

	for i, module in enumerate(modules):
		for key, child in module._modules.items():
			if child is None:
				for j in range(num_replicas):
					replica = module_copies[j][i]
					replica._modules[key] = None
			else:
				module_idx = module_indices[child]
				for j in range(num_replicas):
					replica = module_copies[j][i]
					replica._modules[key] = module_copies[j][module_idx]
		for key, param in module._parameters.items():
			if param is None:
				for j in range(num_replicas):
					replica = module_copies[j][i]
					replica._parameters[key] = None
			else:
				param_idx = param_indices[param]
				for j in range(num_replicas):
					replica = module_copies[j][i]
					replica._parameters[key] = param_copies[j][param_idx]
		for key, buf in module._buffers.items():
			if buf is None:
				for j in range(num_replicas):
					replica = module_copies[j][i]
					replica._buffers[key] = None
			else:
				buffer_idx = buffer_indices[buf]
				for j in range(num_replicas):
					replica = module_copies[j][i]
					replica._buffers[key] = buffer_copies[j][buffer_idx]

	return [module_copies[j][0] for j in range(num_replicas)]

def get_a_var(obj):
	if isinstance(obj, Variable):
		return obj

	if isinstance(obj, list) or isinstance(obj, tuple):
		results = map(get_a_var, obj)
		for result in results:
			if isinstance(result, Variable):
				return result
	if isinstance(obj, dict):
		results = map(get_a_var, obj.items())
		for result in results:
			if isinstance(result, Variable):
				return result
	return None


def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):
	assert len(modules) == len(inputs)
	if kwargs_tup is not None:
		assert len(modules) == len(kwargs_tup)
	else:
		kwargs_tup = ({},) * len(modules)
	if devices is not None:
		assert len(modules) == len(devices)
	else:
		devices = [None] * len(modules)

	lock = threading.Lock()
	results = {}

	def _worker(i, module, input, kwargs, results, lock, device=None):
		if device is None:
			device = get_a_var(input).get_device()
		try:
			with torch.cuda.device(device):
				output = module(*input, **kwargs)
			with lock:
				results[i] = output
		except Exception as e:
			with lock:
				results[i] = e

	if len(modules) > 1:
		threads = [threading.Thread(target=_worker,
									args=(i, module, input, kwargs, results, lock, device),
									)
				   for i, (module, input, kwargs, device) in
				   enumerate(zip(modules, inputs, kwargs_tup, devices))]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()
	else:
		_worker(0, modules[0], inputs[0], kwargs_tup[0], results, lock, devices[0])

	outputs = []
	for i in range(len(inputs)):
		output = results[i]
		if isinstance(output, Exception):
			raise output
		outputs.append(output)
	return outputs

def scatter(inputs, target_gpus, dim=0):
	"""
	Slices variables into approximately equal chunks and
	distributes them across given GPUs. Duplicates
	references to objects that are not variables. Does not
	support Tensors.
	"""
	def scatter_map(obj):
		if isinstance(obj, Variable):
			return Scatter.apply(target_gpus, None, dim, obj)
		assert not torch.is_tensor(obj), "Tensors not supported in scatter."
		if isinstance(obj, tuple):
			return list(zip(*map(scatter_map, obj)))
		if isinstance(obj, list):
			return list(map(list, zip(*map(scatter_map, obj))))
		if isinstance(obj, dict):
			return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
		return [obj for targets in target_gpus]

	return scatter_map(inputs)


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
	"""Scatter with support for kwargs dictionary"""
	inputs = scatter(inputs, target_gpus, dim) if inputs else []
	kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
	if len(inputs) < len(kwargs):
		inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
	elif len(kwargs) < len(inputs):
		kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
	inputs = tuple(inputs)
	kwargs = tuple(kwargs)
	return inputs, kwargs


def gather(outputs, target_device, dim=0):
	"""
	Gathers variables from different GPUs on a specified device
	  (-1 means the CPU).
	"""
	def gather_map(outputs):
		out = outputs[0]
		if isinstance(out, Variable):
			return Gather.apply(target_device, dim, *outputs)
		if out is None:
			return None
		return type(out)(map(gather_map, zip(*outputs)))
	return gather_map(outputs)

class Broadcast(Function):

	@staticmethod
	def forward(ctx, target_gpus, *inputs):
		if not all(input.is_cuda for input in inputs):
			raise TypeError('Broadcast function not implemented for CPU tensors')
		ctx.target_gpus = target_gpus
		if len(inputs) == 0:
			return tuple()
		ctx.num_inputs = len(inputs)
		ctx.input_device = inputs[0].get_device()
		outputs = comm.broadcast_coalesced(inputs, ctx.target_gpus)
		return tuple([t for tensors in outputs for t in tensors])

	@staticmethod
	def backward(ctx, *grad_outputs):
		return (None,) + ReduceAddCoalesced.apply(ctx.input_device, ctx.num_inputs, *grad_outputs)


class ReduceAddCoalesced(Function):

	@staticmethod
	def forward(ctx, destination, num_inputs, *grads):
		ctx.target_gpus = [grads[i].get_device() for i in range(0, len(grads), num_inputs)]

		grads = [grads[i:i + num_inputs]
				 for i in range(0, len(grads), num_inputs)]
		return comm.reduce_add_coalesced(grads, destination)

	@staticmethod
	def backward(ctx, *grad_outputs):
		return (None, None,) + Broadcast.apply(ctx.target_gpus, *grad_outputs)


class Gather(Function):

	@staticmethod
	def forward(ctx, target_device, dim, *inputs):
		assert all(map(lambda i: i.is_cuda, inputs))
		ctx.target_device = target_device
		ctx.dim = dim
		ctx.input_gpus = tuple(map(lambda i: i.get_device(), inputs))
		ctx.input_sizes = tuple(map(lambda i: i.size(ctx.dim), inputs))
		return comm.gather(inputs, ctx.dim, ctx.target_device)

	@staticmethod
	def backward(ctx, grad_output):
		return (None, None) + Scatter.apply(ctx.input_gpus, ctx.input_sizes, ctx.dim, grad_output)


class Scatter(Function):

	@staticmethod
	def forward(ctx, target_gpus, chunk_sizes, dim, input):
		ctx.target_gpus = target_gpus
		ctx.chunk_sizes = chunk_sizes
		ctx.dim = dim
		ctx.input_device = input.get_device() if input.is_cuda else -1
		streams = None
		if ctx.input_device == -1:
			# Perform CPU to GPU copies in a background stream
			streams = [_get_stream(device) for device in ctx.target_gpus]
		outputs = comm.scatter(input, ctx.target_gpus, ctx.chunk_sizes, ctx.dim, streams)
		# Synchronize with the copy stream
		if streams is not None:
			for i, output in enumerate(outputs):
				with torch.cuda.device(ctx.target_gpus[i]):
					main_stream = torch.cuda.current_stream()
					main_stream.wait_stream(streams[i])
					output.record_stream(main_stream)
		return outputs

	@staticmethod
	def backward(ctx, *grad_output):
		return None, None, None, Gather.apply(ctx.input_device, ctx.dim, *grad_output)


# background streams used for copying
_streams = None


def _get_stream(device):
	"""Gets a background stream for copying between CPU and GPU"""
	global _streams
	if device == -1:
		return None
	if _streams is None:
		_streams = [None] * torch.cuda.device_count()
	if _streams[device] is None:
		_streams[device] = torch.cuda.Stream(device)
	return _streams[device]

class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
	def __init__(self, module, device_ids=None, output_device=None, dim=0):
		super(torch.nn.parallel.DistributedDataParallel, self).__init__()

		if device_ids is None:
			device_ids = list(range(torch.cuda.device_count()))
		if output_device is None:
			output_device = device_ids[0]
		self.dim = dim
		self.module = module
		self.device_ids = device_ids
		self.output_device = output_device

		# Sync params and buffers
		for p in self.module.state_dict().values():
			dist.broadcast(p, 0)

		if len(device_ids) > 1:
			# TODO: we don't need to replicate params in here. they're always going to
			# be broadcasted using larger blocks in broadcast_coalesce, so it might be
			# better to not pollute the caches with these small blocks
			self._module_copies = replicate(self.module, self.device_ids)
			self._module_copies[0] = self.module
			for module_copy in self._module_copies[1:]:
				for param, copy_param in zip(self.module.parameters(), module_copy.parameters()):
					copy_param.detach_()
					copy_param.requires_grad = param.requires_grad
		else:
			self._module_copies = [self.module]

		# Split parameters into buckets that will coalesce reductions
		# TODO: different types need different buckets
		t = None
		for p in self.module.parameters():
			tp = type(p.data)
			if t is not None and t is not tp:
				raise ValueError("DistributedDataParallel requires all parameters' data to be of the same type")
			t = tp

		self.bucket_sizes = []
		self.bucket_map = {}
		MB = 1024 * 1024
		self.broadcast_bucket_size = 10 * MB  # used for param sync before forward
		bucket_bytes_cap = 1 * MB
		bucket_bytes = bucket_bytes_cap  # to init the first bucket immediately
		for param_tuple in zip(*map(lambda m: m.parameters(), self._module_copies)):
			if bucket_bytes >= bucket_bytes_cap:
				self.bucket_sizes.append(0)
				bucket_bytes = 0
			self.bucket_sizes[-1] += 1
			for p in param_tuple:
				self.bucket_map[p] = len(self.bucket_sizes) - 1
			bucket_bytes += p.numel() * p.element_size()

		self.buckets = [[[] for _ in range(len(self.device_ids))] for _ in range(len(self.bucket_sizes))]
		self.bucket_events = [[None] * len(self.device_ids) for _ in range(len(self.bucket_sizes))]
		self.reduced = [False] * len(self.bucket_sizes)

		self._register_grad_hooks()

		self.dispatch_lock = threading.Lock()
		self._start_reduction_threads()

	def forward(self, *inputs, **kwargs):
		inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
		self._sync_params()
		if len(self.device_ids) == 1:
			return self.module(*inputs[0], **kwargs[0])
		outputs = self.parallel_apply(self._module_copies, inputs, kwargs)
		return self.gather(outputs, self.output_device)

	def scatter(self, inputs, kwargs, device_ids):
		return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

	def _sync_params(self):
		params = [p.data for p in self.module.parameters()]
		result = comm.broadcast_coalesced(params, self.device_ids, self.broadcast_bucket_size)
		for tensors, module in zip(result[1:], self._module_copies[1:]):
			for tensor, param in zip(tensors, module.parameters()):
				param.data.set_(tensor)

		# cross-node buffer sync
		buffers = list(self.module._all_buffers())
		if len(buffers) > 0:
			flat_buffers = _flatten_tensors(buffers)
			# flat_buffers = torch._utils._flatten_tensors(buffers)
			dist.broadcast(flat_buffers, 0)
			for buf, synced in zip(buffers, torch._utils._unflatten_tensors(flat_buffers, buffers)):
				buf.copy_(synced)

			# intra-node buffer sync
			result = comm.broadcast_coalesced(buffers, self.device_ids, self.broadcast_bucket_size)
			for tensors, module in zip(result[1:], self._module_copies[1:]):
				for tensor, buf in zip(tensors, module._all_buffers()):
					buf.set_(tensor)
	@staticmethod
	def _reduction_thread_fn(queue, group_id, device_ids, reduction_streams, nccl_streams):

		def _process_batch():
			dev_grad_batch, dev_events, job_event = queue.get()
			dev_coalesced = []
			# Coalesce the tensors on all devices and start a local reduction
			for dev_id, grad_batch, event, stream in zip(device_ids, dev_grad_batch,
														dev_events, reduction_streams):
				with torch.cuda.device(dev_id), torch.cuda.stream(stream):
					stream.wait_event(event)
					coalesced = _flatten_tensors(grad_batch)
					dev_coalesced.append(coalesced)
			# Wait for all copies to complete before starting the NCCL kernel
			for stream in reduction_streams:
				stream.synchronize()
			try:
				nccl.reduce(dev_coalesced, root=0, streams=nccl_streams)
			except Exception as e:
				raise e

			# From now on we're only going to work on the first device (from device_ids)
			grad_batch = dev_grad_batch[0]
			coalesced = dev_coalesced[0]
			reduce_stream = reduction_streams[0]
			with torch.cuda.stream(reduce_stream):
				reduce_stream.wait_stream(nccl_streams[0])
				coalesced /= dist.get_world_size()
				dist.all_reduce(coalesced, group=group_id)
				for grad, reduced in zip(grad_batch, _unflatten_tensors(coalesced, grad_batch)):
					grad.copy_(reduced)
			job_event.set()

		with torch.cuda.device(device_ids[0]):
			while True:
				_process_batch()  # just to have a clear scope
def _flatten_tensors(tensors):
	"""Flatten tensors into a single contiguous 1D buffer"""
	if len(tensors) == 1:
		return tensors[0].contiguous().view(-1)
	numels = [tensor.numel() for tensor in tensors]
	size = sum(numels)
	offset = 0
	flat = tensors[0].new(size)
	for tensor, numel in zip(tensors, numels):
		flat.narrow(0, offset, numel).copy_(tensor, broadcast=False)
		offset += numel
	return flat
	