from collections import defaultdict

import numpy as np

from sentiment_discovery.reparameterization import apply_weight_norm, remove_weight_norm
from torch.nn.parallel import DataParallel, DistributedDataParallel
from .utils import copy_state, clip_gradients, calc_grad_norm, clip_gradient
from .fp16 import FP16_Optimizer, fp16_to_fp32, fp32_to_fp16

def no_op(loss_tensors):
	"""no op on input"""
	return loss_tensors

def aggregate_parallel_loss(loss_tensors):
	"""averages input losses"""
	return loss_tensors.mean()

class ModelWrapper(object):
	"""
	Arguments:
		module (nn.Module): module we want to provide wrapper functionality for
	Attributes:
		using_cuda: whether module is on gpu or cpu
		loss_fn: loss function to use to optimize module. Module should perform loss computation
			within call to __forward__
		loss: current loss value from the module's optimization
		agg_func: function to aggregate loss values. Used when module is DataParallel, to aggregate
			the loss values from each of the replicas
		gn: gradient norm of all module parameters resulting from optimizing loss. calculated by
			sum([gn(p) for p in module.parameters()])
		is_training: whether module is in training mode or not.
		init_state: initial state to use for the module if it is stateful
		distributed: whether module is using DistributedDataParallel
		world_size: number of distributed processes in use with DistributedDataParallel
		rank: rank of this instance of distributed processing
		_should_skip: boolean whether module should skip current gradient update of optimization
		weight_norm : boolean whether weight norm has been applied to module parameters or not
		optimizer: instance of optimizer class to optimize module parameters.
		clip: clamp module parameter gradients to +/- clip
	Examples:
		embedder_module = nn.Embedding(data_size, input_size)
		recurrent_module = model.SequenceModel(embedder=embedder_module, cell_type='mlstm',
													num_layers=1, input_size=input_size,
													rnn_size=hidden_size, output_size=data_size,
													dropout=0)
		model=ModelWrapper(recurrent_module, lstm_only=True)
		for iter in range(num_iters):
			text_data, mask_data, timesteps = get_data()
			states, out = model(text=text_data, state_mask=mask_data, timesteps=timesteps, return_sequence=False)
			skip = model.should_skip()
			if not skip:
				model.optim_step()
				model.persist_state(hiddens)
			loss = model.loss
			gn = model.gn
			# log/print values
	"""
	def __init__(self, module, lstm_only=False):
		super(ModelWrapper, self).__init__()
		self.module = module
		self.using_cuda = False
		self._module = self.module
		self.data_parallel = False
		self.weight_norm = False
		self.lstm_only = lstm_only

		self.loss_fn = None
		self.loss = 0.
		self.agg_func = no_op
		self.gn = 0

		self.clip = None
		self.is_training = False
		self.optimizer = None

		self.init_state = None

		self.distributed = False
		self.world_size = -1
		self.rank = -1

		self._should_skip = False

		self.fp16 = False

	def apply_weight_norm(self):
		"""applies weight norm to all module parameters"""
		# if lstm_only apply weight norm only to the lienar gates of lstm
		if self.lstm_only:
			[apply_weight_norm(m, hook_child=False) for m in self.module.rnn.layers]
		else:
			apply_weight_norm(self.module, hook_child=False)
		self.weight_norm = True

	def remove_weight_norm(self):
		"""removes weight norm from all module parameters"""
		remove_weight_norm(self.module)
		self.weight_norm = False

	def add_optimizer(self, optimizer=None, load_optim=False, lr=None, clip=None):
		"""
		Attribute:
			optimizer (torch.optim.Optimizer): either optim class to optimize module parameters or, optimizer
				instance for module parameters. If None is provided then no optimization is done. Default: None
			load_optim (bool): if optimizer is optimizer class instance then load_optim must be true.
				Meant for reloading optimizers from past training runs. Default: False.
			lr (float): learning rate for newly created optimizer. Must be provided if load_optim is
				false and optimizer is not None.
			clip (float): clamp module parameter gradients to +/- clip
		"""
		if optimizer is None:
			return
		if load_optim:
			self.optimizer = optimizer
		else:
			assert lr is not None
			# if self.fp16:
			# 	self.optimizer = FP16_Optimizer(optimizer, self.module, lr=lr)
			# else:
			self.optimizer = optimizer(self.parameters(), lr=lr)
		if self.fp16:
			self.optimizer = FP16_Optimizer(self.optimizer, self.module)
		if clip > 0:
			self.clip = clip

	def initialize(self, batch_size, volatile=False):
		"""initialize hidden state of module"""
		self.init_state = self._module.state0(batch_size, volatile=volatile)
		self.init_state = (self.init_state[0].transpose(0, 1), self.init_state[1].transpose(0, 1))

	def set_loss_fn(self, fn):
		"""set loss function attribute"""
		self.loss_fn = fn

	def train(self, mode=True):
		"""mimic nn.Module.train"""
		self.is_training = mode
		self.module.train(mode)

	def eval(self, mode=True):
		"""mimic nn.Module.eval"""
		self.is_training = not mode
		self.module.eval(mode)

	def cuda(self):
		"""mimic nn.Module.cuda"""
		self.module = self.module.cuda()
		self.cuda_state()
		self.cuda_loss()
		self.using_cuda = True
		return self

	def cuda_state(self):
		"""move init_state to gpu"""
		if self.init_state is not None:
			self.init_state = tuple([x.cuda() for x in self.init_state])

	def cuda_loss(self):
		"""move loss to gpu (necessary if loss is weighted)"""
		if self.loss_fn is not None:
			self.loss_fn.cuda()

	def cpu(self):
		"""mimic nn.Module.cpu"""
		self.module = self.module.cpu()
		self.cpu_state()
		self.cpu_loss()
		self.using_cuda = False

	def cpu_state(self):
		"""move init_state to cpu"""
		if self.init_state is not None:
			self.init_state = tuple([x.cpu() for x in self.init_state])

	def cpu_loss(self):
		"""move loss to cpu (necessary if loss is weighted)"""
		if self.loss_fn is not None:
			self.loss_fn.cuda()

	def half(self):
		# data parallel must be enabled after fp16 is turned on
		assert not self.data_parallel
		self.module = self.module.half()
		old_forward = self.module.forward
		def fp16_forward(*inputs, **kwargs):
			return fp16_to_fp32(old_forward(*(fp32_to_fp16(inputs)), **kwargs))
		self.module.forward = fp16_forward
		self.fp16 = True

	def make_data_parallel(self, device_ids=None, output_device=None, dim=0, distributed=False,
							rank=-1, world_size=2):
		"""make module attribute data parallel"""
		if isinstance(device_ids, int):
			num_devs = device_ids
			device_ids = list(range(device_ids))
		elif device_ids is None:
			num_devs = world_size
		else:
			num_devs = len(device_ids)

		if not distributed:
			# move hidden state from gpu to cpu so it can be scatterred properly by DataParallel
			self.cpu_state()
			self._module = DataParallel(self._module, device_ids=device_ids, output_device=output_device, dim=dim)
			self.agg_func = aggregate_parallel_loss
		else:
			assert rank < num_devs and rank != -1, 'please provide correct rank'
			assert self.init_state is not None
			assert self.using_cuda
			self._module = DistributedDataParallel(self._module, device_ids=[0], output_device=output_device, dim=dim)
			self.distributed = True
			self.rank = rank

		self.data_parallel = True
		return self

	def __call__(self, *inputs, **args):
		del self.loss
		del self.gn
		self.loss = 0
		self.gn = 0

		# handle arguments
		module_input, module_args = self.pack_args(*inputs, **args)

		# run forward pass on input
		_outputs = self._module(*module_input, **module_args)

		# get outpus and loss
		outputs, loss = self.process_outputs(_outputs)

		# if model is training optimize loss
		if self.is_training:
			assert loss is not None
			loss = self.agg_func(loss)
			self.zero_grad()

			loss.backward()
			# compute gradient statistics
			self.gn = calc_grad_norm(self.module)
			# clip gradients
			if self.clip is not None:
				clip_gradient(self.module, self.clip)
		self.loss = loss
		return outputs

	def pack_args(self, *inputs, **args):
		"""
		parse module inputs from inputs and args and construct arg_map.
		arg_map specifies what module inputs are in what index in module_input list
		Returns module_input, module_args:
			module_input: list of all tensors for module input
			module_args: arg_map for module inputs as well as remiaing keyword arguments.
		"""
		module_input = []

		arg_map = {}
		module_args = {}
		module_args['arg_map'] = arg_map
		module_args['loss_fn'] = self.loss_fn
		module_args['transpose'] = True
		inputs_processed = 0
		num_args = 0

		# if text is not in args it should be the first object in inputs
		if 'text' in args and args['text'] is not None:
			module_input.append(args['text'])
		else:
			module_input.append(inputs[inputs_processed])
			inputs_processed += 1
		arg_map['x'] = num_args
		num_args += 1

		# pass module initial state to module_inputs
		module_input.append(self.init_state)
		arg_map['hidden_init'] = num_args
		num_args += 1

		# get input sequence length
		if 'timesteps' in args and args['timesteps'] is not None:
			ts = args['timesteps']
		else:
			ts = inputs[inputs_processed]
			inputs_processed += 1

		# if sequence length is not a tensor move it to module_args so that it doesn't have scatter problems from DataParallel
		if isinstance(ts, int) or type(ts[0]).__module__ == 'numpy':
			module_args['timesteps'] = ts
		else:
			module_input.append(ts)
			arg_map['timesteps'] = num_args
			num_args += 1

		# pack state persistence mask
		if 'state_mask' in args and args['state_mask'] is not None:
			module_input.append(args['state_mask'])
			arg_map['state_mask'] = num_args
			num_args += 1

		# pack return sequence boolean
		if 'return_sequence' in args:
			return_sequence = args['return_sequence']
		else:
			return_sequence = False
		module_args['return_sequence'] = return_sequence
		return module_input, module_args


	def should_skip(self, skip_rule=None):
		"""
		return boolean whether to skip module update
		use either an external formulation or our predefined one
		"""
		if isinstance(skip_rule, str) and skip_rule.lower() == 'no':
			self.skip = False
			return self.skip
		elif skip_rule is not None:
			return skip_rule(self)
		else:
			# skip gradient update whenever grad norm is too large
			self.skip = False
			# make sure that we're not at the begining of training when gradients are too large
			# wait for our gradient norm to first dip below some threshold
			skip_threshold = 1.
			low_gradient_threshold = .5
			if self._should_skip:
				if (self.gn) > skip_threshold or (self.gn) == np.float('nan') or self.gn == np.float('inf'):
					self.skip = True
			else:
				if (self.gn) < low_gradient_threshold:
					self._should_skip = True
			return self.skip


	def process_outputs(self, _outputs):
		"""get module outputs and loss"""
		# if we have no loss function then module won't return loss
		if self.loss_fn is not None:
			outputs, loss = _outputs
		else:
			outputs = _outputs
			loss = 0
		return outputs, loss

	def persist_state(self, hidden, persist=True, inplace=True):
		"""
		possibly persist state from some earlier call to module and use as initial state for a future call to module
		Arguments:
			hidden (Tensor): hidden state to persist. (possibly tuple of tensors)
			persist (bool):  if set to False then state is not persisted, and old state is used. Default: True
			inplace (bool): update ModelWrapper.init_state in place. Default: True
		Returns s:
			s (Tensor): hidden_state to use for future calls to module
		"""
		if persist:
			if inplace:
				del self.init_state
			s = copy_state(hidden, make_cpu=not self.using_cuda or (self.data_parallel and not self.distributed))
			if self.distributed:
				s = tuple([x.cuda() for x in s])
			if inplace:
				self.init_state = s
			return s

	def get_neurons(self, outputs):
		"""
		Given outputs from module get cell state and convert to numpy
		"""
		# outputs=(cell,hidden),outs
		hidden = outputs[0][1]

		hidden_data = hidden.data
		if hidden_data.is_cuda:
			hidden_data = hidden_data.cpu()
		hidden_data = hidden_data[:,-1,:]
		squeezed_data = hidden_data.numpy()
		return squeezed_data

	def zero_grad(self):
		"""zero gradient optimizer"""
		if self.is_training:
			self.optimizer.zero_grad()

	def reset_optim(self):
		"""
		reset state of optimizer. used for resetting optimizer momentum/stats when batch updates are skipped
		"""
		self.optimizer.state = defaultdict(dict)

	def optim_step(self):
		"""step optimizer"""
		if self.is_training:
			self.optimizer.step()

	def parameters(self):
		"""mimic nn.Module.parameters()"""
		return list(self.module.parameters())

	def state_dict(self, destination=None, prefix='', keep_vars=False):
		"""mimic nn.Module.state_dict()"""
		if self.weight_norm:
			self.remove_weight_norm()
			sd = self.module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
			self.apply_weight_norm()
		else:
			sd = self.module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
		return sd

	def load_state_dict(self, state_dict, strict=True):
		"""mimic nn.Module.load_state_dict()"""
		self.module.load_state_dict(state_dict, strict=strict)
