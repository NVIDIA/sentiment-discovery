import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import numpy as np

from sentiment_discovery.modules import StackedLSTM, mLSTMCell
from .utils import make_cuda as make_state_cuda

class SequenceModel(nn.Module):
	"""
	Based on implementation of StackedLSTM from openNMT-py
	https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/StackedRNN.py
	Args:
		embed: instance of torch.nn.Embedding or something with an equivalent __call__ function
		cell: string specifying recurrent cell type ['gru', 'mlstm', 'lstm', 'rnn']. Default: `rnn`
		n_layers: how many of these cells to stack
		in_size: The dimension of the input to the recurrent module (output dimension of embedder)
		rnn_size: The number of features in the hidden states of the lstm cells
		out_size: dimension of linear transformation layer on the output of the stacked rnn cells.
			If <=0 then no output layer is applied. Default: -1
		dropout: probability of dropout layer (applied after rnn, but before output layer). Default: 0
		fused: use fused LSTM kernels if applicable
	Inputs: *inputs, **kwargs
		- **input** (batch, input_size): tensor containing input features
		- **h_0** (num_layers, batch, hidden_size): tensor containing the initial hidden
		  state for each layer of each element in the batch.
		- **c_0** (num_layers, batch, hidden_size): tensor containing the initial cell state
		  for each layer of each element in the batch.
	Outputs: (h_1, c_1), output
		- **h_1** (num_layers, batch, hidden_size): tensor containing the next hidden state
		  for each layer of each element in the batch
		- **c_1** (num_layers, batch, hidden_size): tensor containing the next cell state
		  for each layer of each element in the batch
		- **output** (batch, output_size): tensor containing output of stacked rnn. 
			If `output_size==-1` then this is equivalent to `h_1`
	Examples:
		>>> rnn = nn.StackedLSTM(mLSTMCell, 1, 10, 20, 15, 0)
		>>> input = Variable(torch.randn(6, 3, 10))
		>>> hx, cx = hiddens = rnn.state0(3)
		>>> hx.size() # (1,3,20)
		>>> cx.size() # (1,3,20)
		>>> output = []
		>>> for i in range(6):
		...     hiddens, out = rnn(input[i], hiddens)
		...     output.append(out)
	"""

	def __init__(self, embed, cell, n_layers, in_size, rnn_size, out_size, dropout, fused=False):
		super(SequenceModel, self).__init__()
		self.add_module('embedder', embed)
		cell = cell.lower()
		if cell == 'gru':
			rnn_cell = nn.GRUCell
		elif cell == 'mlstm':
			rnn_cell = lambda size_in, size_rnn: mLSTMCell(size_in, size_rnn, fused_lstm=fused)
		elif cell == 'lstm':
			rnn_cell = nn.LSTMCell
		else:
			rnn_cell = nn.RNNCell
		rnn = StackedLSTM(rnn_cell, n_layers, in_size, rnn_size, out_size, dropout)
		self.add_module('rnn', rnn)

	def rnn_parameters(self):
		"""gets params of rnn/lstm"""
		return self.rnn.parameters() 

	def unpack_args(self, *inputs, **kwargs):
		"""unpacks argument from input list and kwargs dict, and handles missing data/different argument formats"""
		x = None
		hidden_init = None
		state_mask = None
		timesteps = None
		loss_fn = None
		seq_mask = False
		mask_state = False
		calc_loss = False
		return_sequence = False

		assert 'arg_map' in kwargs
		arg_map = kwargs['arg_map']
		x = self.get_arg('x', inputs, arg_map)
		using_cuda = x.is_cuda
		if len(x.size()) == 1:
			x = x.unsqueeze(0)

		transpose = False
		hidden_init = self.get_arg('hidden_init', inputs, arg_map)
		if hidden_init is None:
			# TODO: check if variable.volatile is a valid attribute of a variable 
			hidden_init = self.state0(x.size(0), make_cuda=using_cuda, volatile=x.volatile)
			hidden = hidden_init
		else:
			hidden_is_tuple = isinstance(hidden_init, tuple)
			# hidden_is_tuple = isinstance(hidden_init, (tuple, list))
			if 'transpose' not in kwargs or kwargs['transpose']:
				hidden_init = tuple([y.transpose(0, 1) for y in hidden_init]) if hidden_is_tuple \
								else hidden_init.transpose(0, 1)
				transpose = True
			hidden = hidden_init
			if (hidden[0] if hidden_is_tuple else hidden).size(1) != x.size(0):
				hidden = (hidden[0].narrow(1, 0, x.size(0)), hidden[1].narrow(1, 0, x.size(0))) if hidden_is_tuple \
							else hidden.narrow(1, 0, x.size(0))

		state_mask = self.get_arg('state_mask', inputs, arg_map)
		mask_state = state_mask is not None

		timesteps = self.get_arg('timesteps', inputs, arg_map)
		if timesteps is None:
			if 'timesteps' in kwargs:
				num_iters = kwargs['timesteps']
				if not isinstance(num_iters, int):
					timesteps, num_iters, seq_mask = self.handle_timesteps(num_iters, using_cuda)
				elif num_iters == -1:
					num_iters = x.size(1)-1
			else:
				num_iters = x.size(1)-1
		else:
			num_iters = timesteps
			if not isinstance(num_iters, int):
				timesteps, num_iters, seq_mask = self.handle_timesteps(num_iters, using_cuda)

		loss_fn = kwargs['loss_fn'] if 'loss_fn' in kwargs else None
		calc_loss = loss_fn is not None

		return_sequence = kwargs['return_sequence'] if 'return_sequence' in kwargs else False

		return x, hidden, transpose, state_mask, mask_state, timesteps,\
				num_iters, seq_mask, loss_fn, calc_loss, return_sequence

	def handle_timesteps(self, num_iters, using_cuda=False):
		"""
		Gets lengths of all sequences, maximum sequence length, and boolean whether or not to mask output,
		if there's end-padded sequences.
		"""
		if type(num_iters[0]).__module__ == 'numpy':
			timesteps = Variable(torch.from_numpy(num_iters.reshape([-1])))
			lowest_iters = np.min(num_iters)
			num_iters = np.max(num_iters)
		else:
			timesteps = num_iters
			lowest_iters = num_iters.min().data.cpu()[0]
			num_iters = num_iters.max().data.cpu()[0]
		seq_mask = lowest_iters != num_iters
		if using_cuda:
			timesteps = timesteps.cuda()
		return timesteps, num_iters, seq_mask


	def get_arg(self, arg_name, inputs, arg_map):
		"""gets arg from input based on index specified by arg_map"""
		if arg_name not in arg_map:
			return None
		return inputs[arg_map[arg_name]]

	def forward(self, *inputs, **kwargs):
		"""
		Inputs:
			x (LongTensor): [batch,time] shaped tensor of tokens in sequence to be embedded/processed
			hidden_init: possibly a tuple of tensors of [batch,layers,hidden_dim] shape. Initial hidden state
			state_mask (optional): [batch,time] shaped tensor of 0s,1s specifying whether to persist state
				on a given timestep or reset it. Used to reset state mid sequence.
			timesteps (optional): either int or numpy array/Variable of [batch] specfiying sequence length
				of recurrent modeling.
		Arguments:
			arg_map (required): Dictionary containing the name of an input as key and its index in *inputs.
			transpose: whether to transpose hidden state to [layers,batch,hidden_dim].
				Useful if using an opennmt style stacked rnn. Default: True
			timesteps (optional): same as timesteps in input. If timesteps is specified in neither
				location then number of iterations defaults to x.size(1)-1.
			loss_fn (optional): if provided, will calculate loss and return it with the output.
				Needs same __call__ inputs as torch.nn.CrossEntropyLoss.
			return_sequence (bool): whether to return full sequence of outputs/hiddens in [batch,time,dim]
				shape or just return last output and hidden state. Default: False
		Returns: (out,hidden),[loss]
			out: either last output or stacked tensor of all outputs based one value of return_sequence
			hidden: either last hidden state or stacked tensor of outputs based on value of return_sequence.
				If hidden is tuple of (cell,hidden) state then (cells,hiddens) stacked tensors are returned.
			loss: if loss_fn is provided then loss is returned, else it's not included in output
		"""
		unpacked_args = self.unpack_args(*inputs, **kwargs)
		(x, hidden, transpose, state_mask,
			mask_state,	timesteps, num_iters, seq_mask,
			loss_fn, calc_loss, return_sequence) = unpacked_args
		#check if hidden state is tuple or just cell state (rnn)
		hidden_is_tuple = isinstance(hidden, tuple)

		loss = 0.

		hiddens = []
		outputs = []

		# iterate over sequence
		# print(x.get_device(), self.embedder.weight.get_device(), [x.get_device() for x in self.rnn.parameters()])
		for t in range(num_iters):
			# embed t'th data point
			emb = self.embedder(x[:,t])

			cell_state = hidden[0] if hidden_is_tuple else hidden

			#handle reseting/persistence of states
			if mask_state:
				hidden_mask = state_mask[:,t].type_as(cell_state.data).unsqueeze(1)
				hidden_mask = hidden_mask.expand_as(cell_state)

				# maybe reset, maybe persist cell state based on mask
				reset_cell_state = cell_state*hidden_mask
				hidden = (reset_cell_state, hidden[1]*hidden_mask) if hidden_is_tuple else reset_cell_state

			# recurrent computation
			_hidden, _output = self.rnn(emb, hidden)

			cell_state = _hidden[0] if hidden_is_tuple else _hidden

			#mask output if the sequence has ended and we're past the sequence legnth according to timesteps.
			#propagate hidden state/output from last valid token in sequence
			if seq_mask:
				dont_mask = (t < timesteps).type_as(cell_state.data).unsqueeze(1)
				cell_dont_mask = dont_mask.expand_as(cell_state)
				dont_mask = dont_mask.expand_as(_output)
				if t != 0:
					output = _output*dont_mask + output*(1 - dont_mask)
				else:
					output = _output

				new_cell_state = cell_state*cell_dont_mask + cell_state*(1 - cell_dont_mask)
				hidden = new_cell_state if not hidden_is_tuple else \
						(new_cell_state, _hidden[1]*cell_dont_mask + hidden[1]*(1 - cell_dont_mask))
			else:
				hidden = _hidden
				output = _output

			if return_sequence:
				hiddens.append(hidden)
				outputs.append(output)

			if calc_loss:
				loss += loss_fn(output, x[:,t+1])

		if return_sequence:
			# concatenate hidden/output lists into tensors of [batch,time,dim]
			hiddens = (torch.transpose(torch.stack(list(map(lambda x: x[0], hiddens))).squeeze(), 0, 1),
				torch.transpose(torch.stack(list(map(lambda x: x[1], hiddens))).squeeze(), 0, 1))
			outputs = torch.transpose(torch.stack(outputs), 0, 1)
			rtn = (hiddens, outputs)
		else:
			rtn = (tuple([y.transpose(0, 1) for y in hidden]), output) if transpose else (hidden, output)

		if calc_loss:
			#average loss over time dimension
			#no need to worry about output masking since reconstruction has all the same lengths
			rtn = (rtn, loss / (x.size(1)-1))

		return rtn


	def state0(self, batch_size, make_cuda=False, volatile=False):
		"""gets initial state for a batch of size `batch_size`"""
		hidden_init = self.rnn.state0(batch_size, volatile=volatile)
		if make_cuda:
			hidden_init = make_state_cuda(self.hidden_init)
		return hidden_init
