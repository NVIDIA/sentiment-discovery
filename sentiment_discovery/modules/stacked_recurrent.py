import torch
import torch.nn as nn
from torch.autograd import Variable

class StackedLSTM(nn.Module):
	"""
	Based on implementation of StackedLSTM from openNMT-py
	https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/StackedRNN.py
	Args:
		cell: LSTM/mLSTM cell u want to create. Callable of form `f(input, rnn_size)`
		num_layers: how many of these cells to stack
		input: The dimension of the input to the module
		rnn_size: The number of features in the hidden states of the lstm cells
		output_size: dimension of linear transformation layer on the output of the stacked rnn cells.
			If <=0 then no output layer is applied. Default: -1
		drop_out: probability of dropout layer (applied after rnn, but before output layer). Default: 0
		bias: If `False`, then the layer does not use bias weights for the linear transformation from
			multiplicative->hidden. Default: True
	Inputs: input, (h_0, c_0)
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
			If `output_size==-1` then this is equivalent to `h_1`.
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
	def __init__(self, cell, num_layers, input_size, rnn_size, 
				output_size=-1, dropout=0):
		super(StackedLSTM, self).__init__()

		self.add_module('dropout', nn.Dropout(dropout))
		self.num_layers = num_layers
		self.rnn_size = rnn_size
		if output_size > 0:
			self.add_module('h2o', nn.Linear(rnn_size, output_size))
		self.output_size = output_size
		self.add_module('layers', nn.ModuleList(
			[cell(input_size if x == 0 else rnn_size, rnn_size) for x in range(num_layers)]))

	def forward(self, input, hidden):
		x = input
		h_0, c_0 = hidden
		h_1, c_1 = [], []
		# iterate over layers and propagate hidden state through to top rnn
		for i, layer in enumerate(self.layers):
			h_1_i, c_1_i = layer(x, (h_0[i], c_0[i]))
			if i == 0:
				x = h_1_i
			else:
				x = x + h_1_i
			if i != len(self.layers):
				x = self.dropout(x)
			h_1 += [h_1_i]
			c_1 += [c_1_i]

		h_1 = torch.stack(h_1)
		c_1 = torch.stack(c_1)
		output = h_1
		if self.output_size > 0:
			output = self.h2o(x)

		return (h_1, c_1), output

	def state0(self, batch_size, volatile=False):
			"""
			Get initial hidden state tuple for mLSTMCell
			Args:
				batch_size: The minibatch size we intend to process
			Inputs: batch_size, volatile
				- **batch_size** : integer or scalar tensor representing the minibatch size
				- **volatile** : boolean whether to make hidden state volatile. (requires_grad=False)
			Outputs: h_0, c_0
				- **h_0** (num_layers, batch, hidden_size): tensor containing the next hidden state
				  for each element and layer in the batch
				- **c_0** (num_layers, batch, hidden_size): tensor containing the next cell state
				  for each element and layer in the batch
			"""
			h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.rnn_size),
					requires_grad=False, volatile=volatile)
			c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.rnn_size),
					requires_grad=False, volatile=volatile)
			return (h_0, c_0)
