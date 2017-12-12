import torch.nn as nn
import torch.nn.functional as F

class mLSTMCell(nn.Module):
	r"""
	A long short-term memory (LSTM) cell.
	.. math::
		\begin{array}{ll}

		implementation of
		Multiplicative LSTM.
		Ben Krause, Liang Lu, Iain Murray, and Steve Renals,
		"Multiplicative LSTM for sequence modelling, "
		https://arxiv.org/pdf/1609.07959.pdf
		\end{array}
	Args:
		data_size: The number of expected features in the input x
		hidden_size: The number of features in the hidden state h
		bias: If `False`, then the layer does not use bias weights for the linear transformation from
			multiplicative->hidden. Default: True
	Inputs: input, (h_0, c_0)
		- **input** (batch, input_size): tensor containing input features
		- **h_0** (batch, hidden_size): tensor containing the initial hidden
		  state for each element in the batch.
		- **c_0** (batch. hidden_size): tensor containing the initial cell state
		  for each element in the batch.
	Outputs: h_1, c_1
		- **h_1** (batch, hidden_size): tensor containing the next hidden state
		  for each element in the batch
		- **c_1** (batch, hidden_size): tensor containing the next cell state
		  for each element in the batch
	Examples::
		>>> rnn = nn.mLSTMCell(10, 20)
		>>> input = Variable(torch.randn(6, 3, 10))
		>>> hx = Variable(torch.randn(3, 20))
		>>> cx = Variable(torch.randn(3, 20))
		>>> output = []
		>>> for i in range(6):
		...     hx, cx = rnn(input[i], (hx, cx))
		...     output.append(hx)
	"""

	def __init__(self, data_size, hidden_size, bias=True, fused_lstm=False):
		super(mLSTMCell, self).__init__()

		self.hidden_size = hidden_size
		self.data_size = data_size
		self.bias = bias
		self.fused = fused_lstm

		self.add_module('wx', nn.Linear(data_size, 4*hidden_size, bias=False))
		self.add_module('wh', nn.Linear(hidden_size, 4*hidden_size, bias=bias))
		self.add_module('wmx', nn.Linear(data_size, hidden_size, bias=False))
		self.add_module('wmh', nn.Linear(hidden_size, hidden_size, bias=False))

	def forward(self, data, last_hidden):

		h0, c0 = last_hidden

		m = (self.wmx(data) * self.wmh(h0))

		if not self.fused:
			gates = self.wx(data) + self.wh(m)

			i, f, o, u = gates.chunk(4, 1)

			i = F.sigmoid(i)
			f = F.sigmoid(f)
			u = F.tanh(u)
			o = F.sigmoid(o)

			c1 = f * c0 + i * u
			h1 = o * F.tanh(c1)

			return h1, c1
		else:
			return self._backend.LSTMCell(
				data, (m, c0),
				self.wx.weight, self.wh.weight,
				self.wx.bias, self.wh.bias)
