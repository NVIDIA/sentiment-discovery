import torch

# from torch.nn._functions.rnn import LSTMCell, RNNReLUCell, RNNTanhCell, GRUCell

from .RNNBackend import bidirectionalRNN, stackedRNN, RNNCell
from .cells import mLSTMRNNCell, mLSTMCell

_VF = torch._C._VariableFunctions
_rnn_impls = {
    'LSTM': _VF.lstm_cell,
    'GRU': _VF.gru_cell,
    'RNN_TANH': _VF.rnn_tanh_cell,
    'RNN_RELU': _VF.rnn_relu_cell,
}

def toRNNBackend(inputRNN, num_layers, bidirectional=False, dropout = 0):
    """
    toRNNBackend stub
    """

    if bidirectional:
        return bidirectionalRNN(inputRNN, num_layers, dropout = dropout)
    else:
        return stackedRNN(inputRNN, num_layers, dropout = dropout)


def LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, output_size = None):
    """
    LSTM stub
    """
    inputRNN = RNNCell(4, input_size, hidden_size, _rnn_impls['LSTM'], 2, bias, output_size)
    # inputRNN = RNNCell(4, input_size, hidden_size, LSTMCell, 2, bias, output_size)
    return toRNNBackend(inputRNN, num_layers, bidirectional, dropout=dropout)

def GRU(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, output_size = None):
    """
    GRU stub
    """
    inputRNN = RNNCell(3, input_size, hidden_size, _rnn_impls['GRU'], 1, bias, output_size)
    # inputRNN = RNNCell(3, input_size, hidden_size, GRUCell, 1, bias, output_size)
    return toRNNBackend(inputRNN, num_layers, bidirectional, dropout=dropout)

def ReLU(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, output_size = None):
    """
    ReLU stub
    """
    inputRNN = RNNCell(1, input_size, hidden_size, _rnn_impls['RNN_RELU'], 1, bias, output_size)
    # inputRNN = RNNCell(1, input_size, hidden_size, RNNReLUCell, 1, bias, output_size)
    return toRNNBackend(inputRNN, num_layers, bidirectional, dropout=dropout)

def Tanh(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, output_size = None):
    """
    Tanh stub
    """
    inputRNN = RNNCell(1, input_size, hidden_size, _rnn_impls['RNN_TANH'], 1, bias, output_size)
    inputRNN = RNNCell(1, input_size, hidden_size, RNNTanhCell, 1, bias, output_size)
    return toRNNBackend(inputRNN, num_layers, bidirectional, dropout=dropout)
        
def mLSTM(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, output_size = None):
    """
    mLSTM stub
    """
    print("Creating mlstm")
    inputRNN = mLSTMRNNCell(input_size, hidden_size, bias=bias, output_size=output_size)
    return toRNNBackend(inputRNN, num_layers, bidirectional, dropout=dropout)
