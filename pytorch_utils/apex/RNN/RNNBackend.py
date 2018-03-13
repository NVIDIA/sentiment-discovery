import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F

import math


#This function could have some real bad perf penalties if used incorrectly
#Uses in the RNN API should be fine. DIDN'T USE!
def reverse_dir_tensor(tensor, dim=0):
    """
    reverse_dir_tensor stub
    """
    chunked = [sub_tensor for sub_tensor in
               tensor.chunk(tensor.size(dim), dim)]
    chunked = chunked[::-1]
    return torch.cat( chunked, dim=dim).view(*tensor.size())

def flatten_list(tens_list):
    """
    flatten_list stub
    """
    if not ( isinstance(tens_list, tuple) or isinstance(tens_list, list) ):
        return tens_list
    
    return torch.cat(tens_list, dim=0).view(len(tens_list), *tens_list[0].size() )

#These modules always assumes batch_first
class bidirectionalRNN(nn.Module):
    """
    bidirectionalRNN stub
    """
    def __init__(self, inputRNN, num_layers=1, dropout = 0):
        super(bidirectionalRNN, self).__init__()
        self.dropout = dropout
        self.fwd = stackedRNN(inputRNN, num_layers=num_layers, dropout = dropout)
        self.bckwrd = stackedRNN(inputRNN.new_like(), num_layers=num_layers, dropout = dropout)
        self.rnns = nn.ModuleList([self.fwd, self.bckwrd])
        
    #collect hidden option will return all hidden/cell states from entire RNN
    def forward(self, input, collectHidden=False):
        """
        forward() stub
        """
        seq_len = input.size(0)
        bsz = input.size(1)
        
        fwd_out, fwd_hiddens = list(self.fwd(input, collectHidden = collectHidden))
        bckwrd_out, bckwrd_hiddens = list(self.bckwrd(input, reverse=True, collectHidden = collectHidden))

        output = torch.cat( [fwd_out, bckwrd_out], -1 )
        hiddens = tuple( torch.cat(hidden, -1) for hidden in zip( fwd_hiddens, bckwrd_hiddens) )

        return output, hiddens

    def reset_parameters(self):
        """
        reset_parameters() stub
        """
        for rnn in self.rnns:
            rnn.reset_parameters()
        
    def init_hidden(self, bsz):
        """
        init_hidden() stub
        """
        for rnn in self.rnns:
            rnn.init_hidden(bsz)

    def detach_hidden(self):
        """
        detach_hidden() stub
        """
        for rnn in self.rnns:
            rnn.detachHidden()
        
    def reset_hidden(self, bsz):
        """
        reset_hidden() stub
        """
        for rnn in self.rnns:
            rnn.reset_hidden(bsz)

    def init_inference(self, bsz):    
        """
        init_inference() stub
        """
        for rnn in self.rnns:
            rnn.init_inference(bsz)

   
#assumes hidden_state[0] of inputRNN is output hidden state
#constructor either takes an RNNCell or list of RNN layers
class stackedRNN(nn.Module):        
    """
    stackedRNN stub
    """
    def __init__(self, inputRNN, num_layers=1, dropout=0):
        super(stackedRNN, self).__init__()
        
        self.dropout = dropout
        
        if isinstance(inputRNN, RNNCell):
            self.rnns = [inputRNN]
            for i in range(num_layers-1):
                self.rnns.append(inputRNN.new_like(inputRNN.output_size))
        elif isinstance(inputRNN, list):
            assert len(inputRNN) == num_layers, "RNN list length must be equal to num_layers"
            self.rnns=inputRNN
        else:
            raise RuntimeError()
        
        self.nLayers = len(self.rnns)
        
        #for i, rnn in enumerate(self.rnns):
        #    self.add_module("rnn_layer"+str(i), self.rnns[i])
        self.rnns = nn.ModuleList(self.rnns)


    '''
    Returns output as hidden_state[0] Tensor([sequence steps][batch size][features])
    If collect hidden will also return Tuple(
        [n_hidden_states][layer] Tensor([sequence steps][batch size][features])
    )
    If not collect hidden will also return Tuple(
        [n_hidden_states][layer] Tensor([batch size][features])
    '''
    def forward(self, input, collectHidden=False, reverse=False, reset_mask=None):
        """
        forward() stub
        """
        
        seq_len = input.size(0)
        bsz = input.size(1)

        hidden_states = [ ]

        #Treat first layer independently, needs to be reversed if reverse. Rest
        #is the same with reverse or not.

        layer_output = []
        
        if not reverse:
            for i in range(seq_len):
                if reset_mask is not None:
                    self.rnns[0].reset_hidden(bsz, reset_mask=reset_mask[i])
                layer_output.append(self.rnns[0](input[i]))
        else:
            for i in reversed(range(seq_len)):
                layer_output.append(self.rnns[0](input[i]))
                if reset_mask is not None:
                    self.rnns[0].reset_hidden(bsz, reset_mask=reset_mask[i])

        '''
        transpose output list
        list( [seq_length][hidden_states] x Tensor([bsz][features]) )
        to
        list( [hidden_states][seq_length] x Tensor([bsz][features]) )

        I always endup going through this trick everytime I use it so...

        >>> list_of_list = [ [ i+j*4 for i in range(4) ] for j in range (3) ]
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

        >>> list_of_lists = list( list( entry ) for entry in zip(*list_of_lists) )
            [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]

        >>> list_of_lists = list( list( entry ) for entry in zip(*list_of_lists) )
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

        '''
        layer_output = list( list(entry) for entry in zip(*layer_output) )
        hidden_states.append( layer_output )
                
        for layer in range(1, self.nLayers):
            
            layer_output = []
            #Grab last layers hidden states, assume first hidden state was output/input to next layer
            cur_input = hidden_states[-1][0]
            for inp in cur_input:
                if self.dropout > 0.0 and self.dropout < 1.0:
                    inp = F.dropout(inp, self.dropout, self.training, False)
                if not reverse:
                    if reset_mask is not None:
                        self.rnns[layer].reset_hidden(bsz, reset_mask=reset_mask[i])
                layer_output.append( self.rnns[layer](inp) )
                if reverse:
                    if reset_mask is not None:
                        self.rnns[layer].reset_hidden(bsz, reset_mask=reset_mask[i])

            #transpose output like above
            hidden_states.append( [list(tensors) for tensors in zip(*layer_output) ] )

        #reverse seq order if reverse=True
        if reverse:
            #for layer in range(len(hidden_states)):
            #    for hidden_state in range(len(hidden_states[layer])):
            #        hidden_states[layer][hidden_state] = list(reversed(hidden_states[layer][hidden_state]))
            hidden_states = [ [ [ step for step in reversed(hidden) ]
                                           for hidden in layer ]
                                               for layer in hidden_states ]

        output = hidden_states[-1][0]
        output = torch.cat(output, dim=0).view(seq_len, bsz, -1)
        
        '''
        transpose hidden_states (use trick above) makes list comprehensions straight foreward
        list( [layer][hidden_states][seq_length] x Tensor([bsz][features]) )
        list( [hidden_states][layer][seq_length] x Tensor([bsz][features]) )
        '''
        hidden_states = list( list(entry) for entry in zip(*hidden_states) )

        if not collectHidden:
            hiddens = list( list( layer[-1] for layer in hidden ) for hidden in hidden_states ) 
            '''
            add seq_length into tensor:
            tuple( [hidden_states][seq_length] x Tensor([bsz][features] )
            to
            tuple( [hidden_states] x Tensor([seq_length][bsz][features] )
            '''
            hiddens = tuple( flatten_list( hidden ) for hidden in hiddens )
            return output, hiddens

        else:
            '''
            we want everything returned as
            list( [hidden_states][layer] x Tensor([seq_length][bsz][features]) )
            '''
            hiddens = list( list( flatten_list(layer) for layer in hidden ) for hidden in hidden_states ) 
            return output, hiddens
    
    def reset_parameters(self):
        """
        reset_parameters() stub
        """
        for rnn in self.rnns:
            rnn.reset_parameters()
        
    def init_hidden(self, bsz):
        """
        init_hidden() stub
        """
        for rnn in self.rnns:
            rnn.init_hidden(bsz)

    def detach_hidden(self):
        """
        detach_hidden() stub
        """
        for rnn in self.rnns:
            rnn.detach_hidden()
        
    def reset_hidden(self, bsz):
        """
        reset_hidden() stub
        """
        for rnn in self.rnns:
            rnn.reset_hidden(bsz)

    def init_inference(self, bsz):    
        """ 
        init_inference() stub
        """
        for rnn in self.rnns:
            rnn.init_inference(bsz)

class RNNCell(nn.Module):
    """ 
    RNNCell stub
    gate_multiplier is related to the architecture you're working with
    For LSTM-like it will be 4 and GRU-like will be 3.
    Always assumes input is NOT batch_first.
    Output size that's not hidden size will use output projection
    Hidden_states is number of hidden states that are needed for cell
    if one will go directly to cell as tensor, if more will go as list
    """
    def __init__(self, gate_multiplier, input_size, hidden_size, cell, n_hidden_states = 2, bias = False, output_size = None):
        super(RNNCell, self).__init__()

        self.gate_multiplier = gate_multiplier
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = cell
        self.bias = bias
        self.output_size = output_size
        if output_size is None:
            self.output_size = hidden_size

        self.gate_size = gate_multiplier * self.hidden_size
        self.n_hidden_states = n_hidden_states

        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.output_size))

        #Check if there's recurrent projection
        if(self.output_size != self.hidden_size):
            self.w_ho = nn.Parameter(torch.Tensor(self.output_size, self.hidden_size))

        self.b_ih = self.b_hh = None
        if self.bias:
            self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
            self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
            
        #hidden states for forward
        self.hidden = [ None for states in range(self.n_hidden_states)]

        self.reset_parameters()

    def new_like(self, new_input_size=None):
        """
        new_like() stub
        """
        if new_input_size is None:
            new_input_size = self.input_size
            
        return type(self)(self.gate_multiplier,
                       new_input_size,
                       self.hidden_size,
                       self.cell,
                       self.n_hidden_states,
                       self.bias,
                       self.output_size)

    
    #Use xavier where we can (weights), otherwise use uniform (bias)
    def reset_parameters(self, gain=1):
        """
        reset_parameters() stub
        """
        stdev = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            param.data.uniform_(-stdev, stdev)
    '''
    def reset_parameters(self, gain=1):
        stdv = 1.0 / math.sqrt(self.gate_size)

        self.w_ih.uniform_(-stdv, stdv)
        self.w_hh.uniform_(-stdv, stdv)
        if self.bias:
            self.b_ih.uniform_(-stdv/2, stdv/2)
            self.b_hh.uniform_(-stdv/2, stdv/2)
        
        #for param in self.parameters():
        #    #if (param.dim() > 1):
        #    #    torch.nn.init.xavier_normal(param, gain)
        #    #else:
        #    param.data.uniform_(-stdv, stdv)
    '''
    def init_hidden(self, bsz):
        """
        init_hidden() stub
        """
        for param in self.parameters():
            if param is not None:
                a_param = param
                break

        for i, _ in enumerate(self.hidden):
            if(self.hidden[i] is None or self.hidden[i].data.size()[0] != bsz):

                if i==0:
                    hidden_size = self.output_size
                else:
                    hidden_size = self.hidden_size

                tens = a_param.data.new(bsz, hidden_size).zero_()
                self.hidden[i] = Variable(tens, requires_grad=False)
        
    def reset_hidden(self, bsz, reset_mask=None):
        """
        reset_hidden() stub
        """
        if reset_mask is not None:
            if (reset_mask != 0).any():
                for i, v in enumerate(self.hidden):
                    if reset_mask.numel() == 1:
                        self.hidden[i] = v.data.zero_()
                    else:
                        reset_mask = reset_mask.view(self.hidden[i].size(0), 1).contiguous()
                        self.hidden[i] = v * (1 - reset_mask).type_as(v.data)
            return

        for i, _ in enumerate(self.hidden):
            self.hidden[i] = None
        self.init_hidden(bsz)

    def detach_hidden(self):
        """
        detach_hidden() stub
        """
        for i, _ in enumerate(self.hidden):
            if self.hidden[i] is None:
                raise RuntimeError("Must inialize hidden state before you can detach it")
        for i, _ in enumerate(self.hidden):
            self.hidden[i] = self.hidden[i].detach()
        
    def forward(self, input):
        """
        forward() stub
        if not inited or bsz has changed this will create hidden states
        """
        self.init_hidden(input.size()[0])

        hidden_state = self.hidden[0] if self.n_hidden_states == 1 else self.hidden

        self.hidden = list( self.cell(input, hidden_state, self.w_ih, self.w_hh, b_ih=self.b_ih, b_hh=self.b_hh) )

        if self.output_size != self.hidden_size:
            self.hidden[0] = F.linear(self.hidden[0], self.w_ho)

        return tuple(self.hidden)
