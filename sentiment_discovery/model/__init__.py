import torch
import torch.optim as optim

from .model_wrapper import ModelWrapper
from .sequence_model import SequenceModel
from .run import run_model, epoch_loop
from .serialize import save, restore

def make_model(cell_type='mlstm', num_layers=1, embed_size=64,
				hidden_size=4096, data_size=256, dropout=0, fused=False,
				weight_norm=False, lstm_only=False, saved_path=''):

	emb_module = torch.nn.Embedding(data_size, embed_size)
	recurrent_module = SequenceModel(
							embed=emb_module, cell=cell_type,
							n_layers=num_layers, in_size=embed_size,
							rnn_size=hidden_size, out_size=data_size,
							dropout=dropout, fused=fused)
	model = ModelWrapper(recurrent_module, lstm_only=lstm_only)
	if weight_norm:
		model.apply_weight_norm()

	chkpt = None
	if saved_path != '':
		chkpt = restore(model, saved_path)


	return model, recurrent_module, emb_module, chkpt
