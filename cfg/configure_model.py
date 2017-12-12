import os

from sentiment_discovery.reparameterization import remove_weight_norm
from sentiment_discovery.model import make_model

class ModuleConfig(object):
	def __init__(self, parser):
		super(ModuleConfig, self).__init__()
		self.parser = parser
	def apply(self, cfg, opt):
		"""make model and format model path for reloading parameters"""
		print('configuring model')
		cell_type = opt.rnn_type
		num_layers = opt.layers
		embed_size = opt.embed_size
		hidden_size = opt.rnn_size
		# set in configure_data
		data_size = opt.data_size
		dropout = opt.dropout
		w_norm = opt.weight_norm
		lstm_only = opt.lstm_only
		fused = opt.fuse_lstm

		saved_path = ''
		if opt.load_model != '':
			model_dir = cfg.logger.get_log_dir(opt.model_dir)
			saved_path = os.path.join(model_dir, opt.load_model)
		print(embed_size)
		model, recurrent_module, embedder_module, chkpt = make_model(
			cell_type=cell_type, fused=fused, num_layers=num_layers,
			embed_size=embed_size, hidden_size=hidden_size,
			data_size=data_size, dropout=dropout, weight_norm=w_norm,
			lstm_only=lstm_only, saved_path=saved_path)
		cfg.model = model
		cfg.chkpt = chkpt

		nParams = sum([p.nelement() for p in cfg.model.parameters()])
		print('* number of parameters: %d' % nParams)

def configure_model(parser):
	"""add cmdline args for configuring models"""
	parser.add_argument('-load_model', default='',
						help="""a specific checkpoint file to load from experiment's model directory""")
	parser.add_argument('-should_test', action='store_true',
					help='whether to train or evaluate a model')
	parser.add_argument('-model_dir', default='model',
						help='directory where models are saved to/loaded from')
	parser.add_argument('-rnn_type', default='mlstm',
						help='mlstm, lstm or gru')
	parser.add_argument('-fuse_lstm', action='store_true',
						help='use fused lstm cuda kernels in mLSTM')
	parser.add_argument('-layers', type=int, default=1,
						help='Number of layers in the rnn')
	parser.add_argument('-rnn_size', type=int, default=4096,
						help='Size of hidden states')
	parser.add_argument('-embed_size', type=int, default=64,
						help='Size of embeddings')
	parser.add_argument('-weight_norm', action='store_true',
						help='whether to use weight normalization for training NNs')
	parser.add_argument('-lstm_only', action='store_true',
						help='if `-weight_norm` is applied to the model, apply it to the lstm parmeters only')
	parser.add_argument('-dropout', type=float, default=0.1,
						help='Dropout probability.')
	return ModuleConfig(parser)
