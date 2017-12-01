class ScriptConfig(object):
	def __init__(self):
		super(ScriptConfig, self).__init__()
	def apply(self, cfg, opt):
		pass

def script_config(parser):
	"""creates visualization-specific flags and sets default values of other flags"""
	parser.add_argument('-text', default='The meaning of life is ',
					help="""Initial text """)
	parser.add_argument('-generate', action='store_true',
					help="""after processing text specified by `text` it generates following text up
							to a total length of `seq_length`""")
	parser.add_argument('-temperature', type=float, default=0.4,
					help="""Temperature for sampling.""")
	parser.add_argument('-neuron', type=int, default=0,
					help="""Neuron to read.""")
	parser.add_argument('-overwrite', type=float, default=0,
					help="""If `generate` is specified then overwrite the neuron. Used to make +/- leaning sentiment.
							0 means don't overwrite.""")
	parser.add_argument('-negate', action='store_true',
					help="""If `neuron` corresponds to a negative sentiment neuron rather than positive""")
	parser.add_argument('-layer', type=int, default=-1,
					help="""Layer to read. -1 = last layer""")

	# experiment flags
	parser.set_defaults(experiment_dir='./experiments')
	parser.set_defaults(experiment_name='mlstm')

	# model flags
	parser.set_defaults(should_test=False)
	parser.set_defaults(embed_size=64)
	parser.set_defaults(rnn_type='mlstm')
	parser.set_defaults(rnn_size=4096)
	parser.set_defaults(layers=1)
	parser.set_defaults(dropout=0)
	parser.set_defaults(weight_norm=False)
	parser.set_defaults(lstm_only=False)
	parser.set_defaults(model_dir='model')
	parser.set_defaults(load_model='')

	# data flags
	parser.set_defaults(batch_size=1)
	parser.set_defaults(eval_batch_size=0)
	parser.set_defaults(data_size=256)
	parser.set_defaults(seq_length=256)
	parser.set_defaults(eval_seq_length=0)
	parser.set_defaults(data_set_type='unsupervised')
	parser.set_defaults(persist_state=0)
	parser.set_defaults(transpose=True)
	parser.set_defaults(no_wrap=False)
	parser.set_defaults(cache=False)

	# data processing flags
	parser.set_defaults(lazy=False)
	parser.set_defaults(preprocess=False)
	parser.set_defaults(shuffle=True)
	parser.set_defaults(text_key='sentence')
	parser.set_defaults(eval_text_key='None')
	parser.set_defaults(label_key='label')
	parser.set_defaults(eval_label_key='None')
	parser.set_defaults(delim=',')
	parser.set_defaults(drop_unlabeled=False)
	parser.set_defaults(binarize_sent=False)

	# dataset path flags
	# set no datasets
	parser.set_defaults(train='None')
	parser.set_defaults(split=1.)
	parser.set_defaults(valid='None')
	parser.set_defaults(test='None')

	# device flags
	parser.set_defaults(cuda=False)
	parser.set_defaults(benchmark=False)
	parser.set_defaults(num_gpus=1)

	# system flags
	parser.set_defaults(rank=-1)
	parser.set_defaults(distributed=False)
	parser.set_defaults(world_size=2)
	parser.set_defaults(verbose=1)
	parser.set_defaults(seed=1234)
	return ScriptConfig()
