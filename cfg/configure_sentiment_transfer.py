class ScriptConfig(object):
	def __init__(self):
		super(ScriptConfig, self).__init__()
	def apply(self, cfg, opt):
		pass

def script_config(parser):
	"""creates transfer-specific flags and sets default values of other flags"""
	parser.add_argument('-num_neurons', type=int, default=1,
							help='Number of neurons to consider as a sentiment neuron')
	parser.add_argument('-no_test_eval', action='store_true',
						help='whether to not evaluate the test model (useful when your test set has no labels)')
	parser.add_argument('-write_results', default='',
						help='write results of model on test (or train if none is specified) data to specified filepath')

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
	# increased batch size
	parser.set_defaults(batch_size=128)
	parser.set_defaults(eval_batch_size=0)
	parser.set_defaults(data_size=256)
	parser.set_defaults(seq_length=256)
	parser.set_defaults(eval_seq_length=0)
	# set data_set_type
	parser.set_defaults(data_set_type='supervised')
	parser.set_defaults(persist_state=0)
	# set transpose
	parser.set_defaults(transpose=False)
	parser.set_defaults(no_wrap=False)
	parser.set_defaults(cache=False)

	# data processing flags
	parser.set_defaults(lazy=False)
	parser.set_defaults(preprocess=True)
	# set no shuffle
	parser.set_defaults(shuffle=False)
	parser.set_defaults(text_key='sentence')
	parser.set_defaults(eval_text_key='None')
	parser.set_defaults(label_key='label')
	parser.set_defaults(eval_label_key='None')
	parser.set_defaults(delim=',')
	parser.set_defaults(drop_unlabeled=False)
	parser.set_defaults(binarize_sent=False)
	parser.set_defaults(num_shards=1)

	# dataset path flags
	# set datasets
	parser.set_defaults(train='./data/binary_sst/train.csv')
	parser.set_defaults(split='1000,1,1')
	parser.set_defaults(valid='./data/binary_sst/val.csv')
	parser.set_defaults(test='./data/binary_sst/test.csv')

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
