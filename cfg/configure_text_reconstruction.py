import os
import math

import torch

import sentiment_discovery.learning_rates as learning_rates
from sentiment_discovery.model import save

class ScriptConfig(object):
	def __init__(self):
		super(ScriptConfig, self).__init__()
	def apply(self, cfg, opt):
		print('configuring learning')
		if not opt.no_loss:
			loss_fn = torch.nn.CrossEntropyLoss()
			cfg.model.set_loss_fn(loss_fn)

		cfg.n_batch = num_batches(cfg.train, cfg, opt)
		cfg.nv_batch = num_batches(cfg.valid, cfg, opt)
		cfg.nt_batch = num_batches(cfg.test, cfg, opt)

		cfg.e = opt.start_epoch
		if cfg.chkpt is not None:
			if 'epoch' in cfg.chkpt and opt.start_epoch == 0:
				cfg.e = cfg.chkpt['epoch']

		cfg.outer_loop = get_iter(cfg, opt)

		cfg.lr = None
		cfg.initial_lr = opt.lr
		cfg.inner_lr = None
		if not opt.should_test:

			if cfg.chkpt is not None and 'optim' in cfg.chkpt:
				cfg.model.add_optimizer(cfg.chkpt['optim'], load_optim=True,
					lr=cfg.initial_lr, clip=opt.clip)
			else:
				assert opt.optimizer_type != 'None'
				cfg.model.add_optimizer(eval('torch.optim.%s'%(opt.optimizer_type,)),
					lr=cfg.initial_lr, clip=opt.clip)

			if not opt.no_loss:
				cfg.lr = learning_rates.make(opt.lr_scheduler, start_epoch=cfg.e, lr_factor=opt.lr_factor, start_iter=opt.start_iter,
					max_iters=math.ceil(cfg.n_batch/float(cfg.batch_size)) if opt.max_iters != -1 else opt.max_iters)(cfg.model.optimizer)
			if opt.lr_scheduler == 'linear' and cfg.lr is not None:
				cfg.inner_lr = lambda: cfg.lr.step()
			else:
				cfg.inner_lr = lambda: None
		cfg.optim = cfg.model.optimizer

		cfg.saver = make_saver(cfg, opt)

def make_saver(cfg, opt):
	"""returns callable that saves an instance of the model to the experiment's model directory"""
	cfg.histories = []
	if cfg.chkpt is not None and 'histories' in cfg.chkpt:
		cfg.histories = cfg.chkpt['histories']
	def _saver(basename, history):
		epoch = cfg.e
		if len(cfg.histories) <= epoch:
			cfg.histories.append(history)
		else:
			cfg.histories[-1] = history
		if not opt.distributed or opt.rank == 0:
			checkpoint = {
						'state_dict': cfg.model.state_dict(),
						'opt': cfg.opt,
						'epoch': epoch,
						'optim' : cfg.optim,
						'histories': cfg.histories
					}
			save(cfg.model, os.path.join(cfg.logger.get_log_dir(opt.model_dir), basename), save_dict=checkpoint)
			cfg.logger.log_pkl(cfg.histories, 'histories', os.path.splitext(basename)[0]+'.pkl', 'wb')
	return _saver

def num_batches(loader, cfg, opt):
	n_batch = 0
	if loader is not None:
		n_batch = len(loader)
	return n_batch

def get_iter(cfg, opt):
	"""makes outer training loop"""
	ctr = cfg.e
	while stop_cond(cfg, opt):
		cfg.e = ctr
		yield ctr
		ctr += 1

def stop_cond(cfg, opt):
	"""determines when outer training loop should stop"""
	if opt.max_iters != -1:
		cfg.e < opt.max_iters//cfg.n_batch
	if opt.epochs == -1:
		return True
	return cfg.e < opt.epochs

def script_config(parser):
	"""creates flags for language modeling and sets default values of other flags"""
	parser.add_argument('-start_epoch', type=int, default=0,
						help='epoch to start training at (used to resume training for exponential decay scheduler)')
	parser.add_argument('-start_iter', type=int, default=0,
						help='what iteration to start training at (used mainly for resuming linear learning rate scheduler)')
	parser.add_argument('-epochs', type=int, default=100,
						help='number of epochs to train for')
	parser.add_argument('-max_iters', type=int, default=-1,
						help='total number of training iterations to run')
	parser.add_argument('-lr', type=float, default=0.0005,
						help="""Starting learning rate.""")
	parser.add_argument('--optimizer_type', default='SGD',
						help='Class name of optimizer to use as listed in torch.optim class \
						 (ie. SGD, RMSProp, Adam)')
	parser.add_argument('-lr_scheduler', default='ExponentialLR',
						help='one of [ExponentialLR,LinearLR]')
	parser.add_argument('-lr_factor', default=.95, type=int,
						help='factor by which to decay learning rate')
	parser.add_argument('-clip', type=float, default=1.,
						help="""Clip gradients at this value.""")
	parser.add_argument('-no_loss', action='store_true',
						help='whether or not to track the loss curve of the model')

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
	parser.set_defaults(batch_size=32)
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
	parser.set_defaults(shuffle=False)
	parser.set_defaults(text_key='sentence')
	parser.set_defaults(eval_text_key='None')
	parser.set_defaults(label_key='label')
	parser.set_defaults(eval_label_key='None')
	parser.set_defaults(delim=',')
	parser.set_defaults(drop_unlabeled=False)
	parser.set_defaults(binarize_sent=False)

	# dataset path flags
	parser.set_defaults(train='./data/imdb/unsup.json')
	parser.set_defaults(split='1.')
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
