import os 
import sys
import time
import torch

from experiment_logger import Logger

class SystemConfig(object):
	def __init__(self, parser):
		super(SystemConfig, self).__init__()
		self.parser = parser
	def apply(self, cfg, opt):
		"""initalizes stdout/err, distributed processes, random seeding, and logging"""
		handle_verbose(cfg, opt)
		print('configuring system')
		if opt.distributed:
			# let rank 1 create directory structure to be used to hold distributedsetup file
			while not os.path.exists(os.path.join(os.getcwd(), '.distributed', opt.experiment_name)):
				if opt.rank <= 0:
					os.makedirs(os.path.join(os.getcwd(), '.distributed', opt.experiment_name))
				else:
					time.sleep(1)
			p = 'file://%s'%(os.path.join(os.getcwd(), '.distributed', opt.experiment_name, 'distributed_output.dpt'),)
			torch.distributed.init_process_group(backend='gloo', world_size=opt.world_size,
													init_method=p, rank=opt.rank)
		torch.manual_seed(opt.seed)
		if opt.cuda:
			torch.cuda.manual_seed(opt.seed)

		cfg.distributed = opt.distributed

		make_logger(cfg, opt)

def handle_verbose(cfg, opt):
	if opt.verbose == 1:
		if opt.rank > 0:
			disable_stdout()
	elif opt.verbose == 0:
		disable_stdout()
		disable_stderr()

def disable_stdout():
	sys.stdout = open(os.devnull, 'w')

def disable_stderr():
	sys.stderr = open(os.devnull, 'w')

def make_logger(cfg, opt):
	# if not torch.distributed._initialized or torch.distributed.rank == 0:
		# cfg.logger = Logger(opt.experiment_name, opt.experiment_dir)
	cfg.logger = Logger(opt.experiment_name, opt.experiment_dir)

def configure_system(parser):
	"""add cmdline flags for configuring system and communication with other systems"""
	parser.add_argument('-rank', type=int, default=-1,
						help='rank to be assigned to distributed process group')
	parser.add_argument('-distributed', action='store_true',
						help='whether to run in distributed data parallel mode')
	parser.add_argument('-world_size', type=int, default=2,
						help='world size of distributed execution')
	parser.add_argument('-verbose', type=int, default=1,
						help='which processes should output to std. 0=no processes, 1=master process, 2=all processes.\
						Useful for avoiding unnecessary stdout during distributed training.')
	parser.add_argument('-seed', type=int, default=1234,
						help='random seed')
	parser.add_argument('-experiment_dir', default='./experiments',
						help='Root directory for saving results, models, and figures')
	parser.add_argument('-experiment_name', default='unsupervised_sentiment_discovery',
						help='Name of experiment used for logging to `<experiment_dir>/<experiment_name>`.')
	return SystemConfig(parser)
