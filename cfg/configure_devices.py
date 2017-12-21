import sys

class DeviceConfig(object):
	def __init__(self, parser):
		super(DeviceConfig, self).__init__()
		self.parser = parser
	def apply(self, cfg, opt):
		print('configuring devices')
		cfg.batch_size = opt.batch_size
		cfg.eval_batch_size = opt.eval_batch_size
		cfg.seq_length = opt.seq_length
		cfg.eval_seq_length = opt.eval_seq_length
		if opt.cuda and opt.num_gpus > 1 or opt.distributed:
			cfg.batch_size *= opt.num_gpus
			cfg.eval_batch_size *= opt.num_gpus
			if opt.seq_length < 0:
				cfg.seq_length *= opt.num_gpus
			if opt.eval_seq_length < 0:
				cfg.eval_seq_length *= opt.num_gpus
		cfg.model.initialize(opt.batch_size if opt.distributed else cfg.batch_size, volatile=opt.should_test)
		if opt.cuda or opt.fp16:
			cfg.model.cuda()
		if opt.fp16:
			cfg.model.half()
		if opt.cuda and opt.num_gpus > 1 or opt.distributed:
			cfg.model.make_data_parallel(opt.num_gpus, distributed=opt.distributed,
										rank=opt.rank, world_size=opt.world_size)

def configure_devices(parser):
	parser.add_argument('-cuda', action='store_true',
						help="Use CUDA")
	parser.add_argument('-fp16', action='store_true',
						help="Enable fp16 computation. Enables cuda gpu usage as well. (can't be used in conjunction with -fuse_lstm)")
	parser.add_argument('-num_gpus', type=int, default=1,
						help="number of gpus to use for data parallelism")
	parser.add_argument('-benchmark', action='store_true',
						help="turn on torch.nn.cuda.benchmark")
	return DeviceConfig(parser)
