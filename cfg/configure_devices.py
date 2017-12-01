import sys

class DeviceConfig(object):
	def __init__(self, parser):
		super(DeviceConfig, self).__init__()
		self.parser = parser
	def apply(self, cfg, opt):
		print('configuring devices')
		cfg.batch_size = opt.batch_size
		if opt.cuda and opt.num_gpus > 1 or opt.distributed:
			cfg.batch_size *= opt.num_gpus
		cfg.model.initialize(opt.batch_size if opt.distributed else cfg.batch_size, volatile=opt.should_test)
		cfg.model.cuda()
		if opt.cuda and opt.num_gpus > 1 or opt.distributed:
			cfg.model.make_data_parallel(opt.num_gpus, distributed=opt.distributed,
										rank=opt.rank, world_size=opt.world_size)

def configure_devices(parser):
	parser.add_argument('-cuda', action='store_true',
						help="Use CUDA")
	parser.add_argument('-num_gpus', type=int, default=1,
						help="number of gpus to use for data parallelism")
	parser.add_argument('-benchmark', action='store_true',
						help="turn on torch.nn.cuda.benchmark")
	return DeviceConfig(parser)
