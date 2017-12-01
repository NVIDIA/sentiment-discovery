from torch.optim.lr_scheduler import _LRScheduler

class LinearLR(_LRScheduler):
	"""
	A scheduler for linear learning rate decay to 0 over a specified number of steps.
	Args:
		optimizer (Optimizer): Wrapped optimizer.
		max_iters (int): Period of learning rate decay. When last_iter==max_iters lr=max(min_lr,0)
		last_iter (int): The index of last iteration step. Default: -1
		min_lr (float): smallest allowed learning rate (acts as a clamp to prevent too small learning rates). Default: 1e-8
	Example:
		>>> # Assuming optimizer also uses lr = 0.0005 for all groups
		>>> scheduler = LinearLR(optimizer, max_iters=10, last_iter=-1, min_lr=1e-8)
		>>> for iter in range(10):
		>>> 	train(...)
		>>>		scheduler.step()
		>>> validate(...)
	"""
	def __init__(self, optimizer, max_iters, last_iter=-1, min_lr=1e-8):
		self.optimizer = optimizer
		self.max_iters = max_iters
		self.num_iters = last_iter
		self.min_lr = min_lr
		self.done = False
		if last_iter == -1:
			for group in optimizer.param_groups:
				group.setdefault('initial_lr', group['lr'])
		else:
			for i, group in enumerate(optimizer.param_groups):
				if 'initial_lr' not in group:
					raise KeyError("param 'initial_lr' is not specified "
								   "in param_groups[{}] when resuming an optimizer".format(i))
		self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
		self.step(last_iter + 1)

	def get_lr(self):
		return [self.decay_func(base_lr) for base_lr in self.base_lrs]

	def decay_func(self, init_lr):
		new_lr = init_lr*((self.max_iters-self.num_iters)/self.max_iters)
		return max(new_lr, self.min_lr)

	def step(self, epoch=None):
		if epoch is None:
			epoch = self.num_iters + 1
		self.num_iters = epoch
		for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
			param_group['lr'] = lr
		return self.done
		