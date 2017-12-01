from torch.optim.lr_scheduler import ExponentialLR
from .linear import LinearLR

def make(lr_scheduler, start_epoch=0, lr_factor=None, max_iters=None, start_iter=0):
	if lr_scheduler == 'ExponentialLR':
		return lambda optimizer: ExponentialLR(optimizer, gamma=lr_factor, last_epoch=start_epoch-1)
	elif lr_scheduler == 'LinearLR':
		return lambda optimizer: LinearLR(optimizer, max_iters=max_iters, last_iter=start_iter-1)
	else:
		raise NotImplementedError('Please implement lr scheduler: %s'%(lr_scheduler,))
