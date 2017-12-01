import math

from torch.autograd import Variable

def clip_gradient_coeff(model, clip):
	"""Computes a gradient clipping coefficient based on gradient norm."""
	totalnorm = 0
	for p in model.parameters():
		if p.grad is None:
			continue
		modulenorm = p.grad.data.norm()
		totalnorm += modulenorm ** 2
	totalnorm = math.sqrt(totalnorm)
	return min(1, clip / (totalnorm + 1e-6))

def calc_grad_norm(model):
	"""Computes a gradient clipping coefficient based on gradient norm."""
	totalnorm = 0
	for p in model.parameters():
		if p.grad is None:
			continue
		modulenorm = p.grad.data.norm()
		totalnorm += modulenorm ** 2
	return math.sqrt(totalnorm)

def calc_grad_norms(model):
	"""Computes a gradient clipping coefficient based on gradient norm."""
	norms = []
	for p in model.parameters():
		if p.grad is None:
			continue
		modulenorm = p.grad.data.norm()
		norms += [modulenorm]
	return norms

def clip_gradient(model, clip):
	"""Clip the gradient."""
	if clip is None:
		return
	for p in model.parameters():
		if p.grad is None:
			continue
		p.grad.data = p.grad.data.clamp(-clip, clip)

class clip_gradients_helper:
	"""callable helper class to clip gradients in gradient hook"""
	def __init__(self, clip):
		self.clip = clip
	def __call__(self, x):
		return x.clamp(-self.clip, self.clip)

def clip_gradients(model, clip):
	"""clip gradients via module backward hooks"""
	if clip is None:
		return

	for p in model.parameters():
		p.register_hook(clip_gradients_helper(clip))

def make_cuda(state):
	"""make state tuple cuda type"""
	if isinstance(state, tuple):
		return (state[0].cuda(), state[1].cuda())

	return state.cuda()

def copy_state(state, make_cpu=False):
	"""returns copy of a state tuple"""
	if make_cpu:
		convert_op = lambda x: x.cpu()
	else:
		convert_op = lambda x: x

	if isinstance(state, tuple):
		return (Variable(convert_op(state[0].data)), Variable(convert_op(state[1].data)))

	return Variable(convert_op(state.data))
