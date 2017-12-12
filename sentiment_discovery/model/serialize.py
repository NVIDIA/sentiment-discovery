import torch

def save(model, savepath, save_dict=None, keep_vars=True):
	"""save model weights and other metadata"""
	if save_dict is not None:
		#if save_dict is provided save that instead of model state_dict
		torch.save(state_dict_cpu_copy(save_dict), savepath)
	else:
		torch.save(state_dict_cpu_copy(model.state_dict(keep_vars=keep_vars)), savepath)

def state_dict_cpu_copy(chkpt):
	"""save cpu copy of model state, so it can be reloaded by any device"""
	#if chkpt has things other than state_dict get the state_dict
	if 'state_dict' in chkpt:
		state_dict = chkpt['state_dict']
	else:
		state_dict = chkpt
	for n, p in state_dict.items():
		state_dict[n] = p.cpu()
	return chkpt

def restore(model, load_path):
	"""restore saved model and handle dictionary format and application/removal of weight norm"""
	chkpt = torch.load(load_path)
	#check if checkpoints has only save_dict or more information
	if 'state_dict' in chkpt:
		state_dict = chkpt['state_dict']
	else:
		state_dict = chkpt
	model.load_state_dict(state_dict)
	return chkpt
