from torch.autograd import Variable

def run_model(model, data_iter, data_fn):
	"""runs a model over a given dataset"""
	for data in data_iter:
		data_dict = data_fn(data)
		#if model is on gpu, but not vanilla data parallel
		if model.using_cuda and not (model.data_parallel and not model.distributed):
			#move data to GPU
			for n, p in data_dict.items():
				if isinstance(p, Variable):
					data_dict[n] = p.cuda()
		yield model(**data_dict)

def epoch_loop(model, data_iter, data_fn, persist=False, skip_rule=None, inner_lr=None):
	"""loops over a data loader and performs model training (if needed)"""
	for hidden, output in run_model(model, data_iter, data_fn):
		skip = model.should_skip(skip_rule)

		done = False
		if not skip:
			#persist state for forward propagation of info
			if persist:
				model.persist_state(hidden)
			#perform gradient update/lr decay
			if model.is_training:
				model.optim_step()
				#step learning rate or skip if gradient was skipped
				if inner_lr is not None:
					done = inner_lr()
		else:
			model.reset_optim()

		yield hidden, output, skip, done
