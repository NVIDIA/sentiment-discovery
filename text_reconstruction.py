import os
import time
import math
import sys
import pickle as pkl
import subprocess

import torch
from torch.autograd import Variable
from torch import optim
from torch.utils import data as td
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from sentiment_discovery import modules
import sentiment_discovery.learning_rates as learning_rates
from sentiment_discovery.model import epoch_loop
from cfg import cfg, configure_usage

# @profile
def run_epoch(model, epoch, data2use, data_fn, num_batches, is_training=False,
				_cfg=None, inner_lr=None, saver=None):
	"""runs model over epoch of data, trains if necessary, and saves/returns model progress"""
	if is_training:
		print('entering training epoch %s'% (str(epoch),))
	else:
		print('evaluating epoch %s'% (str(epoch),))
	#handle config
	if _cfg is not None:
		config = _cfg
	else:
		config = cfg
	#set training mode
	model.train(is_training)
	loss_avg = 0
	loss_history = []
	avg_time = 0
	avg_ch_per_sec = 0
	_t = 0
	save_step = 0
	start_epoch_time = time.time()
	#flush standard out
	sys.stdout.flush()

	start = time.time()
	for s, (hidden, output, skip, done) in enumerate(
		epoch_loop(model, data2use, data_fn, persist=is_training, inner_lr=inner_lr, skip_rule='no')):

		if skip:
			print('skipping epoch %s batch %s with gradient norm %s and loss %s'%(
					epoch, s, model.gn, model.loss.data[0]))

		computation_time = time.time()-start
		ch_per_sec = (config.batch_size*config.seq_length)/computation_time

		#get loss and calculate 100 step moving average
		loss = model.loss.data.cpu()[0]
		loss_avg = .99*loss_avg + .01*loss
		loss_history.append(loss)

		#reset performance metric averages after gpu warmed up
		t = s-_t
		if s == 10 or s == 150:
			_t = s
			avg_time = computation_time
			avg_ch_per_sec = ch_per_sec
		else:
			avg_time = avg_time*t/(t+1)+computation_time/(t+1)
			avg_ch_per_sec = avg_ch_per_sec*t/(t+1)+ch_per_sec/(t+1)

		# print metrics and current status every 10 iters
		if s % 10 == 0 or s == num_batches-1:
			timeleft = (num_batches-1-s)*avg_time
			if is_training:
				gn = model.gn
				print('e%s %s / %s loss %.2E loss avg %.2E time %.2E time left %.1E ch/s avg %.2E grad_norm %.1E ' % (
							epoch, s, num_batches, loss, loss_avg, computation_time, timeleft, avg_ch_per_sec, gn))
			else:
				print('e*%s %s / %s loss %.2E loss avg %.2E time %.2E time left %.1E ch/s avg %.2E' % (
							epoch, s, num_batches, loss, loss_avg, computation_time, timeleft, avg_ch_per_sec))

		#flush stdout
		sys.stdout.flush()

		#save model
		if is_training and s%5000 == 0:
			print('saving iter %s of epoch %s'%(str(s), str(epoch)))
			if saver is not None:
				saver('.%s.pt'%(str(s)), loss_history)

		start = time.time()

	#set back to training mode just in case
	if not is_training:
		model.train(True)
	#flush stdout just in case
	sys.stdout.flush()
	return loss_avg/num_batches, loss_history

def make_data_fn(is_training, timesteps2use):
	"""returns data function for processing and returning a feed dictionary to be passed to model"""
	def data_fn(data):
		text_batch = Variable(data[0].long(), requires_grad=False, volatile=not is_training)
		mask = Variable(data[1], requires_grad=False, volatile=not is_training)
		return {'text': text_batch, 'state_mask': mask,
				'return_sequence': False, 'timesteps': timesteps2use}
	return data_fn

def main():
	"""runs script functionality"""
	configure_usage('text_reconstruction.py')

	opt = cfg.opt
	model = cfg.model
	text = cfg.train
	n_batch = cfg.n_batch
	valid = cfg.valid
	nv_batch = cfg.nv_batch
	test = cfg.test
	nt_batch = cfg.nt_batch
	TIMESTEPS = cfg.seq_length
	lr_scheduler = cfg.lr
	outer_loop = cfg.outer_loop
	saver = cfg.saver
	histories = []
	val_histories = []
	test_histories = []
	if opt.benchmark:
		torch.backends.cudnn.benchmark = True
	logger = cfg.logger

	train_fn = make_data_fn(True, TIMESTEPS)
	eval_fn = make_data_fn(False, TIMESTEPS)

	for e in outer_loop:
		loss_avg = 0.
		history = []
		try:
			if opt.train != 'None':
				_, history = run_epoch(model, e, text, train_fn, n_batch, is_training=not opt.should_test,
										inner_lr=cfg.inner_lr, saver=lambda ext, x: saver('e'+str(e)+ext, x))
				saver('e'+str(e)+'.pt',history)
			if opt.valid != 'None':
				_, val_history = run_epoch(model, e, valid, eval_fn, nv_batch, is_training=False)
				val_histories.append(val_history)
				logger.log_pkl(str(np.mean(val_histories)), 'val_history', 'e%s.pkl' % (e,), 'wb')
			if opt.test != 'None':
				_, test_history = run_epoch(model, e, test, eval_fn, nt_batch, is_training=False)
				test_histories.append(test_history)
				logger.log_pkl(str(np.mean(test_histories)), 'test_history', 'e%s.pkl' % (e,), 'wb')

			#save progress
			saver('e'+str(e), history)

			if opt.lr_scheduler == 'linear' or opt.no_loss:
				pass
			lr_scheduler.step()
			e += 1

		except Exception as ex:
			saver('e'+str(e)+'.pt', history)
			print('Exiting from training early')
			raise ex
			exit()

def should_run_single_process():
	"""determines whether to run in normal or distributed mode"""
	if '-distributed' not in sys.argv:
		return True
	else:
		rank = get_integer_arg(sys.argv, '-rank')
		if rank is not None and rank != -1:
			return True
	return False

def distributed_main():
	"""executes a distributed worker process"""
	num_workers = 1
	num_gpus = get_integer_arg(sys.argv, '-num_gpus')
	if num_gpus is not None and num_gpus > 0:
		num_workers = num_gpus
	for w in range(num_workers):
		run_distributed_process(w, num_workers)

def run_distributed_process(process_num, world_size):
	"""runs all worker processes for distributed execution"""
	arglist = list(sys.argv)
	set_integer_arg(arglist, '-rank', process_num)
	set_integer_arg(arglist, '-world_size', world_size)
	env = os.environ.copy()
	env['CUDA_VISIBLE_DEVICES'] = str(process_num)
	command = [get_python_command()] + arglist
	if process_num == world_size-1:
		subprocess.call(command, env=env)
	else:
		subprocess.Popen(command, env=env)

def get_python_command():
	"""which python command to run distributed processes with"""
	if sys.version_info.major == 3:
		return 'python3'
	return 'python'

def get_integer_arg(arglist, arg_string):
	"""tries to get the value of an integer arg from the cmdline args"""
	val = None
	try:
		arg_ind = arglist.index(arg_string)
	except:
		arg_ind = -1
	if arg_ind != -1:
		try:
			val = int(sys.argv[arg_ind+1])
		except:
			pass
	return val

def set_integer_arg(arglist, arg_string, value):
	"""sets the value of an integer arg in the cmdline string"""
	try:
		arg_ind = arglist.index(arg_string)
	except:
		arg_ind = -1
	if arg_ind != -1:
		try:
			_ = int(arglist[arg_ind + 1])
			arglist[arg_ind + 1] = str(value)
		except:
			arglist.insert(arg_ind + 1, str(value))
	else:
		arglist.append(arg_string)
		arglist.append(str(value))

if __name__ == '__main__':
	"""run normal or distributed process"""
	if should_run_single_process():
		main()
	else:
		distributed_main()
