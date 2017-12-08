import os, sys
import argparse
import time
import math
import copy
import pickle as pkl

import torch
from torch.autograd import Variable
from torch import optim
from torch.utils import data as td
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from sentiment_discovery import modules
import sentiment_discovery.neuron_transfer.utils as utils
from sentiment_discovery.neuron_transfer import transform, train_sklearn_logreg
from sentiment_discovery.model import run_model
from cfg import cfg, configure_usage

def format_task_prefix(model_name, dsname):
	"""
	strips -load_model flag of it's file ending, leaving the basepath so we know what
		reconstruction model training we transfered from
	"""
	basepath, filename = os.path.split(model_name)
	basepath = os.path.join(basepath, dsname)
	return os.path.join(basepath, os.path.splitext(filename)[0])

def arrange_labels(loader, Y):
	"""
	gets labels based on dataloaders sampler's indices to avoid possible wrap around from batch sampler
	"""
	inds = []
	for b in loader.sampler:
		inds.append(b)
	return np.array(list(Y))[inds]

def plot_logits(model_name, dsname, X, Y, model, logger, k=10, top_neurons=None):
	"""plot logits and save to appropriate experiment directory"""
	# dsname = os.path.basename(os.path.split(dsname)[0])
	# formatted_name = format_task_prefix(model_name, dsname)
	# format_wo_base_file = os.path.split(formatted_name)[0]
	if top_neurons is None:
		top_neurons = utils.get_top_k_neuron_weights(model.coef_.T, k)
	#make directories for logit pkl and logit graph
	vizpath = logger.get_log_dir('logit_vis')

	top_neurons=list(top_neurons)+[1285]
	#plot logits of each of the top neurons
	for i, n in enumerate(top_neurons):
		utils.plot_logit_and_save(trXt, trY, n, os.path.join(vizpath, str(i)+'_'+str(n)))
		# pkl.dump(trXt[:,n], open(os.path.join(pklpath, str(i)+'_'+str(n))+'.pkl', 'wb'))
		logger.log_pkl(trXt[:,n], 'logit_pkl', str(i)+'_'+str(n)+'.pkl', 'wb')

def get_accuracy_string(full_rep_accs):
	"""format string with accuracies"""
	return '%05.2f/%05.2f/%05.2f'%(full_rep_accs[0], full_rep_accs[1], full_rep_accs[2])

def get_and_save_accuracy_string(logger, model_name, dsname, accs, neuron_type='all_neurons'):
	"""save and return accuracy string to appropriate experiment directory"""
	acc_string = get_accuracy_string(full_rep_accs)
	logger.log_txt(acc_string, 'sentiment_accuracies', neuron_type)
	return acc_string

def print_and_save_accs(logger, model_name, dsname, accs, nnotzero, neuron_type='all_neurons', c=None):
	"""get and print/log accuracy string"""
	acc_string = get_and_save_accuracy_string(logger, model_name, dsname, accs, neuron_type)
	print('%s train/val/test accuracy w/ %s'%(acc_string, neuron_type))
	if c is not None:
		print('%05.2f regularization coef w/ %s'%(c, neuron_type))
	print('%05d features used w/ all neurons %s'%(nnotzero, neuron_type))

def configure_plotting(opt, trXt, trY, vaXt, vaY, teXt, teY):
	"""figure out whether to plot train or test"""
	if opt.test != 'None':
		Xplot = teXt
		Yplot = teY
		plotname = opt.test
	else:
		Xplot = trXt
		Yplot = trY
		plotname = opt.train
	return Xplot, Yplot, plotname

def get_csv_writer(opt, top_logits, top_neurons, all_proba, masked_proba,
					five_proba=None, five_logits=None,five_neurons=None):
	"""makes a generator to be used in data_utils.datasets.csv_dataset.write()"""
	use_five = False
	if opt.num_neurons < 5:
		use_five = True
	header = ['prob w/ all', 'prob w/ %s masked'%(str(opt.num_neurons),)]
	if use_five and five_proba is not None:
		header += ['prob w/ 5 masked']
	if use_five and five_logits is not None:
		assert five_neurons is not None
		assert len(five_neurons) == 5
		header += ['neuron %s'%(str(x),) for x in five_neurons]
	else:
		header += ['neuron %s'%(str(x),) for x in top_neurons]

	yield header

	for i, _ in enumerate(top_logits):
		row = []
		row.append(all_proba[i][1])
		row.append(masked_proba[i][1])
		if use_five and five_proba is not None:
			row.append(five_proba[i][1])
		if use_five and five_logits is not None:
			row.extend(list(five_logits[i].squeeze()))
		else:
			row.extend(list(top_logits[i].squeeze()))
		yield row


if __name__ == '__main__':
	configure_usage('sentiment_transfer.py')

	opt = cfg.opt
	model = cfg.model
	batch_size = cfg.batch_size
	trX = cfg.train
	trY = None
	vaX = cfg.valid
	vaY = None
	teX = cfg.test
	teY = None
	logger = cfg.logger

	# format output logging
	dsname = os.path.basename(os.path.split(opt.train)[0])
	formatted_name = format_task_prefix(opt.load_model, dsname)
	logger.set_sub_experiment(formatted_name)


	format_wo_base_file = os.path.split(formatted_name)[0]

	# featurize data with neurons
	trXt = None
	if trX is not None:
		trY = arrange_labels(trX, trX.dataset.Y)
		trXt = transform(model, trX, batch_size)
		trY = trY[:len(trXt)]
	vaXt = None
	if vaX is not None:
		vaY = arrange_labels(vaX, vaX.dataset.Y)
		vaXt = transform(model, vaX, batch_size)
		vaY = vaY[:len(vaXt)]
	teXt = None
	if teX is not None:
		teY = arrange_labels(teX, teX.dataset.Y)
		teXt = transform(model, teX, batch_size)
		teY = teY[:len(teXt)]

	del model

	Xplot, Yplot, plotname = configure_plotting(opt, trXt, trY, vaXt, vaY, teXt, teY)

	assert trXt is not None
	log_reg_model, full_rep_accs, c, nnotzero = train_sklearn_logreg(trXt, trY, vaXt, vaY, teXt, teY,
																		eval_test=not opt.no_test_eval)
	plot_logits(opt.load_model, plotname, Xplot, Yplot, log_reg_model, logger)
	print_and_save_accs(logger, opt.load_model, plotname, full_rep_accs, nnotzero, 'all_neurons', c)

	logger.log_pkl(full_rep_accs[-1], 'test_probs', 'all_neurons.pkl', 'wb')

	top_neurons = utils.get_top_k_neuron_weights(log_reg_model.coef_.T, k=opt.num_neurons)

	print_neurons = top_neurons
	if opt.num_neurons < 5:
		print_neurons = utils.get_top_k_neuron_weights(log_reg_model.coef_.T, k=5)

	print('neuron(s) %s are sentiment neurons'%(', '.join(list(map(str, print_neurons.tolist())))))
	sys.stdout.flush()

	masked_log_reg_model=utils.get_masked_model(log_reg_model, top_neurons)
	masked_log_reg_model, masked_full_rep_accs, c, nnotzero = train_sklearn_logreg(
																trXt, trY, vaXt, vaY, teXt, teY,
																model=masked_log_reg_model,
																eval_test=not opt.no_test_eval)
	print_and_save_accs(logger, opt.load_model, plotname, masked_full_rep_accs, nnotzero, 'masked_n_neurons')

	if opt.write_results != '':
		five_proba = None
		if opt.num_neurons < 5:
			masked_log_reg_model_5 = utils.get_masked_model(log_reg_model, print_neurons)
			masked_log_reg_model_5, masked_full_rep_accs_5, c_5, nnotzero_5 = train_sklearn_logreg(
																				trXt, trY, vaXt, vaY, teXt, teY,
																				model=masked_log_reg_model_5,
																				eval_test=not opt.no_test_eval)
		csv_writer = get_csv_writer(opt, teXt[:,top_neurons], top_neurons, full_rep_accs[-1],
									masked_full_rep_accs[-1], masked_full_rep_accs_5[-1],
									teXt[:,print_neurons], print_neurons)
		teX.dataset.write(csv_writer, path=opt.write_results)
	logger.log_pkl(masked_full_rep_accs[-1], 'test_probs', 'masked_n_neurons.pkl', 'wb')

	sys.stdout.flush()

	trXt_sentiment = utils.get_neuron_features(trXt, top_neurons)
	vaXt_sentiment = None
	if cfg.valid is not None:
		vaXt_sentiment = utils.get_neuron_features(vaXt, top_neurons)
	teXt_sentiment = None
	if cfg.test is not None:
		teXt_sentiment = utils.get_neuron_features(teXt, top_neurons)

	log_reg_model2, full_rep_accs, c, nnotzero = train_sklearn_logreg(
													trXt_sentiment, trY, vaXt_sentiment, vaY, 
													teXt_sentiment, teY, eval_test=not opt.no_test_eval)
	print_and_save_accs(logger, opt.load_model, plotname, full_rep_accs, nnotzero, 'n_neurons', c)

	logger.log_pkl(full_rep_accs[-1], 'test_probs', 'n_neurons.pkl', 'wb')

	sys.stdout.flush()

	pkl.dump(full_rep_accs[-1], open('extracted_proba.pkl', 'wb'))

	utils.plot_weight_contribs_and_save(log_reg_model.coef_, os.path.join(logger.get_log_dir('sentiment'), 'weight_vis.png'))
	logger.log_pkl((log_reg_model, top_neurons, log_reg_model2), 'sentiment', 'neurons.pkl', 'wb')
