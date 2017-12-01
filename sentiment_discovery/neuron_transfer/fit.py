import sys
import time
import collections

from torch.autograd import Variable
import numpy as np
from sklearn.linear_model import LogisticRegression

from sentiment_discovery.model import run_model


def transform(model, text, batch_size, persist_state=False):
	"""
	Uses model to process text and output a fixed dimension featurization

	Args:
		model: An instance of `modules.RecurrentModule`
		text: A list/1D-array-like of strings to be processed
		batch_size: (integer) Amount of strings to process at once
	Returns:
		hiddens: np.ndarray of featurized hidden outputs of size `[len(text),model.rnn_size]`
	"""
	model.train(False)

	s = 0
	n = len(text)*batch_size
	features = np.array([])
	first_feature = True

	tstart = time.time()

	# warm up hidden state to persist across DS
	if persist_state:
		for _s, hidden in enumerate(inference_loop(model, text, persist_state=persist_state)):
			continue

	for _s, neurons in enumerate(inference_loop(model, text, persist_state=persist_state)):
		if first_feature:
			features = []
			first_feature = False
		features.append(neurons)

	if not first_feature:
		features = (np.concatenate(features))
	print('%0.3f seconds to transform %d examples' %
				  (time.time() - tstart, n))
	sys.stdout.flush()
	return features

def inference_loop(model, data_iter, persist_state=False):
	"""generator that runs model inference on text data and extracts neuron features"""
	for out in run_model(model, data_iter, data_fn):
		yield model.get_neurons(out)
		if persist_state:
			model.persist_state(out[0])

def data_fn(data):
	"""returns data function for processing and returning a feed dictionary to be passed to model"""
	text_batch = Variable(data[0].long(), volatile=True)
	timesteps2use = Variable(data[-1], volatile=True)
	return {'text': text_batch, 'return_sequence': False, 'timesteps': timesteps2use-1}

def train_sklearn_logreg(trX, trY, vaX=None, vaY=None, teX=None, teY=None, penalty='l1',
		C=2**np.arange(-8, 1).astype(np.float), seed=42, model=None, max_iter=100, solver='saga', eval_test=True):
	"""
	slightly modified version of openai implementation https://github.com/openai/generating-reviews-discovering-sentiment/blob/master/utils.py
	if model is not None it doesn't train the model before scoring, it just scores the model
	"""
	# if only integer is provided for C make it iterable so we can loop over
	if not isinstance(C, collections.Iterable):
		C = list([C])
	scores = []
	if model is None:
		for i, c in enumerate(C):
			model = LogisticRegression(C=c, penalty=penalty, random_state=seed+i, max_iter=max_iter, solver=solver)
			model.fit(trX, trY)
			if vaX is not None:
				score = model.score(vaX, vaY)
			else:
				score = model.score(trX, trY)
			scores.append(score)
			del model
		c = C[np.argmax(scores)]
		model = LogisticRegression(C=c, penalty=penalty, random_state=seed+len(C), max_iter=max_iter, solver=solver)
		model.fit(trX, trY)
	else:
		c = model.C
	nnotzero = np.sum(model.coef_ != 0)
	train_score = model.score(trX, trY)*100
	if vaX is None:
		eval_data = trX
		val_score = train_score
	else:
		eval_data = vaX
		val_score = model.score(vaX, vaY)*100.
	if teX is not None and teY is not None:
		if not eval_test:
			score = (train_score, val_score, val_score, model.predict_proba(teX))
		else:
			eval_score = model.score(teX, teY)*100
			score = (train_score, val_score, eval_score, model.predict_proba(teX))
	else:
		score = (train_score, val_score, val_score, model.predict_proba(eval_data))
	return model, score, c, nnotzero
