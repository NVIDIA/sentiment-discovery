import copy

import numpy as np
# configure matplotlib for use without xserver
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_neuron_features(features, neurons):
	"""
	Gets neuron activations from activations specified by `neurons`.

	Args:
		features: numpy arraylike of shape `[n,d]`
		neurons: numpy arraylike of shape `[k]` (where k is the number of neuron activations to select)
				used to index neuron activations from `features`. `1<=neurons[i]<=d` for all `i`
	Returns:
		numpy arraylike of shape `[n,k]`
	"""
	return np.reshape(features[:,neurons], [len(features), -1])

def mask_neuron_weights(weights, neurons, inplace=False):
	"""
	Zero masks rows of weights specified by neurons

	Args:
		weights: numpy array like of shape `[d,num_classes]`
		neurons: 1D numpy array of shape `[k]`. `1<=neurons[i]<d` for all `i`
		inplace: Boolean specifying whether to mask `weights` in place in addition to returning masked_vals
	Returns:
		masked_vals: zero masked `weights` with mask specified by `neurons`
	"""
	mask = np.zeros_like(weights)
	mask[neurons,np.arange(mask.shape[-1])] = 1
	masked_vals = weights*mask
	if inplace:
		weights[:] = masked_vals
	return masked_vals

def get_masked_model(log_reg_model, top_neurons):
	masked_log_reg_model = copy.copy(log_reg_model)
	masked_log_reg_model.coef_ = mask_neuron_weights(masked_log_reg_model.coef_.T, top_neurons).T
	return masked_log_reg_model

def get_top_k_neuron_weights(weights, k=1):
	"""
	Get's the indices of the top weights based on the l1 norm contributions of the weights
	based off of https://rakeshchada.github.io/Sentiment-Neuron.html interpretation of
	https://arxiv.org/pdf/1704.01444.pdf (Radford et. al)
	Args:
		weights: numpy arraylike of shape `[d,num_classes]`
		k: integer specifying how many rows of weights to select
	Returns:
		k_indices: numpy arraylike of shape `[k]` specifying indices of the top k rows
	"""
	weight_penalties = np.squeeze(np.linalg.norm(weights, ord=1, axis=1))
	if k == 1:
		k_indices = np.array([np.argmax(weight_penalties)])
	elif k >= np.log(len(weight_penalties)):
		# runs O(nlogn)
		k_indices = np.argsort(weight_penalties)[-k:][::-1]
	else:
		# runs O(n+klogk)
		k_indices = np.argpartition(weight_penalties, -k)[-k:]
		k_indices = (k_indices[np.argsort(weight_penalties[k_indices])])[::-1]
	return k_indices


def plot_logit_and_save(logits, labels, logit_index, name):
	"""
	Plots histogram (wrt to what label it is) of logit corresponding to logit_index.
	Saves plotted histogram to name.

	Args:
		logits:
		labels:
		logit_index:
		name:
	"""
	logit = logits[:,logit_index]
	plt.title('Distribution of Logit Values')
	plt.ylabel('# of logits per bin')
	plt.xlabel('Logit Value')
	plt.hist(logit[labels < .5], bins=25, alpha=0.5, label='neg')
	plt.hist(logit[labels >= .5], bins=25, alpha=0.5, label='pos')
	plt.legend()
	plt.savefig(name+'.png')
	plt.clf()

def plot_weight_contribs_and_save(coef, name):
	plt.title('Values of Resulting L1 Penalized Weights')
	plt.tick_params(axis='both', which='major')
	coef = normalize(coef)
	plt.plot(range(len(coef[0])), coef.T)
	plt.xlabel('Neuron (Feature) Index')
	plt.ylabel('Neuron (Feature) weight')
	plt.savefig(name)
	plt.clf()

def normalize(coef):
	norm = np.linalg.norm(coef)
	coef = coef/norm
	return coef
		