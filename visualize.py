import os
import math
import collections

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style({'font.family': 'monospace'})

from cfg import cfg, configure_usage

def make_batch(data, bsz):
	ids = torch.ByteTensor(data.encode()).long()
	nbatch = ids.size(0) // bsz
	ids = ids.narrow(0, 0, nbatch * bsz)
	ids = ids.view(bsz, -1).contiguous()
	return ids

def run_step(x, s, embed, rnn, neuron_idx, last_layer=-1, overwrite=None):
	emb = embed(x)
	states, output = rnn(emb, s)
	if isinstance(states, tuple):
		hidden, cell = states
	else:
		hidden = cell = states
	feat = cell.data[last_layer,0,neuron_idx]
	if overwrite is not None and overwrite != 0:
		hidden.data[last_layer,0,neuron_idx] = overwrite
	return states, output, chr(x.data[0]), feat

def transform_text(text, states, embed, rnn, neuron_idx, last_layer=-1):
	text_out = []
	values = []
	for t in range(text.size(1)):
		x = text[:,t]
		states, output, c, feat = run_step(x, states, embed, rnn, neuron_idx, last_layer)
		text_out.append(c)
		values.append(feat)
	return text_out, values, states, output

def get_input(output, temperature=0.):
	if temperature == 0:
		topv, topi = output.data.topk(1)
		inp = Variable(topi[0], volatile=True)
	else:
		probs = F.softmax(output[0].squeeze().div(temperature))
		inp = Variable(torch.multinomial(probs, 1).data, volatile=True)
	if output.is_cuda:
		inp = inp.cuda()
	return inp

def generate_text(output, states, embed, rnn, neuron_idx, last_layer=-1,
					gen_length=32, temperature=0., overwrite=None):
	text_out = []
	values = []
	for t in range(gen_length):
		x = get_input(output, temperature)
		states, output, c, feat = run_step(x, states, embed, rnn, neuron_idx, last_layer, overwrite)
		text_out.append(c)
		values.append(feat)
	return text_out, values

def plot_neuron_heatmap(text, values, savename=None, negate=False, cell_height=.3, cell_width=.15):
	n_limit = 74
	num_chars = len(text)
	total_chars = math.ceil(num_chars/float(n_limit))*n_limit
	mask = np.array([0]*num_chars + [1]*(total_chars-num_chars))
	text = np.array(text+[' ']*(total_chars-num_chars))
	values = np.array(values+[0]*(total_chars-num_chars))
	if negate:
		values *= -1

	values = values.reshape(-1, n_limit)
	text = text.reshape(-1, n_limit)
	mask = mask.reshape(-1, n_limit)
	num_rows = len(values)
	plt.figure(figsize=(cell_width*n_limit, cell_height*num_rows))
	hmap=sns.heatmap(values, annot=text, mask=mask, fmt='', vmin=-1, vmax=1, cmap='RdYlGn',
					xticklabels=False, yticklabels=False, cbar=False)
	plt.tight_layout()
	if savename is not None:
		plt.savefig(savename)
	# clear plot for next graph since we returned `hmap`
	plt.clf()
	return hmap

def visualize(text, embed, rnn, init_state, seq_length, temperature=0, overwrite=0, neuron=2388,
				logger=None, layer=-1, generate=False, cuda=False, negate = False):
	batch_size = 1
	if isinstance(states, tuple):
		hidden, cell = states
	else:
		hidden = states
	last = hidden.size(0)-1
	if layer <= last and layer >= 0:
		last = layer

	text = make_batch(text, batch_size)
	batch = Variable(text)

	if cuda:
		batch = batch.cuda()

	t, v, s, o = transform_text(batch, states, embed, rnn, neuron, last)
	out_text = t
	out_values = v
	if generate:
		if negate:
			overwrite *= -1
		t_g, v_g = generate_text(o, s, embed, rnn, neuron, last, seq_length-len(t), temperature, overwrite)
		out_text += t_g
		out_values += v_g
		print(''.join(out_text).encode('utf-8'))

	save_str = ''
	if logger is not None:
		save_str = logger.get_log_dir('heatmaps')
	save_str = os.path.join(save_str, ''.join(out_text[:100])+'.png')
	plot_neuron_heatmap(out_text, out_values, save_str, negate)

if __name__ == '__main__':
	configure_usage('visualize.py')

	opt = cfg.opt
	embed = cfg.model.module.embedder
	rnn = cfg.model.module.rnn
	states = cfg.model.init_state
	logger = cfg.logger
	visualize(text=opt.text, embed=embed, rnn=rnn, init_state=states, seq_length=opt.seq_length,
				temperature=opt.temperature, overwrite=opt.overwrite, neuron=opt.neuron, logger=logger,
				cuda=opt.cuda, layer=opt.layer, generate=opt.generate, negate=opt.negate)
