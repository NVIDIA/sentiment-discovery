import os
import re
import html

import unidecode
import torch

HTML_CLEANER_REGEX = re.compile('<.*?>')

def clean_html(text):
	"""remove html div tags"""
	return re.sub(HTML_CLEANER_REGEX, ' ', text)

def process_str(text, front_pad='\n ', end_pad=' ', maxlen=None, clean_markup=True,
				clean_unicode=True, encode='utf-8'):
	"""
	Processes utf-8 encoded text according to the criterion specified in seciton 4 of https://arxiv.org/pdf/1704.01444.pdf (Radford et al).
	We use unidecode to clean unicode text into ascii readable text
	"""
	if clean_markup:
		text = clean_html(text)

	if clean_unicode:
		text = unidecode.unidecode(text)

	text = html.unescape(text)
	text = " ".join(text.split())
	if maxlen is not None:
		len2use = maxlen-len(front_pad)-len(end_pad)
		text = text[:len2use]
	text = front_pad+text+end_pad

	if encode is not None:
		text = text.encode(encoding=encode)

	return text

def tokenize_str_batch(strings, rtn_maxlen=True, process=True, maxlen=None):
	"""
	Tokenizes a list of strings into a ByteTensor
	Args:
		strings: List of utf-8 encoded strings to tokenize into ByteTensor form
		rtn_maxlen: Boolean with functionality specified in Returns.lens
	Returns:
		batch_tensor: ByteTensor of shape `[len(strings),maxlen_of_strings]`
		lens: Length of each string in strings after being preprocessed with `preprocess` (useful for
			dynamic length rnns). If `rtn_maxlen` is `True` then max(lens) is returned instead.
	"""
	if process:
		processed_strings = [process_str(x, maxlen=maxlen) for x in strings]
	else:
		processed_strings = [x.encode() for x in strings]
	lens = list(map(len, processed_strings))
	maxlen = max(lens)
	batch_tensor = torch.ByteTensor(len(lens), maxlen)
	for i, string in enumerate(processed_strings):
		_tokenize_str(string, batch_tensor[i])
	if not rtn_maxlen and rtn_maxlen is not None:
		return batch_tensor, lens
	if rtn_maxlen is None:
		return batch_tensor
	return batch_tensor, maxlen

def _tokenize_str(string, char_tensor=None):
	"""
	Parses a utf-8 encoded string and assigns to ByteTensor char_tensor.
	If no char_tensor is provide one is created.
	Typically used internally by `tokenize_str_batch`.
	"""
	if char_tensor is None:
		char_tensor = torch.ByteTensor(len(string.encode()))
	for i, char in enumerate(string):
		char_tensor[i] = char
	return char_tensor

def tokenize_text_file(path, preprocess=True):
	"""Tokenizes a text file to a signle axis ByteTensor."""
	assert os.path.exists(path)
	# Count bytes
	with open(path, 'r') as f:
		tokens = 0
		for line in f:
			tokens += len(line.encode())

	# Tokenize file content
	with open(path, 'r') as f:
		ids = torch.ByteTensor(tokens)
		token = 0
		for _line in f:
			if preprocess:
				line = process_str(_line)
			else:
				line = line.encode()
			for char in line:
				ids[token] = char
				token += 1

	return ids
