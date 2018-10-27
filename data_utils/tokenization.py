from .preprocess import process_str

class Tokenization(object):
	def __init__(self, tokenization, text=None, original_text=None, command_tokens=None, asIds=True):
		self.tokenization = tokenization
		self.text = text
		if self.text is None:
			self.text = self.tokenization
		self.original_text = original_text
		if self.original_text is None:
			self.original_text = self.text
		self.command_tokens = command_tokens
		self.asIds = asIds
		self.parse_command_tokens()

	def parse_command_tokens(self):
		if self.command_tokens is None:
			return
		for command_token in self.command_tokens:
			if self.asIds:
				setattr(self, command_token.name, command_token.Id)
			else:
				setattr(self, command_token.name, command_token.token)

	def __getitem__(self, index):
		return self.tokenization[index]

	def __len__(self):
		return len(self.tokenization)

	def append(self, other):
		if isinstance(other, Tokenization):
			self.tokenization.extend(other.tokenization)
			self.text += other.text
			self.original_text += other.original_text
		else:
			self.tokenization.append(other)
		return self

	def extend(self, other):
		if isinstance(other, Tokenization):
			self.tokenization.extend(other.tokenization)
			self.text += other.text
			self.original_text += other.original_text
		else:
			self.tokenization.extend(other)
		return self

class CommandToken(object):
	def __init__(self, name, token, Id):
		self.name = name
		self.token = token
		self.Id = Id

def prep_command_tokens(tokenlist):
	return [CommandToken(tok[0], tok[1][0], tok[1][1]) for tok in tokenlist]

class Tokenizer(object):
	def __init__(self, command_tokens=None):
		self.command_tokens = command_tokens

	def __call__(self, text, process_fn=None):
		return self.EncodeAsIds(text, process_fn)

	def EncodeAsIds(self, text, process_fn=None):
		raise NotImplementedError('EncodeAsIds not implemented')

	def EncodeAsTokens(self, text, process_fn=None):
		raise NotImplementedError('EncodeAsTokens not implemented')

	def IdToToken(self, id):
		raise NotImplementedError('IdToToken not implemented')

	def TokenToId(self, token):
		raise NotImplementedError('TokenToId not implemented')

	def DecodeIds(self, Ids):
		raise NotImplementedError('DecodeIds not implemented')

	def DecodeTokens(self, Tokens):
		raise NotImplementedError('DecodeTokens not implemented')

class CharacterLevelTokenizer(Tokenizer):
	def __init__(self, process_fn=process_str, pad_token=('<PAD>', 0)):
		self.process_fn = process_fn
		super(CharacterLevelTokenizer, self).__init__(prep_command_tokens([('pad', pad_token)]))

	def EncodeAsIds(self, text, process_fn=None):
		processed_text = text
		if process_fn is None:
			process_fn = self.process_fn
		if process_fn is not None:
			processed_text = process_fn(processed_text)
			processed_text = str(processed_text)
		tokens = [self.TokenToId(c) for c in processed_text]
		return Tokenization(tokens, processed_text, text, self.command_tokens)

	def EncodeAsTokens(self, text, process_fn=None):
		processed_text = text
		if process_fn is None:
			process_fn = self.process_fn
		if process_fn is not None:
			processed_text = process_fn(processed_text)
		processed_text = str(processed_text)
		tokens = [c for c in processed_text]
		return Tokenization(tokens, processed_text, text, self.command_tokens, asIds=False)

	def IdToToken(self, Id):
		return chr(Id)

	def TokenToId(self, token):
		return ord(token)

	def DecodeIds(self, Ids):
		if isinstance(Ids, Tokenization):
			Ids = Ids.tokenization
		return ''.join([self.IdToToken(tok) for tok in Ids])

	def DecodeTokens(self, Tokens):
		if isinstance(Tokens, Tokenization):
			Tokens = Tokens.tokenization
		return ''.join(Tokens)