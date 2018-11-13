from collections import namedtuple
import random
import os

import sentencepiece as spm

def make_tokenizer(tokenizer_type, corpus, model_path=None, vocab_size=None, model_type='bpe', pad_token=0, character_coverage=1.0):
    tokenizer_class = tokenizer_type
    if isinstance(tokenizer_class, str):
        tokenizer_class = eval(tokenizer_class)
    return tokenizer_class(corpus=corpus, vocab_size=vocab_size, model_path=model_path, model_type=model_type,
                            pad_token=pad_token, character_coverage=character_coverage)

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


COMMAND_TUPLE = namedtuple('CommandToken', ('name', 'token', 'Id'))

token_format = "<{0}>"

def prep_command_tokens(tokenlist):
    return [CommandToken(tok[0], token_format.format(tok[0]), tok[1]) for tok in tokenlist]

class CommandToken(object):
    def __init__(self, name, token, Id):
        self.name = name
        self.token = token
        self.Id = Id

    def __str__(self):
        return str(COMMAND_TUPLE(self.name, self.token, self.Id))


class Tokenizer(object):
    def __init__(self, command_tokens=None):
        self.command_tokens = command_tokens
        self.command_token_map = {}
        self.command_id_map = {}
        if command_tokens is not None:
            self.command_token_map = {tok.token: tok for tok in command_tokens}
            self.command_id_map = {tok.Id: tok for tok in command_tokens}
        self.num_command_tokens = len(self.command_tokens)
        if not hasattr(self, 'num_text_tokens'):
            self.num_text_tokens = 0
        if not hasattr(self, 'num_tokens'):
            self.num_tokens = self.num_command_tokens + self.num_text_tokens

    def __call__(self, text, process_fn=None):
        return self.EncodeAsIds(text, process_fn)

    @staticmethod
    def exists(model_path):
        raise NotImplementedError('Tokenizer exists method not implemented')

    def Train(self, corpus):
        raise NotImplementedError('Tokenizer Train not implemented')

    def EncodeAsIds(self, text, process_fn=None):
        raise NotImplementedError('Tokenizer EncodeAsIds not implemented')

    def EncodeAsTokens(self, text, process_fn=None):
        raise NotImplementedError('Tokenizer EncodeAsTokens not implemented')

    def IdToToken(self, Id):
        raise NotImplementedError('Tokenizer IdToToken not implemented')

    def TokenToId(self, token):
        raise NotImplementedError('Tokenizer TokenToId not implemented')

    def DecodeIds(self, Ids):
        raise NotImplementedError('Tokenizer DecodeIds not implemented')

    def DecodeTokens(self, Tokens):
        raise NotImplementedError('Tokenizer DecodeTokens not implemented')
        

class CharacterLevelTokenizer(Tokenizer):
    def __init__(self, pad_token=0, **kwargs):
        self.num_text_tokens = 256
        super(CharacterLevelTokenizer, self).__init__(prep_command_tokens([('pad', pad_token)]))

    @staticmethod
    def exists(model_path):
        return True

    def Train(self, corpus):
        pass

    def EncodeAsIds(self, text, process_fn=None):
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
            processed_text = str(processed_text)
        tokens = [self.TokenToId(c) for c in processed_text]
        return Tokenization(tokens, processed_text, text, self.command_tokens)

    def EncodeAsTokens(self, text, process_fn=None):
        processed_text = text
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


def write_corpus_as_lines(dataset, filepath):
    """
    Take Dataset or corpus, split it into lines, and write it to `filepath`.
    Return the total number of lines, and max length line.
    """
    total_sentence_count = 0
    maxlen = 0
    with open(filepath, 'w') as f:
        for entry in dataset:
            lines = entry.strip().split('\n')
            total_sentence_count += len(lines)
            for line in lines:
                maxlen = max(len(line), maxlen)
                f.write(line+'\n')
    return total_sentence_count, maxlen

class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_type='bpe', vocab_size=None, corpus=None, model_path=None, character_coverage=1.0, pad_token=0, **kwargs):
        self.character_coverage = character_coverage
        self.model_type = model_type.lower()
        self.spm_model = model_path
        self.num_text_tokens = vocab_size
        make_train = not SentencePieceTokenizer.exists(self.spm_model)
        if make_train:
            assert corpus is not None and self.num_text_tokens is not None
            self.Train(corpus, self.num_text_tokens)
        self.load_spm_model()
        super(SentencePieceTokenizer, self).__init__(prep_command_tokens([('pad', pad_token)]))

    @staticmethod
    def exists(model_path):
        if model_path is None:
            return False
        # check if path exists
        dne = not os.path.exists(model_path)
        # check if path.model exists
        if dne and not model_path.endswith('.model'):
            dne = not os.path.exists(model_path+'.model')
        return not dne

    def load_spm_model(self):
        if not os.path.exists(self.spm_model) and not self.spm_model.endswith('.model'):
            self.spm_model = self.spm_model+'.model'
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.spm_model)
        self.vocab_size = len(self.sp)

    def Train(self, corpus, num_text_tokens):
        self.num_text_tokens = num_text_tokens
        use_model_path = self.spm_model
        random_hash = str(random.randint(0, 2147483647))
        if use_model_path is None:
            use_model_path = random_hash
        if use_model_path.endswith('.model'):
            use_model_path = use_model_path[:use_model_path.rfind('.model')]
        input_path = use_model_path+'.txt.'+random_hash
        print('Writing temporary dataset for tokenization to '+input_path)
        line_count, maxlenline = write_corpus_as_lines(corpus, input_path)
        print('Training sentencepiece model')
        train_string = '--input={file_path} --model_prefix={model_prefix} --vocab_size={vocab_size}' \
            + ' --model_type={model_type} --input_sentence_size={input_sentence_size} --character_coverage={character_coverage} ' #\
            # + '--max_sentence_length={max_len}'
        train_string = train_string.format(file_path=input_path, model_prefix=use_model_path, vocab_size=num_text_tokens,
                            model_type=self.model_type, input_sentence_size=int(line_count), character_coverage=self.character_coverage)#,
                            # max_len=str(maxlenline))
        spm.SentencePieceTrainer.Train(train_string)
        os.remove(input_path)
        self.spm_model = use_model_path+'.model'
        print('Sentencepiece model written to '+self.spm_model)

    def EncodeAsIds(self, text, process_fn=None):
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.sp.EncodeAsIds(processed_text)
        return Tokenization(tokens, processed_text, text, self.command_tokens)

    def EncodeAsTokens(self, text, process_fn=None):
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.sp.EncodeAsTokens(processed_text)
        return Tokenization(tokens, processed_text, text, self.command_tokens, asIds=False)

    def IdToToken(self, Id):
        return self.sp.IdToToken(Id)

    def TokenToId(self, token):
        return self.sp.TokenToId(token)

    def DecodeIds(self, Ids):
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        return self.sp.DecodeIds(Ids)

    def DecodeTokens(self, Tokens):
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        return self.sp.DecodeTokens(Tokens)