import os
import copy

import data_utils

class DataConfig(object):
    def __init__(self, parser, defaults={}):
        super(DataConfig,self).__init__()
        self.parser = parser
        self.defaults = defaults

    def apply(self, opt):
        print('configuring data')
        self.apply_defaults(opt)
        return make_loaders(opt)

    def set_defaults(self, **kwargs):
        for k, v in kwargs.items():
            self.defaults[k] = v

    def apply_defaults(self, opt):
        for k, v in self.defaults.items():
            k = k.replace('-', '_')
            if not hasattr(opt, k):
                setattr(opt, k, v)

def make_loaders(opt):
    """makes training/val/test"""
    batch_size = opt.batch_size * opt.world_size
    eval_batch_size = opt.eval_batch_size * opt.world_size
    seq_length = opt.seq_length
    if seq_length < 0:
        seq_length = seq_length * opt.world_size
    eval_seq_length = opt.eval_seq_length
    if opt.eval_seq_length < 0:
        eval_seq_length = eval_seq_length * opt.world_size
    data_loader_args = {'num_workers': 0, 'shuffle': opt.shuffle, 'batch_size': batch_size,
    #data_loader_args = {'num_workers': 1, 'shuffle': opt.shuffle, 'batch_size': batch_size,
                    'pin_memory': True, 'transpose': opt.transpose, 'distributed': opt.world_size > 1,
                    'rank': opt.rank, 'world_size': opt.world_size, 'drop_last': opt.world_size > 1}
    if opt.data_set_type == 'L2R':
        loader_type = data_utils.ShardLoader
        data_loader_args.update({'seq_len': seq_length, 'persist_state': opt.persist_state, 'samples_per_shard': opt.samples_per_shard})
    else:
        loader_type = data_utils.DataLoader
    split = get_split(opt)
    data_set_args = {
        'path': opt.data, 'seq_length': seq_length, 'lazy': opt.lazy, 'delim': opt.delim,
        'text_key': opt.text_key, 'label_key': opt.label_key, 'preprocess': opt.preprocess,
        'ds_type': opt.data_set_type, 'split': split, 'loose': opt.loose_json,
        'tokenizer_type': opt.tokenizer_type, 'tokenizer_model_path': opt.tokenizer_path,
        'vocab_size': opt.vocab_size, 'model_type': opt.tokenizer_model_type,
        'non_binary_cols': opt.non_binary_cols}

    eval_loader_args = copy.copy(data_loader_args)
    eval_set_args = copy.copy(data_set_args)
    eval_set_args['split']=[1.]
    # if optional eval args were set then replace their equivalent values in the arg dict
    if opt.eval_batch_size != 0:
        eval_loader_args['batch_size'] = eval_batch_size
    if opt.eval_seq_length != 0:
        eval_set_args['seq_length'] = eval_seq_length
        if opt.data_set_type == 'L2R':
            eval_loader_args['seq_len'] = eval_seq_length
    if opt.eval_text_key is not None:
        eval_set_args['text_key'] = opt.eval_text_key
    if opt.eval_label_key is not None:
        eval_set_args['label_key'] = opt.eval_label_key

    train = None
    valid = None
    test = None

    if opt.data is not None:
        train, tokenizer = data_utils.make_dataset(**data_set_args)
        if should_split(split):
            train, valid, test = train
    eval_set_args['tokenizer'] = tokenizer

    if opt.valid is not None:
        eval_set_args['path'] = opt.valid
        valid, _ = data_utils.make_dataset(**eval_set_args)
    if test is None and opt.test is not None:
        eval_set_args['path'] = opt.test
        test, _ = data_utils.make_dataset(**eval_set_args)


    if train is not None and opt.batch_size > 0:
        train = loader_type(train, **data_loader_args)
    if valid is not None:
        valid = loader_type(valid, **eval_loader_args)
    if test is not None:
        test = loader_type(test, **eval_loader_args)
    return (train, valid, test), tokenizer

def should_split(split):
    return max(split) != 1.

def get_split(opt):
    splits = []
    if opt.split.find(',') != -1: 
        splits = [float(s) for s in opt.split.split(',')]
    elif opt.split.find('/') != -1:
        splits = [float(s) for s in opt.split.split('/')]
    else:
        splits = [float(opt.split)]
    split_total = sum(splits)
    if split_total < 1.:
        splits.append(1-split_total)
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    if opt.valid is not None:
        splits[1] = 0.
    if opt.test is not None:
        splits[2] = 0.
    final_sum = sum(splits)
    return [s/final_sum for s in splits]

def configure_data(parser):
    """add cmdline flags for configuring datasets"""
    main_parser = parser
    parser = parser.add_argument_group('data options')
    parser.add_argument('--data', nargs='+', default=['./data/imdb/unsup.json'],
                        help="""Filename for training""")
    parser.add_argument('--valid', nargs='*', default=None,
                        help="""Filename for validation""")
    parser.add_argument('--test', nargs='*', default=None,
                        help="""Filename for testing""")
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Data Loader batch size')
    parser.add_argument('--eval-batch-size', type=int, default=0,
                        help='Data Loader batch size for evaluation datasets')
    parser.add_argument('--data-size', type=int, default=256,
                        help='number of tokens in data')
    parser.add_argument('--loose-json', action='store_true',
                        help='Use loose json (one json-formatted string per newline), instead of tight json (data file is one json string)')
    parser.add_argument('--preprocess', action='store_true',
                        help='force preprocessing of datasets')
    parser.add_argument('--delim', default=',',
                        help='delimiter used to parse csv testfiles')
    parser.add_argument('--non-binary-cols', nargs='*', default=None,
                        help='labels for columns to non-binary dataset [only works for csv datasets]')
    parser.add_argument('--split', default='1.',
                        help='comma-separated list of proportions for training, validation, and test split')
    parser.add_argument('--text-key', default='sentence',
                        help='key to use to extract text from json/csv')
    parser.add_argument('--label-key', default='label',
                        help='key to use to extract labels from json/csv')
    parser.add_argument('--eval-text-key', default=None,
                        help='key to use to extract text from json/csv evaluation datasets')
    parser.add_argument('--eval-label-key', default=None,
                        help='key to use to extract labels from json/csv evaluation datasets')
    # tokenizer arguments
    parser.add_argument('--tokenizer-type', type=str, default='CharacterLevelTokenizer', choices=['CharacterLevelTokenizer', 'SentencePieceTokenizer'],
                        help='what type of tokenizer to use')
    parser.add_argument('--tokenizer-model-type', type=str, default='bpe', choices=['bpe', 'char', 'unigram', 'word'],
                        help='Model type to use for sentencepiece tokenization')
    parser.add_argument('--vocab-size', type=int, default=256,
                        help='vocab size to use for non-character-level tokenization')
    parser.add_argument('--tokenizer-path', type=str, default='tokenizer.model',
                        help='path used to save/load sentencepiece tokenization models')
    defaults = {
                'world_size': 1,
                'rank': -1,
                'persist_state': 0,
                'lazy': False,
                'shuffle': False,
                'transpose': False,
                'data_set_type': 'supervised',
                'seq_length': 256,
                'eval_seq_length': 256,
                'samples_per_shard': 100
               }
    return DataConfig(main_parser, defaults=defaults), parser
