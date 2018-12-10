# Arguments

This page of documentation has a list of all the arguments pertaining to certain scripts or pieces of code.
## General Arguments

    These are arguments which are commonly used amongst a number of scripts.

    ```
    group = parser.add_argument_group('general', 'general purpose arguments')
    group.add_argument('--model', type=str, default='mLSTM',
                        help='type of recurrent net (RNNTanh, RNNReLU, LSTM, mLSTM, GRU)')
    group.add_argument('--lr', type=float, default=5e-4,
                        help='initial learning rate')
    group.add_argument('--constant-decay', type=int, default=None,
                        help='number of iterations to decay LR over,' + \
                             ' None means decay to zero over training')
    group.add_argument('--clip', type=float, default=0,
                        help='gradient clipping')
    group.add_argument('--epochs', type=int, default=1,
                        help='upper epoch limit')
    group.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    group.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    group.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    group.add_argument('--save', type=str,  default='lang_model.pt',
                        help='path to save the final model')
    group.add_argument('--load', type=str, default=None,
                        help='path to a previously saved model checkpoint')
    group.add_argument('--load-optim', action='store_true',
                        help='load most recent optimizer to resume training')
    group.add_argument('--save-iters', type=int, default=10000, metavar='N',
                        help='save current model progress interval')
    group.add_argument('--save-optim', action='store_true',
                        help='save most recent optimizer')
    group.add_argument('--fp16', action='store_true',
                        help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    group.add_argument('--dynamic-loss-scale', action='store_true',
                        help='Dynamically look for loss scalar for fp16 convergance help.')
    group.add_argument('--no-weight-norm', action='store_true',
                        help='Add weight normalization to model.')
    group.add_argument('--loss-scale', type=float, default=1,
                        help='Static loss scaling, positive power of 2 values can improve fp16 convergence.')
    group.add_argument('--world-size', type=int, default=1,
                        help='number of distributed workers')
    group.add_argument('--distributed-backend', default='gloo',
                        help='which backend to use for distributed training. One of [gloo, nccl]')
    group.add_argument('--rank', type=int, default=-1,
                        help='distributed worker rank. Typically set automatically from multiproc.py')
    group.add_argument('--optim', default='Adam',
                        help='One of PyTorch\'s optimizers (Adam, SGD, etc). Default: Adam')
    group.add_argument('--chkpt-grad', action='store_true',
                        help='checkpoint gradients to allow for training with larger models and sequences')
    group.add_argument('--multinode-init', action='store_true',
                        help='initialize multinode. Environment variables should be set as according to https://pytorch.org/docs/stable/distributed.html')
    ```

## Data Arguments
    
    These arguments are used to create datasets, dataloaders, and other pieces of the data pipeline.

    ```
    group = group.add_argument_group('data options')
    group.add_argument('--data', nargs='+', default=['./data/imdb/unsup.json'],
                        help="""Filename for training""")
    group.add_argument('--valid', nargs='*', default=None,
                        help="""Filename for validation""")
    group.add_argument('--test', nargs='*', default=None,
                        help="""Filename for testing""")
    group.add_argument('--process-fn', type=str, default='process_str', choices=['process_str', 'process_tweet'],
                        help='what preprocessing function to use to process text. One of [process_str, process_tweet].')
    group.add_argument('--batch-size', type=int, default=128,
                        help='Data Loader batch size')
    group.add_argument('--eval-batch-size', type=int, default=0,
                        help='Data Loader batch size for evaluation datasets')
    group.add_argument('--data-size', type=int, default=256,
                        help='number of tokens in data')
    group.add_argument('--loose-json', action='store_true',
                        help='Use loose json (one json-formatted string per newline), instead of tight json (data file is one json string)')
    group.add_argument('--preprocess', action='store_true',
                        help='force preprocessing of datasets')
    group.add_argument('--delim', default=',',
                        help='delimiter used to parse csv testfiles')
    group.add_argument('--non-binary-cols', nargs='*', default=None,
                        help='labels for columns to non-binary dataset [only works for csv datasets]')
    group.add_argument('--split', default='1.',
                        help='comma-separated list of proportions for training, validation, and test split')
    group.add_argument('--text-key', default='sentence',
                        help='key to use to extract text from json/csv')
    group.add_argument('--label-key', default='label',
                        help='key to use to extract labels from json/csv')
    group.add_argument('--eval-text-key', default=None,
                        help='key to use to extract text from json/csv evaluation datasets')
    group.add_argument('--eval-label-key', default=None,
                        help='key to use to extract labels from json/csv evaluation datasets')
    # tokenizer arguments
    group.add_argument('--tokenizer-type', type=str, default='CharacterLevelTokenizer', choices=['CharacterLevelTokenizer', 'SentencePieceTokenizer'],
                        help='what type of tokenizer to use')
    group.add_argument('--tokenizer-model-type', type=str, default='bpe', choices=['bpe', 'char', 'unigram', 'word'],
                        help='Model type to use for sentencepiece tokenization')
    group.add_argument('--vocab-size', type=int, default=256,
                        help='vocab size to use for non-character-level tokenization')
    group.add_argument('--tokenizer-path', type=str, default='tokenizer.model',
                        help='path used to save/load sentencepiece tokenization models')
    # These are options that are relevant to data loading functionality, but are not meant to be exposed to the command line user.
    # These options are intneded to be set in code by specific scripts.
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
                'samples_per_shard': 1000
               }
    ```

## Model Arguments

    Arguments specific to model creation. Model creation also uses arguments from general arguments. 

    ### Recurrent Model Arguments
    ```
    group = parser.add_argument_group('recurrent', 'arguments for building recurrent nets')
    group.add_argument('--num-hidden-warmup', type=int, default=0,
                        help='number of times to conduct hidden state warmup passes through inputs to be used for transfer tasks')
    group.add_argument('--emsize', type=int, default=64,
                        help='size of word embeddings')
    group.add_argument('--nhid', type=int, default=4096,
                        help='number of hidden units per layer')
    group.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    group.add_argument('--dropout', type=float, default=0.0,
                        help='dropout applied to layers (0 = no dropout)')
    group.add_argument('--neural-alphabet', action='store_true',
                       help='whether to use the neural alphabet encoder structure')
    group.add_argument('--alphabet-size', type=int, default=128,
                       help='number of letters in neural alphabet')
    group.add_argument('--ncontext', type=int, default=2,
                       help='number of context characters used in neural alphabet encoder structure')
    group.add_argument('--residuals', action='store_true',
                        help='whether to implement residual connections between stackedRNN layers')
    ```

    ### Transformer Model Arguments
    ```
    group = parser.add_argument_group('transformer', 'args for specifically building a transformer network')
    group.add_argument('--dropout', type=float, default=0.1,
                        help='dropout probability -- transformer only')
    group.add_argument('--attention-dropout', type=float, default=0.0,
                        help='dropout probability for attention weights -- transformer only')
    group.add_argument('--relu-dropout', type=float, default=0.1,
                        help='dropout probability after ReLU in FFN -- transformer only')
    #ignore the encoder args for transformer. That's meant for seq2seq transformer
    group.add_argument('--encoder-embed-path', type=str, default=None,
                        help='path to pre-trained encoder embedding')
    group.add_argument('--encoder-embed-dim', type=int, default=64, # originally 512 but 64 for char level
                        help='encoder embedding dimension')
    group.add_argument('--encoder-ffn-embed-dim', type=int, default=256, # originally 2048 but scaled for char level
                        help='encoder embedding dimension for FFN')
    group.add_argument('--encoder-layers', type=int, default=6,
                        help='num encoder layers')
    group.add_argument('--encoder-attention-heads', type=int, default=8,
                        help='num encoder attention heads')
    group.add_argument('--encoder-normalize-before', default=False, action='store_true',
                        help='apply layernorm before each encoder block')
    group.add_argument('--encoder-learned-pos', default=False, action='store_true',
                        help='use learned positional embeddings in the encoder')
    group.add_argument('--decoder-embed-path', type=str, default=None,
                        help='path to pre-trained decoder embedding')
    group.add_argument('--decoder-embed-dim', type=int, default=64, # originally 512 but 64 for char level
                        help='decoder embedding dimension')
    group.add_argument('--decoder-ffn-embed-dim', type=int, default=256, # originally 2048 but scaled for char level
                        help='decoder embedding dimension for FFN')
    group.add_argument('--decoder-layers', type=int, default=6,
                        help='num decoder layers')
    group.add_argument('--decoder-attention-heads', type=int, default=8,
                        help='num decoder attention heads')
    group.add_argument('--decoder-learned-pos', default=False, action='store_true',
                        help='use learned positional embeddings in the decoder')
    group.add_argument('--decoder-normalize-before', default=False, action='store_true',
                        help='apply layernorm before each decoder block')
    group.add_argument('--share-decoder-input-output-embed', default=False, action='store_true',
                        help='share decoder input and output embeddings')
    group.add_argument('--share-all-embeddings', default=False, action='store_true',
                        help='share encoder, decoder and output embeddings'
                             ' (requires shared dictionary and embed dim)')
    group.add_argument('--use-final-embed', action='store_true',
                        help='whether to use the final timestep embeddings as output of transformer (in classification)')```

##  Unsupervised LM Arguments
    
    Arguments specific to running Left 2 Right (L2R) language modeling. This script also uses [data args](#data-arguments), [model args](#model-arguments), and [general args](#general-arguments).

    ```
    # Set unsupervised L2R language modeling data option defaults
    data_config.set_defaults(data_set_type='L2R', transpose=True)
    data_group.set_defaults(split='100,1,1')
    # Create unsupervised-L2R-specific options
    group = parser.add_argument_group('language modeling data options')
    group.add_argument('--seq-length', type=int, default=256,
                        help="Maximum sequence length to process (for unsupervised rec)")
    group.add_argument('--eval-seq-length', type=int, default=256,
                        help="Maximum sequence length to process for evaluation")
    group.add_argument('--lazy', action='store_true',
                        help='whether to lazy evaluate the data set')
    group.add_argument('--persist-state', type=int, default=1,
                        help='0=reset state after every sample in a shard, 1=reset state after every shard, -1=never reset state')
    group.add_argument('--train-iters', type=int, default=1000,
                        help="""number of iterations per epoch to run training for""")
    group.add_argument('--eval-iters', type=int, default=100,
                        help="""number of iterations per epoch to run validation/test for""")
    group.add_argument('--decay-style', type=str, default=None, choices=['constant', 'linear', 'cosine', 'exponential'],
                        help='one of constant(None), linear, cosine, or exponential')
    group.add_argument('--stlr-cut-frac', type=float, default=None,
                        help='what proportion of iterations to peak the slanted triangular learning rate')
    group.add_argument('--warmup', type=float, default=0,
                        help='percentage of data to warmup on (.03 = 3% of all training iters). Default 0')
    ```

## Classifier Model Arguments
    Arguments specific specific to building and training a text classification model with a language model encoder. This also uses [model args](#model-arguments).

    ```
    group = parser.add_argument_group('classifier', 'arguments used in training a classifier on top of a language model')
    group.add_argument('--max-seq-len', type=int, default=None,
                        help='maximum sequence length to use for classification. Transformer uses a lot of memory and needs shorter sequences.')
    group.add_argument('--classifier-hidden-layers', default=None, nargs='+',
                        help='sizes of hidden layers for binary classifier on top of language model, so excluding the input layer and final "1"')
    group.add_argument('--classifier-hidden-activation', type=str, default='PReLU',
                        help='[defaults to PReLU] activations used in hidden layers of MLP classifier (ReLU, Tanh, torch.nn module names)')
    group.add_argument('--classifier-dropout', type=float, default=0.1,
                        help='Dropout in layers of MLP classifier')
    group.add_argument('--all-layers', action='store_true',
                        help='if more than one layer is used, extract features from all layers, not just the last layer')
    group.add_argument('--concat-max', action='store_true',
                        help='whether to concatenate max pools onto cell/hidden states of RNNFeaturizer')
    group.add_argument('--concat-min', action='store_true',
                        help='whether to concatenate min pools onto cell/hidden states of RNNFeaturizer')
    group.add_argument('--concat-mean', action='store_true',
                        help='whether to concatenate mean pools onto cell/hidden states of RNNFeaturizer')
    group.add_argument('--get-hidden', action='store_true',
                        help='whether to use the hidden state (as opposed to cell state) as features for classifier')
    group.add_argument('--neurons', default=1, type=int,
                        help='number of nenurons to extract as features')
    group.add_argument('--heads-per-class', type=int, default=1,
                       help='set > 1 for multiple heads per class prediction (variance, regularlization)')
    parser.add_argument('--use-softmax', action='store_true', help='use softmax for classification')
    group.add_argument('--double-thresh', action='store_true',
                       help='whether to report all metrics at once')
    group.add_argument('--dual-thresh', action='store_true',
                        help='for 2 columns positive and negative, thresholds classes s.t. positive, negative, neutral labels are available')
    group.add_argument('--joint-binary-train', action='store_true',
                       help='Train with dual thresholded (positive/negative/neutral) classes and other normal binary classes.\
                             Arguments to non-binary-cols must be passed with positive negative classes first.\
                             Ex: `--non-binary-cols positive negative <other classes>`')```

## Sentiment Transfer Arguments
    These are arguments for performing transfer learning to text classification with logistic regression as in [Radford et .al](https://github.com/openai/generating-reviews-discovering-sentiment). This script leverages [data args](#data-arguments), [model args](#model-arguments), [general args](#general-arguments), and [classifier model args](#classifier-model-arguments).

    ```
    # Set transfer learning data option defaults
    data_group.set_defaults(split='1.', data=['data/binary_sst/train.csv'])
    data_group.set_defaults(valid=['data/binary_sst/val.csv'], test=['data/binary_sst/test.csv'])
    # Create transfer-learning-specific options
    group = parser.add_argument_group('sentiment_transfer', 'arguments used for sentiment_transfer script')
    group.add_argument('--mcc', action='store_true',
                        help='whether to use the matthews correlation coefficient as a measure of accuracy (for CoLA)')
    group.add_argument('--save-results', type=str,  default='sentiment',
                        help='path to save intermediate and final results of transfer')
    group.add_argument('--no-test-eval', action='store_true',
                        help='whether to not evaluate the test model (useful when your test set has no labels)')
    group.add_argument('--write-results', type=str, default='',
                        help='write results of model on test (or train if none is specified) data to specified filepath ')
    group.add_argument('--use-cached', action='store_true',
                        help='reuse cached featurizations from a previous run')
    group.add_argument('--drop-neurons', action='store_true',
                        help='drop top neurons instead of keeping them')```

## Running a Classifier Arguments
    These are arguments for running a classifier on a custom dataset. This script leverages [data args](#data-arguments), [model args](#model-arguments), [general args](#general-arguments), and [classifier model args](#classifier-model-arguments).

    ```
    # Set classification data option defaults
    data_group.set_defaults(split='1.', data=['data/binary_sst/train.csv'])
    data_group.set_defaults(shuffle=False)
    # Create classification-specific options
    group = parser.add_argument_group('run_classifier', 'arguments used for run classifier script')
    group.add_argument('--save_probs', type=str,  default='clf_results.npy',
                        help='path to save numpy of predicted probabilities')
    group.add_argument('--write-results', type=str, default='',
                        help='path to location for CSV -- write results of model on data \
                             input strings + results and variances. Will not write if empty')```

## Finetuning a Classifier Arguments
    These arguments are used to perform end to end finetuning of a classification model with a pretrained language model base. This script leverages [data args](#data-arguments), [model args](#model-arguments), [general args](#general-arguments), and [classifier model args](#classifier-model-arguments).

    ```
    # Set finetuning data option defaults
    data_group.set_defaults(split='1.', data=['data/binary_sst/train.csv'])
    data_group.set_defaults(valid=['data/binary_sst/val.csv'], test=['data/binary_sst/test.csv'])
    data_group.set_defaults(shuffle=True)
    # Create finetuning-specific options
    parser.set_defaults(get_hidden=True)
    data_group.add_argument('--seq-length', type=int, default=256,
                             help="Maximum sequence length to process (for unsupervised rec)")
    data_group.add_argument('--lazy', action='store_true',
                             help='whether to lazy evaluate the data set')
    group = parser.add_argument_group('finetune_classifier', 'arguments used for finetune script')
    group.add_argument('--use-logreg', action='store_true',
                        help='use scikitlearn logistic regression instead of finetuning whole classifier')
    group.add_argument('--stlr-cut-frac', type=float, default=None,
                        help='what proportion of iterations to peak the slanted triangular learning rate')
    group.add_argument('--cos-cut-frac', type=float, default=None,
                        help='what proportion of iterations to peak the cosine learning rate')
    group.add_argument('--lr-decay', type=float, default=1.0,
                        help='amount to multiply lr by to decay every epoch')
    group.add_argument('--momentum', type=float, default=0.0,
                        help='momentum for SGD')
    group.add_argument('--weight-decay', type=float, default=0,
                        help='weight decay for MLP optimization')
    group.add_argument('--freeze-lm', action='store_true',
                        help='keep lanuage model froze -- don\'t backprop to Transformer/RNN')
    group.add_argument('--aux-lm-loss', action='store_true',
                        help='whether to use language modeling objective as aux loss')
    group.add_argument('--aux-lm-loss-weight', type=float, default=1.0,
                        help='LM model weight -- NOTE: default is 1.0 for back compatible. Way too high -- reasonable around 0.02')
    group.add_argument('--aux-head-variance-loss-weight', type=float, default=0,
                        help='Set above 0.0 to force heads to learn different final-layer embeddings. Reasonable value ~10.-100.')
    group.add_argument('--use-class-multihead-average', action='store_true',
                        help='Use average output for multihead per class -- not necessary to use with --class-single-threshold [just average the thresholds]')
    group.add_argument('--thresh-test-preds', type=str, default=None,
                        help='path to thresholds for test outputs')
    group.add_argument('--report-metric', type=str, default='f1', choices=['jacc', 'acc', 'f1', 'mcc', 'precision', 'recall', 'var', 'all'],
                        help='what metric to report performance (save best model)')
    group.add_argument('--all-metrics', action='store_true',
                        help='Overloads report metrics and reports all metrics at once')
    group.add_argument('--threshold-metric', type=str, default='f1', choices=['jacc', 'acc', 'f1', 'mcc', 'precision', 'recall', 'var', 'all'],
                        help='which metric to use when choosing ideal thresholds?')
    group.add_argument('--micro', action='store_true',
                        help='whether to use micro averaging for metrics')
    group.add_argument('--global-tweaks', type=int, default=0,
                        help='HACK: Pass int (1000 for example) to tweak individual thresholds toward best global average [good for SemEval]. Will increase threshold on rare, hard to measure, categories.')
    group.add_argument('--save-finetune', action='store_true',
                        help='save finetuned models at every epoch of finetuning')
    group.add_argument('--model-version-name', type=str, default='classifier',
                        help='space to version model name -- for saving')
    group.add_argument('--automatic-thresholding', action='store_true',
                        help='automatically select classification thresholds based on validation performance. \
                            (test results are also reported using the thresholds)')
    group.add_argument('--no-test-eval', action='store_true',
                        help='Do not report test metrics, write test and val results to disk instead.')
    group.add_argument('--decay-style', type=str, default=None, choices=['constant', 'linear', 'cosine', 'exponential'],
                        help='Learning rate decay, one of constant(None), linear, cosine, or exponential')
    group.add_argument('--warmup-epochs', type=float, default=0.,
                        help='number of epochs to warm up learning rate over.')
    group.add_argument('--decay-epochs', type=float, default=-1, 
                        help='number of epochs to decay for. If -1 decays for all of training')
    group.add_argument('--load-finetuned', action='store_true',
                        help='load not just the language model but a previously finetuned full classifier checkpoint')```
