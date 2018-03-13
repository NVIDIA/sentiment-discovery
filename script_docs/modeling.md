The `main.py` script used for training language models accepts the following arguments:

```
PyTorch Sentiment-Discovery Language Modeling

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         type of recurrent net (RNNTanh, RNNReLU, LSTM, mLSTM,
                        GRU)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --save_iters N        save current model progress interval
  --fp16                Run model in pseudo-fp16 mode (fp16 storage fp32
                        math).
  --dynamic_loss_scale  Dynamically look for loss scalar for fp16 convergance
                        help.
  --no_weight_norm      Add weight normalization to model.
  --loss_scale LOSS_SCALE
                        Static loss scaling, positive power of 2 values can
                        improve fp16 convergence.
  --world_size WORLD_SIZE
                        number of distributed workers
  --distributed_backend DISTRIBUTED_BACKEND
                        which backend to use for distributed training. One of
                        [gloo, nccl]
  --rank RANK           distributed worker rank. Typically set automatically
                        from multiproc.py
  --optim OPTIM         One of SGD or Adam

data options:
  --data DATA           filename for training
  --valid VALID         filename for validation
  --test TEST           filename for testing
  --batch_size BATCH_SIZE
                        Data Loader batch size
  --eval_batch_size EVAL_BATCH_SIZE
                        Data Loader batch size for evaluation datasets
  --data_size DATA_SIZE
                        number of tokens in data
  --loose_json          Use loose json (one json-formatted string per
                        newline), instead of tight json (data file is one json
                        string)
  --preprocess          force preprocessing of datasets
  --delim DELIM         delimiter used to parse csv testfiles
  --split SPLIT         comma-separated list of proportions for training,
                        validation, and test split
  --text_key TEXT_KEY   key to use to extract text from json/csv
  --label_key LABEL_KEY
                        key to use to extract labels from json/csv
  --eval_text_key EVAL_TEXT_KEY
                        key to use to extract text from json/csv evaluation
                        datasets
  --eval_label_key EVAL_LABEL_KEY
                        key to use to extract labels from json/csv evaluation
                        datasets

language modeling data options:
  --seq_length SEQ_LENGTH
                        Maximum sequence length to process (for unsupervised
                        rec)
  --eval_seq_length EVAL_SEQ_LENGTH
                        Maximum sequence length to process for evaluation
  --lazy                lazily load the data from disk (necessary for amazon)
  --persist_state PERSIST_STATE
                        0=reset state after every sample in a shard, 1=reset
                        state after every shard, -1=never reset state
  --num_shards NUM_SHARDS
                        number of total shards for unsupervised training
                        dataset. If a `split` is specified, appropriately
                        portions the number of shards amongst the splits.
  --val_shards VAL_SHARDS
                        number of shards for validation dataset if validation
                        set is specified and not split from training
  --test_shards TEST_SHARDS
                        number of shards for test dataset if test set is
                        specified and not split from training
```

The training dataset is arranged into `num_shards` different shards which are partitioned amongst train/val/test according to the specified split. 
Validation/test sets can also be supplied manually and will override the val/test datasets resulting from split. 
These manually supplied evaluation datasets have `num_shards` as their default number of shards, but this can be overridden with `val_shards`/`test_shards` respectively.

The dataset entries are transposed to allow for the concatenation of sequences in order to persist hidden state across subsequences of a shard in the training/eval process.

Hidden states are reset either at the start of every sequence, every shard, or never based on the value of persist_state. (See language modeling data options).

A negative sequence length calculates sequence length s.t. there are `-seq_length` subsequences that can be sampled in the dataset.
