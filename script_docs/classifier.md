The `classifier.py` script accepts the following arguments.

```
PyTorch Sentiment Discovery Classification

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         type of recurrent net (RNNTanh, RNNReLU, LSTM, mLSTM,
                        GRU
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --all_layers          if more than one layer is used, extract features from
                        all layers, not just the last layer
  --load_model LOAD_MODEL
                        path to save the classification
  --save_probs SAVE_PROBS
                        path to save numpy of predicted probabilities
  --fp16                Run model in pseudo-fp16 mode (fp16 storage fp32
                        math).
  --neurons NEURONS     number of nenurons to extract as features

data options:
  --data DATA           filename for training
  --batch_size BATCH_SIZE
                        Data Loader batch size
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
```