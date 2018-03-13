The `transfer.py` script accepts the following arguments:

```
PyTorch Sentiment Discovery Transfer Learning

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         type of recurrent net (RNNTanh, RNNReLU, LSTM, mLSTM,
                        GRU
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --all_layers          if more than one layer is used, extract features from
                        all layers, not just the last layer
  --epochs EPOCHS       number of epochs to run Logistic Regression
  --seed SEED           random seed
  --load_model LOAD_MODEL
                        path to trained world language model
  --save_results SAVE_RESULTS
                        path to save intermediate and final results of
                        transfer
  --fp16                Run model in pseudo-fp16 mode (fp16 storage fp32
                        math).
  --neurons NEURONS     number of nenurons to extract as features
  --no_test_eval        do not evaluate the test set (useful when
                        your test set has no labels)
  --write_results WRITE_RESULTS
                        write results of model on test (or train if none is
                        specified) data to specified filepath [only supported
                        for csv datasets currently]
  --use_cached          reuse cached featurizations from a previous run

data options:
  --data DATA           Filename for training
  --valid VALID         Filename for validation
  --test TEST           Filename for testing
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
```

By default a sentiment classification model is saved to `<save_results>/classifier.pt` along with various other useful data such as the featurized datasets, regression accuracies, predicted regression probabilities, and the identities of the top features.

One can also save the results of test set evaluation to an easily interpretable csv/json (depending on the type of the original test set) by using the `--write_results` option.