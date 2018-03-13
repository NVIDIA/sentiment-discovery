The `generate.py` script accepts the following options:
```
PyTorch Sentiment Discovery Generation/Visualization

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         type of recurrent net (RNNTanh, RNNReLU, LSTM, mLSTM,
                        GRU
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --all_layers          if more than one layer is used, extract features from
                        all layers, not just the last layer
  --tied                tie the word embedding and softmax weights
  --load_model LOAD_MODEL
                        model checkpoint to use
  --save SAVE           output file for generated text
  --gen_length GEN_LENGTH
                        number of tokens to generate
  --seed SEED           random seed
  --temperature TEMPERATURE
                        temperature - higher will increase diversity
  --log-interval LOG_INTERVAL
                        reporting interval
  --fp16                run in fp16 mode
  --neuron NEURON       specifies which neuron to analyze for visualization or
                        overwriting. Defaults to maximally weighted neuron
                        of classifier [not working yet]
  --visualize           generates heatmap of main neuron activation [not
                        working yet]
  --overwrite OVERWRITE
                        Overwrite value of neuron s.t. generated text reads as
                        a +1/-1 classification [not working yet]
  --text TEXT           warm up generation with specified text first [not
                        working yet]
```

PNGs of generated heatmaps are saved to a similarly named path as the `--save` textfile.

Overwrite is also not limited to one. Since cell state is what's being overwritten, any value in [-\infty, \infty]