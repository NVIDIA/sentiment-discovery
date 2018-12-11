#! /bin/bash
python experiments/run_clf_multihead.py --text-key title --train data/semeval/train.csv  --val data/semeval/val.csv --test data/semeval/test.csv --process-fn process_tweet
