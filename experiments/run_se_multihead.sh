#! /bin/bash
python experiments/run_clf_multihead.py --text-key title --train ../neel-data/csvs/noformat/semeval-train-noformat-emoji.csv  --val ../neel-data/csvs/noformat/semeval-val-noformat-emoji.csv --test ../neel-data/csvs/noformat/semeval-val-noformat-emoji.csv --process-fn process_tweet
