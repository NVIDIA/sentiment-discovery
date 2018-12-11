#! /bin/bash
python experiments/run_clf_single_head.py --train ../neel-data/csvs/14k-nvidia-processed-IDs.train.csv --val ../neel-data/csvs/14k-nvidia-processed-IDs.val.csv --test ../neel-data/csvs/14k-nvidia-processed-IDs.test.csv --process-fn process_tweet --text-key title
