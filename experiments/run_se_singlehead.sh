#! /bin/bash
python experiments/run_clf_single_head.py --train ../neel-data/csvs/SemEval-7k-processed-IDs.train.csv --val ../neel-data/csvs/SemEval-7k-processed-IDs.val.csv --test ../neel-data/csvs/SemEval-7k-processed-IDs.test.csv
