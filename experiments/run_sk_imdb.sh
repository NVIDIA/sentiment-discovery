#! /bin/bash
python3 transfer.py --load mlstm.pt --save-results imdb --data ./data/imdb/train.json --test ./data/imdb/test.json --split .8 --batch-size 64 --fp16 
