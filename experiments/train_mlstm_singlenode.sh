#! /bin/bash
python -m multiproc pretrain.py --data ./data/amazon/reviews.json --text-key reviewText --label-key overall --loose-json --lazy --dynamic-loss-scale --fp16 --distributed-backend nccl --train-iters 73000 --split 1000,1,1 --save mlstm.pt --lr 3e-3 --optim Adam --decay-style linear --constant-decay 100000 --batch-size 256
