import argparse
import itertools
import sys
import subprocess
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Let's run some multihead experiments!")
    parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu to run on')
    parser.add_argument('--train', type=str, default='./data/semeval/val.csv',
                        help='using nvidia training dataset')
    parser.add_argument('--val', type=str, default='./data/semeval/val.csv',
                        help='using nvidia val dataset')
    parser.add_argument('--test', type=str, default='./data/semeval/val.csv')
    parser.add_argument('--process-fn', type=str, default='process_str', choices=['process_str', 'process_tweet'],
                        help='what preprocessing function to use to process text. One of [process_str, process_tweet].')
    parser.add_argument('--text-key', default='text', type=str)

    args = parser.parse_args()

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    plutchik_cols = ' '.join(['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust'])

    base_command = "python3 finetune_classifier.py --data {train} --valid {val} --test {test} --warmup-epochs 0.5 --epochs 20 " \
        + "--text-key {text_key} --optim Adam --all-metrics --automatic-thresholding --batch-size 16 --save-finetune --process-fn {proc} " \
        + "--aux-lm-loss --aux-lm-loss-weight 0.02 --classifier-hidden-layers 4096 2048 1024 8 --classifier-dropout 0.3 --non-binary-cols " + plutchik_cols + ' '

    transformer_options = "--lr 1e-5 --tokenizer-type SentencePieceTokenizer --tokenizer-path ama_32k_tokenizer.model --vocab-size 32000 --decoder-layers 12 "\
        +" --decoder-embed-dim 768 --decoder-ffn-embed-dim 3072 --decoder-learned-pos --model transformer --load transformer.pt --use-final-embed --max-seq-len 150 " \
        +"  --dropout 0.2 --attention-dropout 0.2 --relu-dropout 0.2 --model-version-name transformer_multihead" 

    mlstm_options = " --lr 1e-5 --load mlstm.pt --model-version-name mlstm_multihead"

    formatted_base_command = base_command.format(train=args.train, val=args.val, test=args.test, text_key=args.text_key, proc=args.process_fn)
    transformer_command = formatted_base_command + transformer_options
    print('*' * 100)
    print("EXPERIMENT: Transformer, {}, {}, {}".format('multihead', args.train, args.val))
    print('*' * 100)
    sys.stdout.flush()
    sys.stderr.flush()
    subprocess.call(transformer_command.split(), stdout=sys.stdout, stderr=sys.stderr, env=env)

    mlstm_command = formatted_base_command + mlstm_options
    print('*' * 100)
    print("EXPERIMENT: mLSTM, {}, {}, {}".format('multihead', args.train, args.val))
    print('*' * 100)
    sys.stdout.flush()
    sys.stderr.flush()
    subprocess.call(mlstm_command.split(), stdout=sys.stdout, stderr=sys.stderr, env=env)