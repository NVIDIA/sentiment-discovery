import argparse
import itertools
import sys
import subprocess
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Let's run some sst experiments!")
    parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu to run on')

    args = parser.parse_args()

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    base_command = "python3 finetune_classifier.py --warmup-epochs 0.5 --epochs 20 " \
        + "--optim Adam --all-metrics --threshold-metric f1 --automatic-thresholding --batch-size 16 " \
        + "--aux-lm-loss --aux-lm-loss-weight 0.02 --save-finetune " 

    transformer_options = "--lr 1e-5 --tokenizer-type SentencePieceTokenizer --tokenizer-path ama_32k_tokenizer.model --vocab-size 32000 --decoder-layers 12 "\
        +" --decoder-embed-dim 768 --decoder-ffn-embed-dim 3072 --decoder-learned-pos --model transformer --load transformer.pt --use-final-embed --max-seq-len 150 " \
        +"  --dropout 0.2 --attention-dropout 0.2 --relu-dropout 0.2 --model-version-name transformer_sst_binary" 

    mlstm_options = " --lr 1e-5 --load mlstm.pt --model-version-name mlstm_sst_binary"

    formatted_base_command = base_command
    transformer_command = formatted_base_command + transformer_options
    print('*' * 100)
    print("EXPERIMENT: Transformer, {}, ".format('sst',))
    print('*' * 100)
    sys.stdout.flush()
    sys.stderr.flush()
    subprocess.call(transformer_command.split(), stdout=sys.stdout, stderr=sys.stderr, env=env)

    mlstm_command = formatted_base_command + mlstm_options
    print('*' * 100)
    print("EXPERIMENT: mLSTM, {}, ".format('sst', ))
    print('*' * 100)
    sys.stdout.flush()
    sys.stderr.flush()
    subprocess.call(mlstm_command.split(), stdout=sys.stdout, stderr=sys.stderr, env=env)