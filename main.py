import argparse
import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable


from fp16 import FP16_Module, FP16_Optimizer

import data
import model
from model import DistributedDataParallel as DDP

from apex.reparameterization import apply_weight_norm, remove_weight_norm
from configure_data import configure_data
from learning_rates import LinearLR

parser = argparse.ArgumentParser(description='PyTorch Sentiment-Discovery Language Modeling')
parser.add_argument('--model', type=str, default='mLSTM',
                    help='type of recurrent net (Tanh, ReLU, LSTM, mLSTM, GRU)')
parser.add_argument('--emsize', type=int, default=64,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=4096,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate')
parser.add_argument('--constant_decay', type=int, default=None,
                    help='number of iterations to decay LR over,' + \
                         ' None means decay to zero over training')
parser.add_argument('--clip', type=float, default=0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='lang_model.pt',
                    help='path to save the final model')
parser.add_argument('--load', type=str, default='',
                    help='path to a previously saved model checkpoint')
parser.add_argument('--load_optim', action='store_true',
                    help='load most recent optimizer to resume training')
parser.add_argument('--save_iters', type=int, default=2000, metavar='N',
                    help='save current model progress interval')
parser.add_argument('--save_optim', action='store_true',
                    help='save most recent optimizer')
parser.add_argument('--fp16', action='store_true',
                    help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--dynamic_loss_scale', action='store_true',
                    help='Dynamically look for loss scalar for fp16 convergance help.')
parser.add_argument('--no_weight_norm', action='store_true',
                    help='Add weight normalization to model.')
parser.add_argument('--loss_scale', type=float, default=1,
                    help='Static loss scaling, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--world_size', type=int, default=1,
                    help='number of distributed workers')
parser.add_argument('--distributed_backend', default='gloo',
                    help='which backend to use for distributed training. One of [gloo, nccl]')
parser.add_argument('--rank', type=int, default=-1,
                    help='distributed worker rank. Typically set automatically from multiproc.py')
parser.add_argument('--base-gpu', type=int, default=0,
                    help='base gpu to use as gpu 0')
parser.add_argument('--optim', default='Adam',
                    help='One of PyTorch\'s optimizers (Adam, SGD, etc). Default: Adam')
parser.add_argument('--tcp-port', type=int, default=6000,
                   help='tcp port so as to avoid address already in use errors')

# Add dataset args to argparser and set some defaults
data_config, data_parser = configure_data(parser)
data_config.set_defaults(data_set_type='L2R', transpose=True)
data_parser.set_defaults(split='100,1,1')
data_parser = parser.add_argument_group('language modeling data options')
data_parser.add_argument('--seq_length', type=int, default=256,
                         help="Maximum sequence length to process (for unsupervised rec)")
data_parser.add_argument('--eval_seq_length', type=int, default=256,
                         help="Maximum sequence length to process for evaluation")
data_parser.add_argument('--lazy', action='store_true',
                         help='whether to lazy evaluate the data set')
data_parser.add_argument('--persist_state', type=int, default=1,
                         help='0=reset state after every sample in a shard, 1=reset state after every shard, -1=never reset state')
data_parser.add_argument('--num_shards', type=int, default=102,
                         help="""number of total shards for unsupervised training dataset. If a `split` is specified,
                                 appropriately portions the number of shards amongst the splits.""")
data_parser.add_argument('--val_shards', type=int, default=0,
                         help="""number of shards for validation dataset if validation set is specified and not split from training""")
data_parser.add_argument('--test_shards', type=int, default=0,
                         help="""number of shards for test dataset if test set is specified and not split from training""")
data_parser.add_argument('--train-iters', type=int, default=1000,
                        help="""number of iterations per epoch to run training for""")
data_parser.add_argument('--eval-iters', type=int, default=100,
                        help="""number of iterations per epoch to run validation/test for""")

args = parser.parse_args()

torch.backends.cudnn.enabled = False
args.cuda = torch.cuda.is_available()

# initialize distributed process group and set device
if args.rank > 0 or args.base_gpu != 0:
    torch.cuda.set_device((args.rank+args.base_gpu) % torch.cuda.device_count())

if args.world_size > 1:
    distributed_init_file = os.path.splitext(args.save)[0]+'.distributed.dpt'
    torch.distributed.init_process_group(backend=args.distributed_backend, world_size=args.world_size,
                                         init_method='tcp://localhost:{}'.format(args.tcp_port), rank=args.rank)
#                                                    init_method='file://'+distributed_init_file, rank=args.rank)

# Set the random seed manually for reproducibility.
if args.seed > 0:
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

if args.loss_scale != 1 and args.dynamic_loss_scale:
    raise RuntimeError("Static loss scale and dynamic loss scale cannot be used together.")

###############################################################################
# Load data
###############################################################################

# Starting from sequential data, the unsupervised dataset type loads the corpus
# into rows. With the alphabet as the our corpus and batch size 4, we get
# ┌ a b c d e f ┐
# │ g h i j k l │
# │ m n o p q r │
# └ s t u v w x ┘.
# These rows are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
#
# The unsupervised dataset further splits the corpus into shards through which
# the hidden state is persisted. The dataset also produces a hidden state
# reset mask that resets the hidden state at the start of every shard. A valid
# mask might look like
# ┌ 1 0 0 0 0 0 ... 0 0 0 1 0 0 ... ┐
# │ 1 0 0 0 0 0 ... 0 1 0 0 0 0 ... │
# │ 1 0 0 0 0 0 ... 0 0 1 0 0 0 ... │
# └ 1 0 0 0 0 0 ... 1 0 0 0 0 0 ... ┘.
# With 1 indicating to reset hidden state at that particular minibatch index
(train_data, val_data, test_data), tokenizer = data_config.apply(args)

###############################################################################
# Build the model
###############################################################################
args.data_size = tokenizer.num_tokens
ntokens = args.data_size
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
print('* number of parameters: %d' % sum([p.nelement() for p in model.parameters()]))
if args.cuda:
    model.cuda()

rnn_model = model

optim = None
if args.load != '':
    sd = torch.load(args.load, map_location='cpu')
    if args.load_optim:
        optim_sd = torch.load(os.path.join(os.path.dirname(args.load), 'optim.pt'), map_location='cpu')
        rng = torch.load(os.path.join(os.path.dirname(args.load), 'rng.pt'))
        torch.cuda.set_rng_state(rng[0])
        torch.set_rng_state(rng[1])
    try:
        model.load_state_dict(sd)
    except:
        apply_weight_norm(model.rnn, hook_child=False)
        model.load_state_dict(sd)
        remove_weight_norm(model.rnn)

if not args.no_weight_norm:
    apply_weight_norm(model, 'rnn', hook_child=False)

# create optimizer and fp16 models
if args.fp16:
    model = FP16_Module(model)
    optim = eval('torch.optim.'+args.optim)(model.parameters(), lr=args.lr)
    optim = FP16_Optimizer(optim,
                           static_loss_scale=args.loss_scale,	
                           dynamic_loss_scale=args.dynamic_loss_scale)
else:
    optim = eval('torch.optim.'+args.optim)(model.parameters(), lr=args.lr)

if args.load_optim:
    pass
    optim.load_state_dict(optim_sd)

# add linear learning rate scheduler
if train_data is not None:
    if args.constant_decay:
        num_iters = args.constant_decay
    else:
        num_iters = args.train_iters * args.epochs

    init_step = -1
    if args.load_optim:
        init_step = optim_sd['iter']-optim_sd['skipped_iter']
        train_data.batch_sampler.start_iter = (optim_sd['iter'] % len(train_data)) + 1

    LR = LinearLR(optim, num_iters, last_iter=init_step)

# wrap model for distributed training
if args.world_size > 1:
    model = DDP(model)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

# get_batch subdivides the source data into chunks of length args.seq_length.
# If source is equal to the example output of the data loading example, with
# a seq_length limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the data loader. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM. A Variable representing an appropriate
# shard reset mask of the same dimensions is also returned.

def get_batch(data):
    reset_mask_batch = data[1].long()
    data = data[0].long()
    if args.cuda:
        data = data.cuda()
        reset_mask_batch = reset_mask_batch.cuda()
    text_batch = Variable(data[:, :-1].t().contiguous(), requires_grad=False)
    target_batch = Variable(data[:, 1:].t().contiguous(), requires_grad=False)
    reset_mask_batch = Variable(reset_mask_batch[:,:text_batch.size(0)].t().contiguous(), requires_grad=False)
    return text_batch, target_batch, reset_mask_batch

def init_hidden(batch_size):
    return rnn_model.rnn.init_hidden(args.batch_size)

def evaluate(data_source, max_iters):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    init_hidden(args.batch_size)
    total_loss = 0
    ntokens = args.data_size
    with torch.no_grad():
        data_iter = iter(data_source)
        i = 0
        while i < max_iters:
            batch = next(data_iter)
            data, targets, reset_mask = get_batch(batch)
            output, hidden = model(data, reset_mask=reset_mask)
            output_flat = output.view(-1, ntokens).contiguous().float()
            loss = criterion(output_flat, targets.view(-1).contiguous())
            if isinstance(model, DDP):
                torch.distributed.all_reduce(loss.data)
                loss.data /= args.world_size
            total_loss += loss.data[0]
            i += 1
    return total_loss / max(max_iters, 1)

def train(max_iters, total_iters=0, skipped_iters=0, elapsed_time=False):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    t0 = start_time
    ntokens = args.data_size
    hidden = init_hidden(args.batch_size)
    curr_loss = 0.
    distributed = isinstance(model, DDP)
    def log(epoch, i, lr, ms_iter, total_time, loss, scale):
        print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.2E} | ms/batch {:.3E} | total time {:.3E}\
                  loss {:.2E} | ppl {:8.2f} | loss scale {:8.2f}'.format(
                      epoch, i, max_iters, lr,
                      ms_iter, total_time, loss, math.exp(min(loss, 20)), scale
                  )
        )
    data_iter = iter(train_data)
    i = 0
    while i < max_iters:
        batch = next(data_iter)
        data, targets, reset_mask = get_batch(batch)
        optim.zero_grad()
        output, hidden = model(data, reset_mask=reset_mask)
        loss = criterion(output.view(-1, ntokens).contiguous().float(), targets.view(-1).contiguous())
        total_loss += loss.data.float()

        if args.fp16:
            optim.backward(loss, update_master_grads=False)
        else:
            loss.backward()

        if distributed:
            torch.distributed.all_reduce(loss.data)
            loss.data /= args.world_size
            model.allreduce_params()

        # clipping gradients helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip > 0:
            if not args.fp16:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            else:
                optim.clip_master_grads(clip=args.clip)

        if args.fp16:
            optim.update_master_grads()

        optim.step()

        # step learning rate and log training progress
        lr = optim.param_groups[0]['lr']
        if not args.fp16:
            LR.step()
        else:
            # if fp16 optimizer skips gradient step due to explosion do not step lr
            if not optim.overflow:
                LR.step()
            else:
                skipped_iters += 1

        # log current results
        if ((i+1) % args.log_interval == 0) and (i != max_iters - 1):
            cur_loss = total_loss[0] / args.log_interval
            cur_time = time.time()
            elapsed = cur_time - start_time
            total_elapsed = cur_time - t0 + elapsed_time
            log(epoch, i+1, lr, elapsed * 1000 / args.log_interval, total_elapsed, 
                cur_loss, args.loss_scale if not args.fp16 else optim.loss_scale)
            total_loss = 0
            start_time = cur_time
            sys.stdout.flush()

        # save current model progress. If distributed only save from worker 0
        if args.save_iters and total_iters % args.save_iters == 0 and total_iters > 0 and args.rank < 1:
            if args.rank < 1:
                with open(os.path.join(os.path.splitext(args.save)[0], 'e%s.pt'%(str(total_iters),)), 'wb') as f:
                    torch.save(model.state_dict(), f)
                if args.save_optim:
                    with open(os.path.join(os.path.splitext(args.save)[0], 'optim.pt'), 'wb') as f:
                        optim_sd = optim.state_dict()
                        optim_sd['iter'] = total_iters
                        optim_sd['skipped_iter'] = skipped_iters
                        torch.save(optim_sd, f)
                        del optim_sd

                    with open(os.path.join(os.path.splitext(args.save)[0], 'rng.pt'), 'wb') as f:
                        torch.save((torch.cuda.get_rng_state(), torch.get_rng_state()),f)
            if args.cuda:
                torch.cuda.synchronize()
        total_iters += 1
        i += 1
    #final logging
    elapsed_iters = max_iters % args.log_interval
    if elapsed_iters == 0:
        elapsed_iters = args.log_interval
    cur_loss = total_loss[0] / elapsed_iters
    cur_time = time.time()
    elapsed = cur_time - start_time
    total_elapsed = cur_time - t0 + elapsed_time
    log(epoch, max_iters, lr, elapsed * 1000/ elapsed_iters, total_elapsed,
        cur_loss, args.loss_scale if not args.fp16 else optim.loss_scale)

    return cur_loss, skipped_iters

# Loop over epochs.
lr = args.lr
best_val_loss = None

# If saving process intermittently create directory for saving
if args.save_iters > 0 and not os.path.exists(os.path.splitext(args.save)[0]) and args.rank < 1:
    os.makedirs(os.path.splitext(args.save)[0])

# At any point you can hit Ctrl + C to break out of training early.
try:
    total_iters = 0
    elapsed_time = 0
    skipped_iters = 0
    if args.load_optim:
        total_iters = optim_sd['iter']
        skipped_iters = optim_sd['skipped_iter']
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        val_loss, skipped_iters = train(args.train_iters, total_iters, skipped_iters, elapsed_time)
        elapsed_time += time.time() - epoch_start_time
        total_iters += args.train_iters
        if val_data is not None:
            print('entering eval')
            val_loss = evaluate(val_data, args.eval_iters)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(min(val_loss, 20))))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss and args.rank <= 0:
            torch.save(model.state_dict(), args.save)
            best_val_loss = val_loss

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
if os.path.exists(args.save):
    model.load_state_dict(torch.load(args.save, 'cpu'))

if not args.no_weight_norm and args.rank <= 0:
    remove_weight_norm(rnn_model)
    with open(args.save, 'wb') as f:
        torch.save(model.state_dict(), f)

if test_data is not None:
    # Run on test data.
    test_loss = evaluate(test_data, args.eval_iters)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
