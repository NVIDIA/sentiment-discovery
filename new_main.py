# transformer_main.py
import argparse
import os
import sys
import time
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from fp16 import FP16_Module, FP16_Optimizer

import data
import model as m
from model import DistributedDataParallel as DDP

from apex.reparameterization import apply_weight_norm, remove_weight_norm
from configure_data import configure_data
from learning_rates import LinearLR, CosineAnnealingLR, WarmupLR, SlantedTriangularLR
from arguments import add_general_args, add_model_args, add_unsupervised_data_args

rnn_model = None

def setup_model_and_optim(args, train_data, tokenizer):
    ntokens = args.data_size
    if args.model.lower() == 'transformer':
        embed_tokens = m.Embedding(ntokens, args.decoder_embed_dim, padding_idx=tokenizer.command_name_map['pad'].Id)
        model = m.TransformerModel(m.DecoderPreprocessor(args, embed_tokens),
                                    m.TransformerDecoder(args, embed_tokens))
    elif args.model.lower() == 'bert':
        model = m.BERTModel(args)
    else:
        model = m.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
        global rnn_model
        rnn_model = model
    LR_Warmer = None
    print('* number of parameters: %d' % sum([p.nelement() for p in model.parameters()]))
    if args.cuda:
        model.cuda()

    optim = None
    if args.load != '':
        sd = torch.load(args.load, map_location='cpu')
        if args.load_optim:
            #optim_sd = torch.load(os.path.join(os.path.dirname(args.load), 'optim.pt'), map_location='cpu')
            rng = torch.load(os.path.join(os.path.dirname(args.load), 'rng.pt'))
            torch.cuda.set_rng_state(rng[0])
            torch.set_rng_state(rng[1])
        try:
            model.load_state_dict(sd)
        except:
            apply_weight_norm(model, hook_child=False)
            model.load_state_dict(sd)
            remove_weight_norm(model)

    if not args.no_weight_norm:
        apply_weight_norm(model, hook_child=False)

    if optim is None:
        optim_choice = 'Adam' if args.stlr_cut_frac else args.optim
        if args.fp16:
            model = FP16_Module(model)
            optim = eval('torch.optim.'+args.optim)(model.parameters(), lr=args.lr)
            optim = FP16_Optimizer(optim,
                               static_loss_scale=args.loss_scale,
                               dynamic_loss_scale=args.dynamic_loss_scale)
        else:
            optim = eval('torch.optim.'+args.optim)(model.parameters(), lr=args.lr)

    if args.load_optim:
        optim.load_state_dict(optim_sd)

    # add linear learning rate scheduler
    if train_data is not None:
        if args.constant_decay:
            num_iters = args.constant_decay
        else:
            num_iters = args.train_iters * args.epochs

        init_step = -1
        if args.load_optim:
            #TODO: this no longer makes sense given the new data loaders
            init_step = optim_sd['iter']-optim_sd['skipped_iter']
            train_data.batch_sampler.start_iter = (optim_sd['iter'] % len(train_data)) + 1

        if args.stlr_cut_frac is None:
            LR = CosineAnnealingLR(optim, start_lr=args.lr, warmup_iter=200, num_iters=num_iters)
        else:
            LR = SlantedTriangularLR(optim, cut_frac=args.stlr_cut_frac, num_iters=num_iters)

        if args.warmup != 0:
            warmup_iter = args.warmup * num_iters
            LR_Warmer = WarmupLR(optim, warmup_iter, last_iter=init_step)

    # wrap model for distributed training
    if args.world_size > 1:
        model = DDP(model)

    criterion = nn.CrossEntropyLoss(reduce=False)
    return model, optim, LR, LR_Warmer, criterion

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

def get_batch(data, args):
    reset_mask_batch = data[1].long()
    padding_mask_batch = data[2].float()
    data = data[0].long()
    if args.cuda:
        data = data.cuda()
        reset_mask_batch = reset_mask_batch.cuda()
        padding_mask_batch = padding_mask_batch.cuda()
    text_batch = Variable(data[:,:-1].t().contiguous(), requires_grad=False)
    target_batch = Variable(data[:,1:].t().contiguous(), requires_grad=False)
    reset_mask_batch = Variable(reset_mask_batch[:,:text_batch.size(0)].t().contiguous(), requires_grad=False)
    padding_mask_batch = Variable(padding_mask_batch[:,:text_batch.size(0)].t().contiguous(), requires_grad=False)
    return text_batch, target_batch, reset_mask_batch, padding_mask_batch

def init_hidden(args):
    if rnn_model is not None:
        rnn_model.rnn.init_hidden(args.batch_size)

def evaluate(data_source, model, criterion, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    init_hidden(args)
    total_loss = 0
    ntokens = args.data_size
    max_iters = args.eval_iters
    with torch.no_grad():
        data_iter = iter(data_source)
        i = 0
        while i < max_iters:
            batch = next(data_iter)
            data, targets, reset_mask, padding_mask = get_batch(batch, args)

            output, hidden = model(data, reset_mask=reset_mask)
            losses = criterion(output.view(-1, ntokens).contiguous().float(), targets.view(-1).contiguous())
            padding_mask = padding_mask.view(-1)
            portion_unpadded = padding_mask.sum() / padding_mask.size(0)
            loss = portion_unpadded * torch.mean(losses * (padding_mask.view(-1).float()))
            if isinstance(model, DDP):
                torch.distributed.all_reduce(loss.data)
                loss.data /= args.world_size
            total_loss += loss.data.float()
            i+=1
    return total_loss / max_iters

def train(epoch, model, optim, train_data, LR, LR_Warmer, criterion, args, total_iters=0, skipped_iters=0, elapsed_time=False):
    # Turn on training mode which enables dropout.
    model.train()
    init_hidden(args)
    total_loss = 0
    start_time = time.time()
    t0 = start_time
    ntokens = args.data_size
    curr_loss = 0.
    distributed = isinstance(model, DDP)
    max_iters = args.train_iters
    def log(epoch, i, lr, ms_iter, total_time, loss, scale):
        print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.2E} | ms/batch {:.3E} | total time {:.3E}\
                  loss {:.2E} | ppl {:8.2f} | loss scale {:8.2f}'.format(
                      epoch, i, max_iters, lr,
                      ms_iter, total_time, loss, math.exp(min(loss, 20)), scale
                  )
        )
    i = 0
    data_iter = iter(train_data)
    while i < max_iters:
        batch = next(data_iter)
        data, targets, reset_mask, padding_mask = get_batch(batch, args)
        optim.zero_grad()
        output, _ = model(data, reset_mask=reset_mask)

        losses = criterion(output.view(-1, ntokens).contiguous().float(), targets.view(-1).contiguous())
        padding_mask = padding_mask.view(-1)
        portion_unpadded = padding_mask.sum() / padding_mask.size(0)
        loss = portion_unpadded * torch.mean(losses * (padding_mask.view(-1).float()))
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
            if args.warmup != 0:
                LR_Warmer.step()
        else:
            # if fp16 optimizer skips gradient step due to explosion do not step lr
            if not optim.overflow:
                LR.step()
                if args.warmup != 0:
                    LR_Warmer.step()
            else:
                skipped_iters += 1

        if ((i+1) % args.log_interval == 0):
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
        if args.save_iters and total_iters % (args.save_iters) == 0 and total_iters > 0 and args.rank < 1:
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
        i+=1
    #final logging
    elapsed_iters = max_iters % args.log_interval
    if elapsed_iters == 0:
        return cur_loss, skipped_iters

    cur_time = time.time()
    elapsed = cur_time - start_time
    total_elapsed = cur_time - t0 + elapsed_time
    log(epoch, max_iters, lr, elapsed * 1000/ elapsed_iters, total_elapsed,
        cur_loss, args.loss_scale if not args.fp16 else optim.loss_scale)

    return cur_loss, skipped_iters


def main():
    parser = argparse.ArgumentParser(description='PyTorch Sentiment-Discovery Language Modeling')
    parser = add_general_args(parser)
    parser = add_model_args(parser)
    data_config, data_parser = add_unsupervised_data_args(parser)
    args = parser.parse_args()

    torch.backends.cudnn.enabled = False
    args.cuda = torch.cuda.is_available()

    # initialize distributed process group and set device
    if args.rank > 0:
        torch.cuda.set_device(args.rank % torch.cuda.device_count())

    if args.world_size > 1:
        distributed_init_file = os.path.splitext(args.save)[0]+'.distributed.dpt'
        torch.distributed.init_process_group(backend=args.distributed_backend, world_size=args.world_size,
                                             init_method='tcp://localhost:6000', rank=args.rank)
    # Set the random seed manually for reproducibility.
    if args.seed is not None and args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    if args.loss_scale != 1 and args.dynamic_loss_scale:
        raise RuntimeError("Static loss scale and dynamic loss scale cannot be used together.")

    (train_data, val_data, test_data), tokenizer = data_config.apply(args)

    args.data_size = tokenizer.num_tokens
    model, optim, LR, LR_Warmer, criterion = setup_model_and_optim(args, train_data, tokenizer)

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
            if args.rank <= 0:
                with open(args.save+'.train_lock', 'wb') as f:
                    pass
            epoch_start_time = time.time()
            val_loss, skipped_iters = train(epoch, model, optim, train_data, LR, LR_Warmer, criterion,
                                            args, total_iters, skipped_iters, elapsed_time)
            elapsed_time += time.time() - epoch_start_time
            total_iters += args.train_iters
            if val_data is not None:
                print('entering eval')
                val_loss = evaluate(val_data, model, criterion, args)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(min(val_loss, 20))))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss and args.rank <= 0:
                torch.save(model.state_dict(), args.save)
                best_val_loss = val_loss
            if args.world_size == 1 or torch.distributed.get_rank() == 0:
                os.remove(args.save+'.train_lock')
            if args.world_size > 1:
                torch.distributed.barrier()
            torch.cuda.synchronize()

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    while os.path.exists(args.save+'.train_lock'):
        time.sleep(1)

    # Load the best saved model.
    if os.path.exists(args.save):
    #    with open(args.save, 'rb') as f:
        model.load_state_dict(torch.load(args.save, 'cpu'))

    if not args.no_weight_norm and args.rank <= 0:
        remove_weight_norm(rnn_model)
        with open(args.save, 'wb') as f:
            torch.save(model.state_dict(), f)

    if test_data is not None:
        # Run on test data.
        print('entering test')
        test_loss = evaluate(test_data, model, criterion, args)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(min(test_loss, 20))))
        print('=' * 89)

if __name__ == "__main__":
    main()



