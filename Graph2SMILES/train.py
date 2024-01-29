import argparse
import datetime
import logging
import numpy as np
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from models.graph2smiles import Graph2SMILES
import time
from torch.nn.init import xavier_uniform_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from typing import Dict
from utils import parsing
from utils.data_utils import G2SDataset
from utils.preprocess_utils import load_vocab
from utils.train_utils import get_lr, grad_norm, log_rank_0, NoamLR, \
    param_count, param_norm, set_seed, setup_logger


def get_train_parser():
    parser = argparse.ArgumentParser("train")
    parsing.add_common_args(parser)
    parsing.add_train_args(parser)
    parsing.add_predict_args(parser)

    return parser


def init_dist(args):
    if args.local_rank != -1:
        dist.init_process_group(backend=args.backend,
                                init_method='env://',
                                timeout=datetime.timedelta(0, 7200))
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = False

    if dist.is_initialized():
        logging.info(f"Device rank: {dist.get_rank()}")
        sys.stdout.flush()


def get_model(args, model_class, vocab: Dict[str, int], device):
    state = {}
    if args.load_from:
        log_rank_0(f"Loading pretrained state from {args.load_from}")
        state = torch.load(args.load_from, map_location=torch.device("cpu"))
        pretrain_args = state["args"]
        pretrain_args.local_rank = args.local_rank
        parsing.log_args(pretrain_args, phase="pretraining")

        model = model_class(pretrain_args, vocab)
        pretrain_state_dict = state["state_dict"]
        pretrain_state_dict = {k.replace("module.", ""): v for k, v in pretrain_state_dict.items()}
        model.load_state_dict(pretrain_state_dict)
        log_rank_0("Loaded pretrained model state_dict.")
    else:
        model = model_class(args, vocab)
        for p in model.parameters():
            if p.dim() > 1 and p.requires_grad:
                xavier_uniform_(p)

    model.to(device)
    if args.local_rank != -1:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
        log_rank_0("DDP setup finished")

    return model, state


def get_optimizer_and_scheduler(args, model, state):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    scheduler = NoamLR(
        optimizer,
        model_size=args.decoder_hidden_size,
        warmup_steps=args.warmup_steps
    )

    if state and args.resume:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        log_rank_0("Loaded pretrained optimizer and scheduler state_dicts.")

    return optimizer, scheduler


def init_loader(args, dataset, batch_size: int, bucket_size: int = 1000,
                shuffle: bool = False, epoch: int = None):
    dataset.sort()
    dataset.shuffle_in_bucket(bucket_size=bucket_size)
    dataset.batch(
        batch_type=args.batch_type,
        batch_size=batch_size
    )

    if args.local_rank != -1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        if epoch is not None:
            sampler.set_epoch(epoch)
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_cores,
        collate_fn=lambda _batch: _batch[0],
        pin_memory=True
    )

    return loader


def _optimize(args, model, optimizer, scheduler):
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
    optimizer.step()
    scheduler.step()
    g_norm = grad_norm(model)
    model.zero_grad(set_to_none=True)

    return g_norm


def train_main(args):
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = args.device

    init_dist(args)
    parsing.log_args(args, phase="training")

    vocab_file = os.path.join(args.processed_data_path, "vocab.txt")
    if not os.path.exists(vocab_file):
        raise ValueError(f"Vocab file {vocab_file} not found!")
    vocab = load_vocab(args)

    os.makedirs(args.model_path, exist_ok=True)
    model_class = Graph2SMILES
    dataset_class = G2SDataset
    assert args.compute_graph_distance

    model, state = get_model(args, model_class, vocab, device)
    log_rank_0(model)
    log_rank_0(f"Number of parameters = {param_count(model)}")

    optimizer, scheduler = get_optimizer_and_scheduler(args, model, state)

    train_bin = os.path.join(args.processed_data_path, "train.npz")
    val_bin = os.path.join(args.processed_data_path, "val.npz")

    train_dataset = dataset_class(args, file=train_bin)
    val_dataset = dataset_class(args, file=val_bin)

    total_step = state["total_step"] if state else 0
    accum = 0
    g_norm = 0
    losses, accs, ems = [], [], []
    o_start = time.time()
    log_rank_0("Start training")

    for epoch in range(args.epoch):
        model.train()
        model.zero_grad(set_to_none=True)
        train_loader = init_loader(args, train_dataset,
                                   batch_size=args.train_batch_size,
                                   shuffle=True,
                                   epoch=epoch)
        for batch_idx, train_batch in enumerate(train_loader):
            if total_step > args.max_steps:
                log_rank_0("Max steps reached, finish training")
                exit(0)
            train_batch.to(device)
            batch_losses, acc, em = model(train_batch)
            loss = batch_losses.mean()
            loss.backward()
            losses.append(loss.item())
            accs.append(acc.item() * 100)
            ems.append(em.item() * 100)

            accum += 1
            if accum == args.accumulation_count:
                _optimize(args, model, optimizer, scheduler)
                accum = 0
                total_step += 1

            if (accum == 0) and (total_step > 0) and (total_step % args.log_iter == 0):
                log_rank_0(f"Step {total_step}, loss: {np.mean(losses)}, "
                           f"acc: {np.mean(accs): .4f}, em: {np.mean(ems): .4f}, "
                           f"p_norm: {param_norm(model): .4f}, g_norm: {g_norm: .4f}, "
                           f"lr: {get_lr(optimizer): .6f}, "
                           f"elapsed time: {time.time() - o_start: .0f}")
                losses, accs, ems = [], [], []

            if (accum == 0) and (total_step > 0) and (total_step % args.eval_iter == 0):
                model.eval()
                val_count = 100
                val_losses, val_accs, val_ems = [], [], []

                val_loader = init_loader(args, val_dataset,
                                         batch_size=args.val_batch_size,
                                         shuffle=True,
                                         epoch=None)
                with torch.no_grad():
                    for val_idx, val_batch in enumerate(val_loader):
                        if val_idx >= val_count:
                            break
                        val_batch.to(device)
                        val_batch_losses, val_acc, val_em = model(val_batch)
                        val_loss = val_batch_losses.mean()
                        val_losses.append(val_loss.item())
                        val_accs.append(val_acc.item() * 100)
                        val_ems.append(val_em.item() * 100)

                log_rank_0(f"Validation (with teacher) at step {total_step}, "
                           f"val loss: {np.mean(val_losses)}, "
                           f"val acc: {np.mean(val_accs): .4f}, "
                           f"val em: {np.mean(val_ems): .4f}")
                model.train()

            # Important: saving only at one node or the ckpt would be corrupted!
            if dist.is_initialized() and dist.get_rank() > 0:
                continue

            if (accum == 0) and (total_step > 0) and (total_step % args.save_iter == 0):
                n_iter = total_step // args.save_iter - 1
                log_rank_0(f"Saving at step {total_step}")
                state = {
                    "args": args,
                    "total_step": total_step,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                torch.save(state, os.path.join(args.model_path, f"model.{total_step}_{n_iter}.pt"))

        # lastly
        if (args.accumulation_count > 1) and (accum > 0):
            _optimize(args, model, optimizer, scheduler)
            accum = 0
            # total_step += 1           # for partial batch, do not increase total_step

        if args.local_rank != -1:
            dist.barrier()


if __name__ == "__main__":
    train_parser = get_train_parser()
    args = train_parser.parse_args()

    # set random seed
    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args)

    # maximize display for debugging
    np.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(profile="full")

    args.local_rank = int(os.environ["LOCAL_RANK"]) if os.environ.get("LOCAL_RANK") else -1
    train_main(args)
