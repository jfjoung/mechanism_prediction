import argparse
import datetime
import glob
import numpy as np
import os
import sys
import time
import torch
import torch.distributed as dist
from models.graph2smiles import Graph2SMILES
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from train import get_model
from utils import parsing
from utils.data_utils import canonicalize_smiles, load_vocab, G2SDataset
from utils.train_utils import log_tensor, log_rank_0, param_count, set_seed, setup_logger


def get_predict_parser():
    parser = argparse.ArgumentParser("predict")
    parsing.add_common_args(parser)
    parsing.add_preprocess_args(parser)
    parsing.add_train_args(parser)
    parsing.add_predict_args(parser)

    return parser


def get_predictions(args, model, vocab_tokens, test_loader, device, start):
    all_predictions = []
    with torch.no_grad():
        for test_idx, test_batch in enumerate(test_loader):
            if test_idx % args.log_iter == 0:
                log_rank_0(f"Doing inference on test step {test_idx}, "
                           f"time: {time.time() - start: .2f} s")

            test_batch.to(device)
            results = model.predict_step(
                reaction_batch=test_batch,
                batch_size=test_batch.size,
                beam_size=args.beam_size,
                n_best=args.n_best,
                temperature=args.temperature,
                min_length=args.predict_min_len,
                max_length=args.predict_max_len
            )

            for predictions, scores in zip(results["predictions"], results["scores"]):
                smis_with_scores = []
                for prediction, score in zip(predictions, scores):
                    predicted_idx = prediction.detach().cpu().numpy()
                    score = score.detach().cpu().numpy()
                    predicted_tokens = [vocab_tokens[idx] for idx in predicted_idx[:-1]]
                    smi = "".join(predicted_tokens)
                    smis_with_scores.append(f"{smi}_{score}")
                smis_with_scores = ",".join(smis_with_scores)
                all_predictions.append(f"{smis_with_scores}\n")

    return all_predictions


def main(args):
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = args.device
    if args.local_rank != -1:
        dist.init_process_group(backend=args.backend, init_method='env://', timeout=datetime.timedelta(0, 7200))
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = True

    if torch.distributed.is_initialized():
        log_rank_0(f"Device rank: {torch.distributed.get_rank()}")
    os.makedirs(os.path.join("./results", args.data_name), exist_ok=True)

    parsing.log_args(args, phase="prediction")

    # initialization ---------------------------------- ckpt parsing
    if args.do_validate:
        checkpoints = glob.glob(os.path.join(args.load_from, "*.pt"))
        checkpoints = sorted(
            checkpoints,
            key=lambda ckpt: int(ckpt.split(".")[-2].split("_")[-1]),
            reverse=True
        )
        checkpoints = [ckpt for ckpt in checkpoints
                       if (args.checkpoint_step_start <= int(ckpt.split(".")[-2].split("_")[0]))
                       and (args.checkpoint_step_end >= int(ckpt.split(".")[-2].split("_")[0]))]
        file_bin = os.path.join(args.processed_data_path, "val.npz")
        file_tgt = os.path.join(args.processed_data_path, "tgt-val.txt")
    elif args.do_predict:
        checkpoints = [os.path.join(args.model_path, args.load_from)]
        file_bin = os.path.join(args.processed_data_path, "test.npz")
        file_tgt = os.path.join(args.processed_data_path, "tgt-test.txt")
    else:
        raise ValueError("Either --do_validate or --do_predict need to be specified!")

    model = None
    test_dataset = None
    vocab_tokens = None
    smis_tgt = []
    start = time.time()
    for ckpt_i, checkpoint in enumerate(checkpoints):
        result_file = os.path.join(args.test_output_path, f"result.{ckpt_i}")
        result_stat_file = os.path.join(args.test_output_path, f"result.stat.{ckpt_i}")

        if os.path.exists(result_file) and False:
            log_rank_0(f"Result file found at {result_file}, skipping prediction.")
        else:
            log_rank_0(f"Loading from {checkpoint}")
            try:
                state = torch.load(checkpoint)
            except RuntimeError:
                log_rank_0(f"Error loading {checkpoint}, skipping")
                continue            # some weird corrupted files

            pretrain_args = state["args"]
            pretrain_args.load_from = None # Hard code from tracing back to older models

            pretrain_state_dict = state["state_dict"]
            pretrain_args.local_rank = args.local_rank
            if not hasattr(pretrain_args, "n_latent"):
                pretrain_args.n_latent = 1
            args.n_latent = pretrain_args.n_latent
            if not hasattr(pretrain_args, "shared_attention_layer"):
                pretrain_args.shared_attention_layer = 0

            if model is None:
                # initialization ---------------------------------- model
                log_rank_0("Model is None, building model")
                log_rank_0("First logging args for training")
                parsing.log_args(pretrain_args, phase="training")

                # backward
                assert args.model == pretrain_args.model or \
                    pretrain_args.model == "g2s_series_rel", \
                    f"Pretrained model is {pretrain_args.model}!"
                model_class = Graph2SMILES
                dataset_class = G2SDataset
                args.compute_graph_distance = True

                # initialization ---------------------------------- vocab
                vocab = load_vocab(args)
                vocab_tokens = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]

                model, state = get_model(pretrain_args, model_class, vocab, device)
                if hasattr(model, "module"):
                    model = model.module        # unwrap DDP model to enable accessing model func directly

                log_rank_0(model)
                log_rank_0(f"Number of parameters = {param_count(model)}")

                # initialization ---------------------------------- data
                test_dataset = dataset_class(args, file=file_bin)
                test_dataset.batch(
                    batch_type=args.batch_type,
                    batch_size=args.predict_batch_size
                )
                with open(file_tgt, "r") as f:
                    total = sum(1 for _ in f)

                with open(file_tgt, "r") as f:
                    for line_tgt in f:
                        smi_tgt = "".join(line_tgt.split())
                        smi_tgt = canonicalize_smiles(smi_tgt)
                        smis_tgt.append(smi_tgt)

            pretrain_state_dict = {k.replace("module.", ""): v for k, v in pretrain_state_dict.items()}
            model.load_state_dict(pretrain_state_dict)
            log_rank_0(f"Loaded pretrained state_dict from {checkpoint}")
            model.eval()

            if args.local_rank != -1:
                test_sampler = DistributedSampler(test_dataset, shuffle=False)
            else:
                test_sampler = SequentialSampler(test_dataset)

            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=1,
                sampler=test_sampler,
                num_workers=args.num_cores,
                collate_fn=lambda _batch: _batch[0],
                pin_memory=True
            )

            all_predictions = get_predictions(
                args, model, vocab_tokens, test_loader, device, start)

            if args.local_rank > 0:
                continue

            # saving prediction results
            with open(result_file, "w") as of:
                of.writelines(all_predictions)

        if args.do_score:
            if os.path.exists(result_stat_file) and False:
                log_rank_0(f"Result stat file found at {result_stat_file}, skipping scoring.")
                continue

            invalid = 0
            accuracies = np.zeros([total, args.n_best], dtype=np.float32)

            with open(result_file, "r") as f_predict:
                for i, (smi_tgt, line_predict) in enumerate(zip(smis_tgt, f_predict)):
                    if smi_tgt == "CC":         # problematic SMILES
                        continue

                    line_predict = "".join(line_predict.split())
                    smis_predict = line_predict.split(",")
                    smis_predict = [smi.split("_")[0] for smi in smis_predict]
                    smis_predict = [canonicalize_smiles(smi, trim=False, suppress_warning=True) for smi in smis_predict]
                    if not smis_predict[0]:
                        invalid += 1
                    smis_predict = [smi for smi in smis_predict if smi and not smi == "CC"]
                    smis_predict = list(dict.fromkeys(smis_predict))

                    for j, smi in enumerate(smis_predict[:args.n_best]):
                        if smi == smi_tgt:
                            accuracies[i, j:] = 1.0
                            break

            with open(result_stat_file, "w") as of:
                line = f"Total: {total}, top 1 invalid: {invalid / total * 100: .2f} %"
                log_rank_0(line)
                of.write(f"{line}\n")

                mean_accuracies = np.mean(accuracies, axis=0)
                for n in range(args.n_best):
                    line = f"Top {n+1} accuracy: {mean_accuracies[n] * 100: .2f} %"
                    log_rank_0(line)
                    of.write(f"{line}\n")

            log_rank_0(f"Elapsed time: {time.time() - start: .2f} s")


if __name__ == "__main__":
    # initialization ---------------------------------- args, logs and devices
    predict_parser = get_predict_parser()
    args = predict_parser.parse_args()

    setup_logger(args, warning_off=True)
    np.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(profile="full")

    set_seed(args.seed)

    main(args)
