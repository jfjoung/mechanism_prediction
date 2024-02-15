import csv
import glob
import logging
import misc
import multiprocessing
import numpy as np
import os
import random
import time
import torch
from datetime import datetime
from onmt.bin.translate import _get_parser, translate
from rdkit import RDLogger
from tqdm import tqdm
from utils import canonicalize_smiles, MECH

global G_predictions


def csv2kv(_args):
    prediction_row, n_best = _args
    k = canonicalize_smiles(prediction_row["prod_smi"])
    v = []

    for i in range(n_best):
        try:
            prediction = prediction_row[f"cand_precursor_{i + 1}"]
        except KeyError:
            break

        if not prediction or prediction == "9999":          # padding
            break

        prediction = canonicalize_smiles(prediction)
        v.append(prediction)

    return k, v


def match_results(_args):
    global G_predictions
    test_row, n_best = _args
    predictions = G_predictions

    accuracy = np.zeros(n_best, dtype=np.float32)

    reactants, reagent, gt = test_row["rxn_smiles"].strip().split(">")
    k = canonicalize_smiles(reactants)

    if k not in predictions:
        logging.info(f"Product {reactants} not found in predictions (after canonicalization), skipping")
        return accuracy

    gt = canonicalize_smiles(gt)
    for j, prediction in enumerate(predictions[k]):
        if prediction == gt:
            accuracy[j:] = 1.0
            break

    return accuracy


class ATPredictor:
    """Class for Augmented Transformer Testing"""

    def __init__(self, args):
        self.model_name = args.model_name
        self.data_name = args.data_name
        self.log_file = args.log_file
        self.processed_data_path = args.processed_data_path
        self.model_path = args.model_path
        self.test_output_path = args.test_output_path
        self.aug_factor = args.aug_factor
        self.output_file = os.path.join(self.test_output_path, "predictions.csv")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        opt, _unknown = _get_parser().parse_known_args()
        self.model_args = opt

        logging.info("Overwriting model args, (hardcoding essentially)")
        self.overwrite_model_args()
        misc.log_args(self.model_args, message="Updated model args")

    def overwrite_model_args(self):
        """Overwrite model args"""
        # Paths
        # Overwriting model path with the last checkpoint
        checkpoints = glob.glob(os.path.join(self.model_path, "model_step_*.pt"))
        last_checkpoint = sorted(checkpoints, reverse=True)[0]
        # self.model_args.models = [self.model_path]
        # last_checkpoint = os.path.join(self.model_path, "model_step_1250000.pt")
        self.model_args.models = [last_checkpoint]
        self.model_args.src = os.path.join(self.processed_data_path, "src-test-cano.txt")
        self.model_args.output = os.path.join(self.test_output_path, "predictions_on_test.txt")

    def translate_own(self, opt):
        from onmt.utils.logging import init_logger
        from onmt.utils.misc import split_corpus
        from onmt.translate.translator import build_translator

        import onmt.opts as opts
        from onmt.utils.parse import ArgumentParser

        translator = build_translator(opt, logger=logger, report_score=True)
        src_shards = split_corpus(opt.src, opt.shard_size)
        tgt_shards = split_corpus(opt.tgt, opt.shard_size)
        shard_pairs = zip(src_shards, tgt_shards)

        scores, predictions = [], []
        for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
            logger.info("Translating shard %d." % i)
            score, prediction = translator.translate(
                src=src_shard,
                tgt=tgt_shard,
                src_dir=opt.src_dir,
                batch_size=opt.batch_size,
                batch_type=opt.batch_type,
                attn_debug=opt.attn_debug,
                align_debug=opt.align_debug
                )
            scores.append(score)
            predictions.append(prediction)
            
        yield scores, predictions


    def predict(self):
        """Actual file-based predicting, a wrapper to onmt.bin.translate()"""
        # if os.path.exists(self.model_args.output):
        #     logging.info(f"Results found at {self.model_args.output}, skip prediction.")
        # else:
        #     self.keep_cano()
        #     translate(self.model_args)

        self.keep_cano()
        translate(self.model_args)
        # scores, predictions = self.translate_own(self.model_args)
        # print(scores)
        # print(predictions)
        logger = misc.setup_logger(args.log_file)
        self.compile_into_csv()
        self.score()

    def keep_cano(self):
        fn = os.path.join(self.processed_data_path, "src-test.txt")
        ofn = self.model_args.src
        logging.info(f"Truncating {fn} to {ofn}, keeping only first (canonical) SMILES")

        # with open(fn, "r") as f, open(ofn, "w") as of:
        #     for i, line in enumerate(f):
        #         if i % self.aug_factor == 0 and not MECH:
        #             of.write(line)
        #         else:
        #             of.write(line)
        
        with open(fn, "r") as f, open(ofn, "w") as of:
            for i, line in enumerate(f):
                if i % self.aug_factor == 0:
                    of.write(line)
    
    def compile_into_csv(self):
        logging.info("Compiling into predictions.csv")

        src_file = os.path.join(self.processed_data_path, "src-test-cano.txt")

        with open(src_file, "r") as f:
            total_src = sum(1 for _ in f)

        with open(self.model_args.output, "r") as f:
            total_gen = sum(1 for _ in f)

        n_best = self.model_args.n_best
        assert total_src == total_gen / n_best, \
            f"File length mismatch! Source total: {total_src}, " \
            f"prediction total: {total_gen}, n_best: {n_best}"

        proposed_col_names = [f'cand_precursor_{i}' for i in range(1, self.model_args.n_best + 1)]
        headers = ['prod_smi']
        headers.extend(proposed_col_names)

        with open(src_file, "r") as src_f, \
                open(self.model_args.output, "r") as pred_f, \
                open(self.output_file, "w") as of:
            header_line = ",".join(headers)
            of.write(f"{header_line}\n")

            for src_line in src_f:
                of.write("".join(src_line.strip().split()))

                for j in range(n_best):
                    cand = pred_f.readline()
                    of.write(",")
                    of.write("".join(cand.strip().split()))
                of.write("\n")

    def score(self):
        global G_predictions
        n_best = 50

        logging.info(f"Scoring predictions with model: {self.model_name}")

        # Load predictions and transform into a huge table {cano_prod: [cano_cand, ...]}
        logging.info(f"Loading predictions from {self.output_file}")
        predictions = {}
        p = multiprocessing.Pool(args.num_cores)

        with open(self.output_file, "r") as prediction_csv:
            prediction_reader = csv.DictReader(prediction_csv)
            for result in tqdm(p.imap(csv2kv,
                                      ((prediction_row, n_best) for prediction_row in prediction_reader))):
                k, v = result
                predictions[k] = v

        G_predictions = predictions

        p.close()
        p.join()
        p = multiprocessing.Pool(args.num_cores)  # re-initialize to see the global variable

        # Results matching
        logging.info(f"Matching against ground truth from {args.test_file}")
        with open(args.test_file, "r") as test_csv:
            test_reader = csv.DictReader(test_csv)
            accuracies = p.imap(match_results,
                                ((test_row, n_best) for test_row in test_reader))
            accuracies = np.stack(list(accuracies))

        p.close()
        p.join()

        # Log statistics
        mean_accuracies = np.mean(accuracies, axis=0)
        for n in range(n_best):
            logging.info(f"Top {n + 1} accuracy: {mean_accuracies[n]}")


if __name__ == "__main__":
    train_parser = misc.get_parser()
    args, unknown = train_parser.parse_known_args()

    # logger setup
    RDLogger.DisableLog("rdApp.*")
    os.makedirs("./logs/predict", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/predict/{args.log_file}.{dt}"
    logger = misc.setup_logger(args.log_file)
    misc.log_args(args, message="Logging arguments")

    start = time.time()

    predictor = ATPredictor(args)
    predictor.predict()

    logging.info(f"Prediction done, total time: {time.time() - start: .2f} s")
