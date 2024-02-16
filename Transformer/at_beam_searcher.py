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
from utils import canonicalize_smiles
from at_processor import tokenize
import tree_builder_ver2
import pickle

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

        if not prediction or prediction == "9999":  # padding
            break

        prediction = canonicalize_smiles(prediction)
        v.append(prediction)

    return k, v


def pickle2kv(_args):
    prediction_row = _args
    total_paths=[]
    for node in prediction_row:
        path = node.get_path()
        total_paths.append(path)

    if total_paths:
        k = total_paths[0][0].plain_smiles
        v = [path[-1].plain_smiles for path in total_paths]
    else: k, v = None, None
    return k, v


def match_results(_args):
    global G_predictions
    test_row, n_best = _args
    predictions = G_predictions

    accuracy = np.zeros(n_best, dtype=np.float32)

    reactants, reagent, gt = test_row["rxn_smiles"].strip().split(">")
    k = canonicalize_smiles(reactants)

    if k not in predictions:
        logging.info(f"Reactant {reactants} not found in predictions (after canonicalization), skipping")
        return accuracy

    gt = canonicalize_smiles(gt)


    for j, prediction in enumerate(predictions[k]):
        prod_set = set(prediction.split('.'))
        if gt in prod_set:
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
        os.makedirs(self.processed_data_path, exist_ok=True)
        self.model_path = args.model_path
        self.test_output_path = args.test_output_path
        self.aug_factor = args.aug_factor
        self.test_unseen_path = args.test_unseen_path
        os.makedirs(self.test_output_path, exist_ok=True)
        self.test_unseen_name = self.test_unseen_path.split('/')[-1].split(".")[0]
        self.output_file = os.path.join(self.test_output_path, "predictions_{}.pickle".format(self.test_unseen_name))
        self.depth = 7 #Depth for beam search

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
        self.model_args.models = [os.path.join(self.model_path, args.checkpoint)]
        self.model_args.src = os.path.join(self.processed_data_path, "src-unseen-test.txt")
        self.model_args.output = os.path.join(self.test_output_path, "predictions_on_test_{}.txt".format(self.test_unseen_name))

    def translate_beam_search(self,opt):
        from onmt.utils.misc import split_corpus
        from onmt.translate.translator import build_translator
        translator = build_translator(opt, logger=logger, report_score=True)
        src_shards = split_corpus(opt.src, 1) #opt.shard_size)
        tgt_shards = split_corpus(opt.tgt, 1) #opt.shard_size)
        shard_pairs = zip(src_shards, tgt_shards)

        # scores, predictions = [], []
        with open(self.output_file, "wb") as file:
            results=[]
            for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
                k=3
                logger.info("Translating shard %d." % i)
                starting_node = tree_builder_ver2.Node([1, src_shard[0]])
                terminated_node, continue_node = starting_node.get_topK_leafs(k)

                iter=0
                while len(terminated_node) <k:
                    src_smiles=[node.smiles for node in continue_node]
                    try:
                        scores, predictions = translator.translate(
                            src=src_smiles,
                            tgt=tgt_shard,
                            batch_size=opt.batch_size,
                            batch_type=opt.batch_type,
                            attn_debug=opt.attn_debug,
                            align_debug=opt.align_debug
                        )
                    except:
                        break
                    for score, prediction, node in zip(scores, predictions, continue_node):
                        child=[[float(sc.cpu().data.numpy()), pred] for sc, pred in zip(score, prediction)]
                        node.add(child)

                    terminated_node, continue_node = starting_node.get_topK_leafs(k)

                    depths = [node.depth for node in continue_node]

                    if any(x > self.depth for x in depths):
                        break
                    if iter>50:
                        break
                    # print(iter)
                    iter+=1
                # pathways=[]
                # #TODO: if the reaction is not terminated, terminated_node will be None. It is a problem for match_results.
                if not terminated_node:
                    terminated_node = [starting_node, starting_node, starting_node]
                # for nodes in terminated_node:
                #     path = nodes.get_path()
                #     pathways.append([[node.plain_smiles, node.weighted_rank, node.rank] for node in path])

                pathways=[node for node in terminated_node]
                results.append(pathways)
                # print(pathways)
            pickle.dump(results, file)
        return True


    def preprocess(self):
        ofn = self.model_args.src

        with open(self.test_unseen_path, 'r') as file, open(ofn, 'w') as output:
            csv_reader = csv.DictReader(file)

            for _, row in enumerate(csv_reader):
                src, tgt = tokenize((row, 1))
                output.write(f"{src[0]}")

    def predict(self):
        """Actual file-based predicting, a wrapper to onmt.bin.translate()"""

        self.preprocess()
        self.translate_beam_search(self.model_args)
        self.score()

    def score(self):
        global G_predictions
        n_best = 3

        logging.info(f"Scoring predictions with model: {self.model_name}")

        # Load predictions and transform into a huge table {cano_prod: [cano_cand, ...]}
        logging.info(f"Loading predictions from {self.output_file}")
        predictions = {}
        p = multiprocessing.Pool(args.num_cores)

        with open(self.output_file, 'rb') as file:
            prediction_reader = pickle.load(file)
            for result in tqdm(p.imap(pickle2kv,
                                      (prediction_row for prediction_row in prediction_reader))):
                k, v = result
                predictions[k] = v
        G_predictions = predictions
        p.close()
        p.join()
        p = multiprocessing.Pool(args.num_cores)  # re-initialize to see the global variable

        # Results matching
        logging.info(f"Matching against ground truth from {self.test_unseen_path}")
        with open(self.test_unseen_path, "r") as test_csv:
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
