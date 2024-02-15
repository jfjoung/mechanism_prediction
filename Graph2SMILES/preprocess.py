import argparse
import csv
import logging
import numpy as np
import os
import sys
import time
import torch
from multiprocessing import Pool
from rdkit import Chem
from tqdm import tqdm
from typing import Tuple
from utils import parsing
from utils.preprocess_utils import \
    get_graph_features_from_smi, get_token_ids, \
    load_vocab, make_vocab, smi_tokenizer, MECH
from utils.train_utils import set_seed, setup_logger

# global vocab
G_vocab = {}


def get_preprocess_parser():
    parser = argparse.ArgumentParser("preprocess")
    parsing.add_common_args(parser)
    parsing.add_preprocess_args(parser)

    return parser


def _split_and_tokenize(row):
    try:
        reactants, reagents, products = row["rxn_smiles"].split(">")
        if not MECH:
            mols_r = Chem.MolFromSmiles(reactants)
            mols_p = Chem.MolFromSmiles(products)
        else:
            mols_r = Chem.MolFromSmiles(reactants, sanitize=False)
            mols_r.UpdatePropertyCache(strict=False)
            mols_p = Chem.MolFromSmiles(products, sanitize=False)
            mols_p.UpdatePropertyCache(strict=False)

        if mols_r is None or mols_p is None:
            return None, None

        [a.ClearProp('molAtomMapNumber') for a in mols_r.GetAtoms()]
        [a.ClearProp('molAtomMapNumber') for a in mols_p.GetAtoms()]

        if not MECH:
            cano_smi_r = Chem.MolToSmiles(mols_r, isomericSmiles=True, canonical=True)
            cano_smi_p = Chem.MolToSmiles(mols_p, isomericSmiles=True, canonical=True)
        else:
            cano_smi_r = Chem.MolToSmiles(mols_r, isomericSmiles=False, canonical=False)
            cano_smi_p = Chem.MolToSmiles(mols_p, isomericSmiles=False, canonical=False)

        src_line = f"{smi_tokenizer(cano_smi_r.strip())}\n"
        tgt_line = f"{smi_tokenizer(cano_smi_p.strip())}\n"

    except Exception as e:
        logging.info(e)
        logging.info(row["rxn_smiles"].split(">"))
        return None, None

    return src_line, tgt_line


def split_src_tgt(args):
    """Split reaction SMILES into source and target"""
    logging.info("Splitting reaction SMILES into source and target")
    p = Pool(args.num_cores)

    for phase, fn in [("train", args.train_file),
                      ("val", args.val_file),
                      ("test", args.test_file)]:
        ofn_src = os.path.join(args.processed_data_path, f"src-{phase}.txt")
        ofn_tgt = os.path.join(args.processed_data_path, f"tgt-{phase}.txt")
        if os.path.exists(ofn_src) and os.path.exists(ofn_tgt):
            logging.info(f"{ofn_src} and {ofn_tgt} found! Skipping for phase {phase}")
            continue

        invalid_count = 0
        with open(fn, "r") as f, open(ofn_src, "w") as of_src, open(ofn_tgt, "w") as of_tgt:
            csv_reader = csv.DictReader(f)
            for src_line, tgt_line in tqdm(p.imap(_split_and_tokenize, csv_reader)):
                if src_line and tgt_line:
                    of_src.write(src_line)
                    of_tgt.write(tgt_line)
                else:
                    invalid_count += 1

        logging.info(f"Invalid count: {invalid_count}")

    p.close()
    p.join()


def get_seq_features_from_line(_args) -> Tuple[np.ndarray, int, np.ndarray, int]:
    i, src_line, tgt_line, max_src_len, max_tgt_len = _args
    assert isinstance(src_line, str) and isinstance(tgt_line, str)
    if i > 0 and i % 10000 == 0:
        logging.info(f"Processing {i}th SMILES")

    src_tokens = src_line.strip().split()
    if not src_tokens:
        src_tokens = ["C", "C"]             # hardcode to ignore
    tgt_tokens = tgt_line.strip().split()

    global G_vocab
    src_token_ids, src_lens = get_token_ids(src_tokens, G_vocab, max_len=max_src_len)
    tgt_token_ids, tgt_lens = get_token_ids(tgt_tokens, G_vocab, max_len=max_tgt_len)

    src_token_ids = np.array(src_token_ids, dtype=np.int32)
    tgt_token_ids = np.array(tgt_token_ids, dtype=np.int32)

    return src_token_ids, src_lens, tgt_token_ids, tgt_lens


def binarize_g2s(args):
    for phase in ["train", "val", "test"]:
        src_file = os.path.join(args.processed_data_path, f"src-{phase}.txt")
        tgt_file = os.path.join(args.processed_data_path, f"tgt-{phase}.txt")
        output_file = os.path.join(args.processed_data_path, f"{phase}.npz")
        logging.info(f"Binarizing (g2s) src {src_file} and tgt {tgt_file}, saving to {output_file}")

        with open(src_file, "r") as f:
            src_lines = f.readlines()
        with open(tgt_file, "r") as f:
            tgt_lines = f.readlines()

        logging.info("Getting seq features")
        start = time.time()

        p = Pool(args.num_cores)
        seq_features_and_lengths = p.imap(
            get_seq_features_from_line,
            ((i, src_line, tgt_line, args.max_src_len, args.max_tgt_len)
             for i, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)))
        )

        p.close()
        p.join()

        seq_features_and_lengths = list(seq_features_and_lengths)

        logging.info(f"Done seq featurization, time: {time.time() - start}. Collating")
        src_token_ids, src_lens, tgt_token_ids, tgt_lens = zip(*seq_features_and_lengths)
        del seq_features_and_lengths

        src_token_ids = np.stack(src_token_ids, axis=0)
        src_lens = np.array(src_lens, dtype=np.int32)
        tgt_token_ids = np.stack(tgt_token_ids, axis=0)
        tgt_lens = np.array(tgt_lens, dtype=np.int32)

        logging.info("Getting graph features")
        start = time.time()

        p = Pool(args.num_cores)
        graph_features_and_lengths = p.imap(
            get_graph_features_from_smi,
            enumerate(src_lines)
        )

        p.close()
        p.join()

        graph_features_and_lengths = list(graph_features_and_lengths)
        logging.info(f"Done graph featurization, time: {time.time() - start}. Collating and saving...")
        a_scopes, a_scopes_lens, b_scopes, b_scopes_lens, a_features, a_features_lens, \
            b_features, b_features_lens, a_graphs, b_graphs = zip(*graph_features_and_lengths)
        del graph_features_and_lengths

        a_scopes = np.concatenate(a_scopes, axis=0)
        b_scopes = np.concatenate(b_scopes, axis=0)
        a_features = np.concatenate(a_features, axis=0)
        b_features = np.concatenate(b_features, axis=0)
        a_graphs = np.concatenate(a_graphs, axis=0)
        b_graphs = np.concatenate(b_graphs, axis=0)

        a_scopes_lens = np.array(a_scopes_lens, dtype=np.int32)
        b_scopes_lens = np.array(b_scopes_lens, dtype=np.int32)
        a_features_lens = np.array(a_features_lens, dtype=np.int32)
        b_features_lens = np.array(b_features_lens, dtype=np.int32)

        np.savez(
            output_file,
            src_token_ids=src_token_ids,
            src_lens=src_lens,
            tgt_token_ids=tgt_token_ids,
            tgt_lens=tgt_lens,
            a_scopes=a_scopes,
            b_scopes=b_scopes,
            a_features=a_features,
            b_features=b_features,
            a_graphs=a_graphs,
            b_graphs=b_graphs,
            a_scopes_lens=a_scopes_lens,
            b_scopes_lens=b_scopes_lens,
            a_features_lens=a_features_lens,
            b_features_lens=b_features_lens
        )


def preprocess_main(args):
    parsing.log_args(args, phase="preprocessing")
    os.makedirs(args.processed_data_path, exist_ok=True)

    split_src_tgt(args)
    make_vocab(args)

    global G_vocab
    G_vocab = load_vocab(args)

    binarize_g2s(args)


if __name__ == "__main__":
    preprocess_parser = get_preprocess_parser()
    args, unknown = preprocess_parser.parse_known_args()

    # set random seed
    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args)

    # maximize display for debugging
    np.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(profile="full")

    start = time.time()
    preprocess_main(args)
    logging.info(f"Preprocessing done, total time: {time.time() - start: .2f} s")
