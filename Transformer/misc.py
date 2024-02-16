import argparse
import logging
import sys


def get_parser():
    parser = argparse.ArgumentParser("augmented_transformer")
    parser.add_argument("--model_name", help="model name", type=str, default="")
    parser.add_argument("--data_name", help="name of dataset, for easier reference", type=str, default="")
    parser.add_argument("--log_file", help="log file", type=str, default="")
    parser.add_argument("--train_file", help="train SMILES file", type=str, default="")
    parser.add_argument("--val_file", help="validation SMILES files", type=str, default="")
    parser.add_argument("--test_file", help="test SMILES files", type=str, default="")
    parser.add_argument("--processed_data_path", help="output path for processed data", type=str, default="")
    parser.add_argument("--model_path", help="model output path", type=str, default="")
    parser.add_argument("--test_output_path", help="test output path", type=str, default="")
    parser.add_argument("--test_unseen_path", help="test unseen path", type=str, default="")
    parser.add_argument("--checkpoint", help="checkpoint file name", type=str, default="")

    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument("--num_cores", help="number of cpu cores to use", type=int, default=None)
    parser.add_argument("--aug_factor", help="augmentation factor", type=int, default=1)

    return parser


def log_args(args, message: str):
    logging.info(message)
    for k, v in vars(args).items():
        logging.info(f"**** {k} = *{v}*")


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
