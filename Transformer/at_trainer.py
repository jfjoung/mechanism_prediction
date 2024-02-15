import logging
import misc
import numpy as np
import os
import random
import time
import torch
from datetime import datetime
from onmt.bin.train import _get_parser, train
from rdkit import RDLogger


class ATTrainer:
    """Class for Augmented Transformer Training"""

    def __init__(self, args):
        self.model_name = args.model_name
        self.data_name = args.data_name
        self.log_file = args.log_file
        self.processed_data_path = args.processed_data_path
        self.model_path = args.model_path

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        opt, _unknown = _get_parser().parse_known_args()
        self.model_args = opt

        logging.info("Overwriting model args")
        self.overwrite_model_args()
        misc.log_args(self.model_args, message="Updated model args")

    def overwrite_model_args(self):
        """Overwrite model args"""
        # Paths
        self.model_args.train_from = os.path.join(self.model_path, "model_step_1420000.pt")
        self.model_args.log_file = self.log_file
        self.model_args.data = os.path.join(self.processed_data_path, "bin")
        self.model_args.save_model = os.path.join(self.model_path, "model")

        # ---------------------
        # import glob
        # checkpoints = glob.glob(os.path.join(self.model_path, "model_step_*.pt"))
        # if checkpoints:
        #     last_checkpoint = sorted(checkpoints, reverse=True)[0]
        #     # self.model_args.train_from = os.path.join(self.model_path, last_checkpoint)
        #     self.model_args.train_from = os.path.join(self.model_path, "model_step_1070000.pt")


    def train(self):
        """A wrapper to onmt.bin.train()"""
        train(self.model_args)


if __name__ == "__main__":
    train_parser = misc.get_parser()
    args, unknown = train_parser.parse_known_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")
    os.makedirs("./logs/train", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/train/{args.log_file}.{dt}"
    logger = misc.setup_logger(args.log_file)
    misc.log_args(args, message="Logging arguments")

    start = time.time()

    trainer = ATTrainer(args)
    trainer.train()

    logging.info(f"Training done, total time: {time.time() - start: .2f} s")
