import csv
import logging
import random
import misc
import os
import time
from datetime import datetime
from multiprocessing import Pool
from onmt.bin.preprocess import _get_parser, preprocess
from rdkit import Chem, RDLogger
from tqdm import tqdm
from utils import smi_tokenizer


def tokenize(_task):
    row, aug_factor = _task
    src_lines = []
    tgt_lines = []
    try:
        reactants, reagents, products = row["rxn_smiles"].split(">")
        mols_r = Chem.MolFromSmiles(reactants)
        mols_p = Chem.MolFromSmiles(products)
        if mols_r is None or mols_p is None:
            return [], []

        [a.ClearProp('molAtomMapNumber') for a in mols_r.GetAtoms()]
        [a.ClearProp('molAtomMapNumber') for a in mols_p.GetAtoms()]

        cano_smi_r = Chem.MolToSmiles(mols_r, isomericSmiles=True, canonical=True)
        cano_smi_p = Chem.MolToSmiles(mols_p, isomericSmiles=True, canonical=True)

        src_lines.append(f"{smi_tokenizer(cano_smi_r.strip())}\n")
        tgt_lines.append(f"{smi_tokenizer(cano_smi_p.strip())}\n")

        if aug_factor > 1:
            for _ in range(aug_factor - 1):
                smi_r = Chem.MolToSmiles(mols_r, isomericSmiles=True, doRandom=True)
                smi_p = Chem.MolToSmiles(mols_p, isomericSmiles=True, doRandom=True)

                smis_r = smi_r.split(".")
                random.shuffle(smis_r)
                smi_r = ".".join(smis_r)

                src_lines.append(f"{smi_tokenizer(smi_r.strip())}\n")
                tgt_lines.append(f"{smi_tokenizer(smi_p.strip())}\n")

    except Exception as e:
        logging.info(e)
        logging.info(row["rxn_smiles"].split(">"))
        return [], []

    return src_lines, tgt_lines


class ATProcessor:
    """Class for Augmented Transformer Preprocessing"""

    def __init__(self, args):
        self.model_name = args.model_name
        self.data_name = args.data_name
        self.log_file = args.log_file
        self.train_file = args.train_file
        self.val_file = args.val_file
        self.test_file = args.test_file
        self.processed_data_path = args.processed_data_path
        self.num_cores = args.num_cores
        self.aug_factor = args.aug_factor

        os.makedirs(self.processed_data_path, exist_ok=True)

        opt, _unknown = _get_parser().parse_known_args()
        self.model_args = opt

        logging.info("Overwriting model args")
        self.overwrite_model_args()
        misc.log_args(self.model_args, message="Updated model args")

        random.seed(args.seed)

    def overwrite_model_args(self):
        """Overwrite model args"""
        # Paths
        self.model_args.log_file = self.log_file
        self.model_args.save_data = os.path.join(self.processed_data_path, "bin")
        self.model_args.train_src = [os.path.join(self.processed_data_path, "src-train.txt")]
        self.model_args.train_tgt = [os.path.join(self.processed_data_path, "tgt-train.txt")]
        self.model_args.valid_src = os.path.join(self.processed_data_path, "src-val.txt")
        self.model_args.valid_tgt = os.path.join(self.processed_data_path, "tgt-val.txt")
        # Runtime args
        self.model_args.overwrite = True
        self.model_args.share_vocab = True
        self.model_args.src_seq_length = 1000
        self.model_args.tgt_seq_length = 1000
        self.model_args.subword_prefix = "ThisIsAHardCode"  # an arg for BART, leading to weird logging error

    def check_data_format(self) -> None:
        """Check that all files exists and the data format is correct for the first few lines"""
        check_count = 100

        logging.info(f"Checking the first {check_count} entries for each file")
        for fn in [self.train_file, self.val_file, self.test_file]:
            if not fn:
                continue
            assert os.path.exists(fn), f"{fn} does not exist!"

            with open(fn, "r") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for i, row in enumerate(csv_reader):
                    if i > check_count:
                        break

                    assert (c in row for c in ["id", "rxn_smiles"]), \
                        f"Error processing file {fn} line {i}, ensure columns 'id' and " \
                        f"'rxn_smiles' is included!"

                    reactants, reagents, products = row["rxn_smiles"].split(">")
                    Chem.MolFromSmiles(reactants)       # simply ensures that SMILES can be parsed
                    Chem.MolFromSmiles(products)        # simply ensures that SMILES can be parsed

        logging.info("Data format check passed")

    def preprocess(self) -> None:
        """Actual file-based preprocessing"""
        self.split_src_tgt(aug_factor=self.aug_factor)
        # preprocess(self.model_args)

    def split_src_tgt(self, aug_factor: int = 1):
        """Split reaction SMILES into source and target"""
        logging.info("Splitting reaction SMILES into source and target")
        p = Pool(self.num_cores)

        # for phase, fn in [("train", self.train_file),
        #                   ("val", self.val_file),
        #                   ("test", self.test_file)]:
        for phase, fn in [("test", self.test_file)]:
            ofn_src = os.path.join(self.processed_data_path, f"src-{phase}.txt")
            ofn_tgt = os.path.join(self.processed_data_path, f"tgt-{phase}.txt")
            # if os.path.exists(ofn_src) and os.path.exists(ofn_tgt):
            #     logging.info(f"{ofn_src} and {ofn_tgt} found! Skipping for phase {phase}")
            #     continue

            invalid_count = 0
            with open(fn, "r") as f, open(ofn_src, "w") as of_src, open(ofn_tgt, "w") as of_tgt:
                csv_reader = csv.DictReader(f)
                for src_lines, tgt_lines in tqdm(p.imap(tokenize,
                                                        ((row, aug_factor) for row in csv_reader))):
                    if src_lines and tgt_lines:
                        of_src.writelines(src_lines)
                        of_tgt.writelines(tgt_lines)
                    else:
                        invalid_count += 1

            logging.info(f"Invalid count: {invalid_count}")

        p.close()
        p.join()


if __name__ == "__main__":
    preprocess_parser = misc.get_parser()
    args, unknown = preprocess_parser.parse_known_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")
    os.makedirs("./logs/preprocess", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/preprocess/{args.log_file}.{dt}"
    logger = misc.setup_logger(args.log_file)
    misc.log_args(args, message="Logging arguments")

    start = time.time()

    processor = ATProcessor(args)
    processor.check_data_format()
    processor.preprocess()

    logging.info(f"Preprocessing done, total time: {time.time() - start: .2f} s")
