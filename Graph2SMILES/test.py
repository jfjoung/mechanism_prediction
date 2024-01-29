import networkx as nx
from rdkit import Chem
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
