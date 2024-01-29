import logging
from rdkit import Chem
import networkx as nx
import numpy as np
import os
import re
from rdkit import Chem
from typing import Dict, List, Tuple
from utils.chem_utils import get_atom_features_sparse, get_bond_features
from utils.rxn_graphs import RxnGraph
from utils.train_utils import log_rank_0

MECH = False

def smi_tokenizer(smi: str):
    """Tokenize a SMILES molecule or reaction, adapted from https://github.com/pschwllr/MolecularTransformer"""
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == "".join(tokens)

    return " ".join(tokens)


def get_token_ids(tokens: list, vocab: Dict[str, int], max_len: int) -> Tuple[List, int]:
    token_ids = []
    token_ids.extend([vocab[token] for token in tokens])
    token_ids = token_ids[:max_len-1]
    token_ids.append(vocab["_EOS"])

    lens = len(token_ids)
    while len(token_ids) < max_len:
        token_ids.append(vocab["_PAD"])

    return token_ids, lens


def get_graph_from_smiles(smi: str):
    if not MECH:
        mol = Chem.MolFromSmiles(smi)
    else:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        mol.UpdatePropertyCache(strict=False)

    if mol is None or mol.GetNumBonds() == 0:
        mol = Chem.MolFromSmiles("CC")      # hardcode to ignore
    rxn_graph = RxnGraph(reac_mol=mol)

    if rxn_graph.G_undir.number_of_nodes() <= 2:
        logging.info("default: 2", smi)
    return rxn_graph


def get_graph_features_from_smi(_args):
    i, smi = _args
    assert isinstance(smi, str)
    if i > 0 and i % 10000 == 0:
        logging.info(f"Processing {i}th SMILES")

    atom_features = []
    bond_features = []
    edge_dict = {}

    if not smi.strip():
        smi = "CC"          # hardcode to ignore

    smi = "".join(smi.split())
    graph = get_graph_from_smiles(smi).reac_mol

    mol = graph.mol
    assert mol.GetNumAtoms() == len(graph.G_dir)

    G = nx.convert_node_labels_to_integers(graph.G_dir, first_label=0)

    # node iteration to get sparse atom features
    for v, attr in G.nodes(data="label"):
        atom_feat = get_atom_features_sparse(mol.GetAtomWithIdx(v))
        atom_features.append(atom_feat)

    a_graphs = [[] for _ in range(len(atom_features))]

    # edge iteration to get (dense) bond features
    for u, v, attr in G.edges(data='label'):
        bond_feat = get_bond_features(mol.GetBondBetweenAtoms(u, v))
        bond_feat = [u, v] + bond_feat
        bond_features.append(bond_feat)

        eid = len(edge_dict)
        edge_dict[(u, v)] = eid
        a_graphs[v].append(eid)

    b_graphs = [[] for _ in range(len(bond_features))]

    # second edge iteration to get neighboring edges (after edge_dict is updated fully)
    for bond_feat in bond_features:
        u, v = bond_feat[:2]
        eid = edge_dict[(u, v)]

        for w in G.predecessors(u):
            if not w == v:
                b_graphs[eid].append(edge_dict[(w, u)])

    # padding
    for a_graph in a_graphs:
        while len(a_graph) < 11:            # OH MY GOODNESS... Fe can be bonded to 10...
            a_graph.append(1e9)

    for b_graph in b_graphs:
        while len(b_graph) < 11:            # OH MY GOODNESS... Fe can be bonded to 10...
            b_graph.append(1e9)

    a_scopes = np.array(graph.atom_scope, dtype=np.int32)
    a_scopes_lens = a_scopes.shape[0]
    b_scopes = np.array(graph.bond_scope, dtype=np.int32)
    b_scopes_lens = b_scopes.shape[0]
    a_features = np.array(atom_features, dtype=np.int32)
    a_features_lens = a_features.shape[0]
    b_features = np.array(bond_features, dtype=np.int32)
    b_features_lens = b_features.shape[0]
    a_graphs = np.array(a_graphs, dtype=np.int32)
    b_graphs = np.array(b_graphs, dtype=np.int32)

    return a_scopes, a_scopes_lens, b_scopes, b_scopes_lens, \
        a_features, a_features_lens, b_features, b_features_lens, a_graphs, b_graphs


def make_vocab(args):
    logging.info(f"Making vocab")
    vocab = {}

    for phase in ["train", "val", "test"]:
        for fn in [
            os.path.join(args.processed_data_path, f"src-{phase}.txt"),
            os.path.join(args.processed_data_path, f"tgt-{phase}.txt")
        ]:
            with open(fn, "r") as f:
                for line in f:
                    tokens = line.strip().split()
                    for token in tokens:
                        if token in vocab:
                            vocab[token] += 1
                        else:
                            vocab[token] = 1

    vocab_file = os.path.join(args.processed_data_path, "vocab.txt")
    logging.info(f"Saving vocab into {vocab_file}")
    with open(vocab_file, "w") as of:
        of.write("_PAD\n_UNK\n_SOS\n_EOS\n")
        for token, count in vocab.items():
            of.write(f"{token}\t{count}\n")


def load_vocab(args) -> Dict[str, int]:
    vocab_file = os.path.join(args.processed_data_path, "vocab.txt")
    log_rank_0(f"Loading vocab from {vocab_file}")
    vocab = {}
    with open(vocab_file, "r") as f:
        for i, line in enumerate(f):
            token = line.strip().split("\t")[0]
            vocab[token] = i

    return vocab
