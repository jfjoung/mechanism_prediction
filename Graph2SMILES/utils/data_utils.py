import logging
import numpy as np
import os
import time
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from typing import List, Tuple
from utils.chem_utils import ATOM_FDIM, BOND_FDIM, get_atom_features_sparse
from utils.ctypes_calculator import DistanceCalculator
from utils.preprocess_utils import load_vocab, MECH
from utils.train_utils import log_rank_0


def len2idx(lens) -> np.ndarray:
    end_indices = np.cumsum(lens)
    start_indices = np.concatenate([[0], end_indices[:-1]], axis=0)
    indices = np.stack([start_indices, end_indices], axis=1)

    return indices


def canonicalize_smiles(smiles, remove_atom_number=True, trim=True, suppress_warning=False):
    # print('Mech,', MECH)
    if not MECH:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            cano_smiles = ""

        else:
            try:
                if trim and mol.GetNumHeavyAtoms() < 2:
                    if not suppress_warning:
                        logging.info(f"Problematic smiles: {smiles}, setting it to 'CC'")
                    cano_smiles = "CC"          # TODO: hardcode to ignore
                else:
                    if remove_atom_number:
                        [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
                    cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            except RuntimeError as e:
                cano_smiles = ""

        return cano_smiles
    else:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)

        if mol is None:
            cano_smiles = ""
        else:
            mol.UpdatePropertyCache(strict=False)
            try:
                if trim and mol.GetNumHeavyAtoms() < 2:
                    if not suppress_warning:
                        logging.info(f"Problematic smiles: {smiles}, setting it to 'CC'")
                    cano_smiles = "CC"          # TODO: hardcode to ignore
                else:
                    if remove_atom_number:
                        [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]
                    cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            except RuntimeError as e:
                cano_smiles = ""

        return cano_smiles



class G2SBatch:
    def __init__(self,
                 fnode: torch.Tensor,
                 fmess: torch.Tensor,
                 agraph: torch.Tensor,
                 bgraph: torch.Tensor,
                 atom_scope: List,
                 bond_scope: List,
                 tgt_token_ids: torch.Tensor,
                 tgt_lengths: torch.Tensor,
                 distances: torch.Tensor = None):
        self.fnode = fnode
        self.fmess = fmess
        self.agraph = agraph
        self.bgraph = bgraph
        self.atom_scope = atom_scope
        self.bond_scope = bond_scope
        self.tgt_token_ids = tgt_token_ids
        self.tgt_lengths = tgt_lengths
        self.distances = distances

        self.size = len(tgt_lengths)

    def to(self, device):
        self.fnode = self.fnode.to(device)
        self.fmess = self.fmess.to(device)
        self.agraph = self.agraph.to(device)
        self.bgraph = self.bgraph.to(device)
        self.tgt_token_ids = self.tgt_token_ids.to(device)
        self.tgt_lengths = self.tgt_lengths.to(device)

        if self.distances is not None:
            self.distances = self.distances.to(device)

    def pin_memory(self):
        self.fnode = self.fnode.pin_memory()
        self.fmess = self.fmess.pin_memory()
        self.agraph = self.agraph.pin_memory()
        self.bgraph = self.bgraph.pin_memory()
        self.tgt_token_ids = self.tgt_token_ids.pin_memory()
        self.tgt_lengths = self.tgt_lengths.pin_memory()

        if self.distances is not None:
            self.distances = self.distances.pin_memory()

        return self

    def log_tensor_shape(self):
        log_rank_0(f"fnode: {self.fnode.shape}, "
                   f"fmess: {self.fmess.shape}, "
                   f"tgt_token_ids: {self.tgt_token_ids.shape}, "
                   f"tgt_lengths: {self.tgt_lengths}")


class G2SDataset(Dataset):
    def __init__(self, args, file: str):
        self.args = args

        self.a_scopes = []
        self.b_scopes = []
        self.a_features = []
        self.b_features = []
        self.a_graphs = []
        self.b_graphs = []
        self.a_scopes_lens = []
        self.b_scopes_lens = []
        self.a_features_lens = []
        self.b_features_lens = []

        self.src_token_ids = []             # loaded but not batched
        self.src_lens = []
        self.tgt_token_ids = []
        self.tgt_lens = []

        self.data_indices = []
        self.batch_sizes = []
        self.batch_starts = []
        self.batch_ends = []

        self.vocab = load_vocab(args)
        self.vocab_tokens = [k for k, v in sorted(self.vocab.items(), key=lambda tup: tup[1])]

        log_rank_0(f"Loading preprocessed features from {file}")
        feat = np.load(file)
        for attr in ["a_scopes", "b_scopes", "a_features", "b_features", "a_graphs", "b_graphs",
                     "a_scopes_lens", "b_scopes_lens", "a_features_lens", "b_features_lens",
                     "src_token_ids", "src_lens", "tgt_token_ids", "tgt_lens"]:
            setattr(self, attr, feat[attr])

        # mask out chiral tag (as UNSPECIFIED)
        if args.mask_rel_chirality == 1:
            self.a_features[:, 6] = 2

        assert len(self.a_scopes_lens) == len(self.b_scopes_lens) == \
               len(self.a_features_lens) == len(self.b_features_lens) == \
               len(self.src_token_ids) == len(self.src_lens) == \
               len(self.tgt_token_ids) == len(self.tgt_lens), \
               f"Lengths of source and target mismatch!"

        self.a_scopes_indices = len2idx(self.a_scopes_lens)
        self.b_scopes_indices = len2idx(self.b_scopes_lens)
        self.a_features_indices = len2idx(self.a_features_lens)
        self.b_features_indices = len2idx(self.b_features_lens)

        del self.a_scopes_lens, self.b_scopes_lens, self.a_features_lens, self.b_features_lens

        self.data_size = len(self.src_token_ids)
        self.data_indices = np.arange(self.data_size)

        self.distance_calculator = None
        if self.args.compute_graph_distance:
            self.distance_calculator = DistanceCalculator()

        log_rank_0(f"Loaded and initialized G2SDataset, size: {self.data_size}")

    def sort(self):
        if self.args.verbose:
            start = time.time()
            log_rank_0("Calling G2SDataset.sort()")
            self.data_indices = np.argsort(self.src_lens)
            log_rank_0(f"Done, time: {time.time() - start: .2f} s")
        else:
            self.data_indices = np.argsort(self.src_lens)

    def shuffle_in_bucket(self, bucket_size: int):
        if self.args.verbose:
            start = time.time()
            log_rank_0("Calling G2SDataset.shuffle_in_bucket()")
            for i in range(0, self.data_size, bucket_size):
                np.random.shuffle(self.data_indices[i:i+bucket_size])
            log_rank_0(f"Done, time: {time.time() - start: .2f} s")
        else:
            for i in range(0, self.data_size, bucket_size):
                np.random.shuffle(self.data_indices[i:i + bucket_size])

    def batch(self, batch_type: str, batch_size: int):
        start = time.time()
        log_rank_0("Calling G2SDataset.batch()")

        self.batch_sizes = []
        if batch_type in ["samples", "atoms"]:
            raise NotImplementedError

        elif batch_type.startswith("tokens"):
            sample_size = 0
            max_batch_src_len = 0
            max_batch_tgt_len = 0

            for data_idx in self.data_indices:
                src_len = self.src_lens[data_idx]
                tgt_len = self.tgt_lens[data_idx]

                max_batch_src_len = max(src_len, max_batch_src_len)
                max_batch_tgt_len = max(tgt_len, max_batch_tgt_len)
                while self.args.enable_amp and not max_batch_src_len % 8 == 0:  # for amp
                    max_batch_src_len += 1
                while self.args.enable_amp and not max_batch_tgt_len % 8 == 0:  # for amp
                    max_batch_tgt_len += 1

                if batch_type == "tokens" and \
                        max_batch_src_len * (sample_size + 1) <= batch_size:
                    sample_size += 1
                elif batch_type == "tokens_sum" and \
                        (max_batch_src_len + max_batch_tgt_len) * (sample_size + 1) <= batch_size:
                    sample_size += 1
                elif self.args.enable_amp and not sample_size % 8 == 0:
                    sample_size += 1
                else:
                    self.batch_sizes.append(sample_size)

                    sample_size = 1
                    max_batch_src_len = src_len
                    max_batch_tgt_len = tgt_len
                    while self.args.enable_amp and not max_batch_src_len % 8 == 0:  # for amp
                        max_batch_src_len += 1
                    while self.args.enable_amp and not max_batch_tgt_len % 8 == 0:  # for amp
                        max_batch_tgt_len += 1

            # lastly
            self.batch_sizes.append(sample_size)
            self.batch_sizes = np.array(self.batch_sizes)
            assert np.sum(self.batch_sizes) == self.data_size, \
                f"Size mismatch! Data size: {self.data_size}, sum batch sizes: {np.sum(self.batch_sizes)}"

            self.batch_ends = np.cumsum(self.batch_sizes)
            self.batch_starts = np.concatenate([[0], self.batch_ends[:-1]])

        else:
            raise ValueError(f"batch_type {batch_type} not supported!")

        log_rank_0(f"Done, time: {time.time() - start: .2f} s, total batches: {self.__len__()}")

    def __getitem__(self, index: int) -> G2SBatch:
        batch_index = index
        batch_start = self.batch_starts[batch_index]
        batch_end = self.batch_ends[batch_index]

        data_indices = self.data_indices[batch_start:batch_end]

        # collating, essentially
        # source (graph)
        graph_features = []
        a_lengths = []
        for data_index in data_indices:
            start, end = self.a_scopes_indices[data_index]
            a_scope = self.a_scopes[start:end]
            a_length = a_scope[-1][0] + a_scope[-1][1] - a_scope[0][0]

            start, end = self.b_scopes_indices[data_index]
            b_scope = self.b_scopes[start:end]

            start, end = self.a_features_indices[data_index]
            a_feature = self.a_features[start:end]
            a_graph = self.a_graphs[start:end]

            start, end = self.b_features_indices[data_index]
            b_feature = self.b_features[start:end]
            b_graph = self.b_graphs[start:end]

            graph_feature = (a_scope, b_scope, a_feature, b_feature, a_graph, b_graph)
            graph_features.append(graph_feature)
            a_lengths.append(a_length)

        fnode, fmess, agraph, bgraph, atom_scope, bond_scope = collate_graph_features(graph_features)

        # target (seq)
        tgt_token_ids = self.tgt_token_ids[data_indices]
        tgt_lengths = self.tgt_lens[data_indices]

        tgt_token_ids = tgt_token_ids[:, :max(tgt_lengths)]

        tgt_token_ids = torch.as_tensor(tgt_token_ids, dtype=torch.long)
        tgt_lengths = torch.tensor(tgt_lengths, dtype=torch.long)

        distances = None
        if self.args.compute_graph_distance:
            distances = collate_graph_distances(graph_features, a_lengths, self.distance_calculator)

        """
        logging.info("--------------------src_tokens--------------------")
        for data_index in data_indices:
            smi = "".join(self.vocab_tokens[src_token_id] for src_token_id in self.src_token_ids[data_index])
            logging.info(smi)
        logging.info("--------------------distances--------------------")
        logging.info(f"{distances}")
        exit(0)
        """

        g2s_batch = G2SBatch(
            fnode=fnode,
            fmess=fmess,
            agraph=agraph,
            bgraph=bgraph,
            atom_scope=atom_scope,
            bond_scope=bond_scope,
            tgt_token_ids=tgt_token_ids,
            tgt_lengths=tgt_lengths,
            distances=distances
        )
        # g2s_batch.log_tensor_shape()

        return g2s_batch

    def __len__(self):
        return len(self.batch_sizes)


def collate_graph_features(graph_features: List[Tuple], directed: bool = True
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                      List[np.ndarray], List[np.ndarray]]:
    if directed:
        padded_features = get_atom_features_sparse(Chem.Atom("*"))
        fnode = [np.array(padded_features)]
        fmess = [np.zeros(shape=[1, 2 + BOND_FDIM], dtype=np.int32)]
        agraph = [np.zeros(shape=[1, 11], dtype=np.int32)]
        bgraph = [np.zeros(shape=[1, 11], dtype=np.int32)]

        n_unique_bonds = 1
        edge_offset = 1

        atom_scope, bond_scope = [], []

        for bid, graph_feature in enumerate(graph_features):
            a_scope, b_scope, atom_features, bond_features, a_graph, b_graph = graph_feature

            a_scope = a_scope.copy()
            b_scope = b_scope.copy()
            atom_features = atom_features.copy()
            bond_features = bond_features.copy()
            a_graph = a_graph.copy()
            b_graph = b_graph.copy()

            atom_offset = len(fnode)
            bond_offset = n_unique_bonds
            n_unique_bonds += int(bond_features.shape[0] / 2)              # This should be correct?

            a_scope[:, 0] += atom_offset
            b_scope[:, 0] += bond_offset
            atom_scope.append(a_scope)
            bond_scope.append(b_scope)

            # node iteration is reduced to an extend
            fnode.extend(atom_features)

            # edge iteration is reduced to an append
            bond_features[:, :2] += atom_offset
            fmess.append(bond_features)

            a_graph += edge_offset
            a_graph[a_graph >= 999999999] = 0           # resetting padding edge to point towards edge 0
            agraph.append(a_graph)

            b_graph += edge_offset
            b_graph[b_graph >= 999999999] = 0           # resetting padding edge to point towards edge 0
            bgraph.append(b_graph)

            edge_offset += bond_features.shape[0]

        # densification
        fnode = np.stack(fnode, axis=0)
        fnode_one_hot = np.zeros([fnode.shape[0], sum(ATOM_FDIM)], dtype=np.float32)

        for i in range(len(ATOM_FDIM) - 1):
            fnode[:, i+1:] += ATOM_FDIM[i]          # cumsum, essentially

        for i, feat in enumerate(fnode):            # Looks vectorizable?
            # fnode_one_hot[i, feat[feat < 9999]] = 1
            fnode_one_hot[i, feat[feat < sum(ATOM_FDIM)]] = 1

        fnode = torch.as_tensor(fnode_one_hot, dtype=torch.float)
        fmess = torch.as_tensor(np.concatenate(fmess, axis=0), dtype=torch.float)

        agraph = np.concatenate(agraph, axis=0)
        column_idx = np.argwhere(np.all(agraph[..., :] == 0, axis=0))
        agraph = agraph[:, :column_idx[0, 0] + 1]       # drop trailing columns of 0, leaving only 1 last column of 0

        bgraph = np.concatenate(bgraph, axis=0)
        column_idx = np.argwhere(np.all(bgraph[..., :] == 0, axis=0))
        bgraph = bgraph[:, :column_idx[0, 0] + 1]       # drop trailing columns of 0, leaving only 1 last column of 0

        agraph = torch.as_tensor(agraph, dtype=torch.long)
        bgraph = torch.as_tensor(bgraph, dtype=torch.long)

    else:
        raise NotImplementedError

    return fnode, fmess, agraph, bgraph, atom_scope, bond_scope


def collate_graph_distances(graph_features: List[Tuple], a_lengths: List[int],
                            distance_calculator=None) -> torch.Tensor:
    max_len = max(a_lengths)

    distances = []
    for bid, (graph_feature, a_length) in enumerate(zip(graph_features, a_lengths)):
        _, _, _, bond_features, _, _ = graph_feature
        bond_features = bond_features.copy()

        # compute adjacency
        adjacency = np.zeros((a_length, a_length), dtype=np.bool_)
        for bond_feature in bond_features:
            u, v = bond_feature[:2]
            adjacency[u, v] = True

        distance = distance_calculator.calculate(adjacency, a_length, max_distance=a_length)

        # bucket
        distance[(distance > 8) & (distance < 15)] = 8
        distance[distance >= 15] = 9
        distance[distance == 0] = 10

        # reset diagonal
        np.fill_diagonal(distance, 0)

        """
        graph_distance = distance.copy()

        # compute graph distance
        distance = adjacency.copy()
        shortest_paths = adjacency.copy()
        path_length = 2
        stop_counter = 0
        non_zeros = 0

        while 0 in distance:
            shortest_paths = np.matmul(shortest_paths, adjacency)
            shortest_paths = path_length * (shortest_paths > 0)
            new_distance = distance + (distance == 0) * shortest_paths

            # if np.count_nonzero(new_distance) == np.count_nonzero(distance):
            if np.count_nonzero(new_distance) <= non_zeros:
                stop_counter += 1
            else:
                non_zeros = np.count_nonzero(new_distance)
                stop_counter = 0

            if args.task == "reaction_prediction" and stop_counter == 3:
                break

            distance = new_distance
            path_length += 1

        # bucket
        distance[(distance > 8) & (distance < 15)] = 8
        distance[distance >= 15] = 9
        distance[distance == 0] = 10            # different molecules

        # reset diagonal
        np.fill_diagonal(distance, 0)
        """

        # padding
        padded_distance = np.ones((max_len, max_len), dtype=np.uint8) * 11
        # padded_distance[:a_length, :a_length] = graph_distance
        padded_distance[:a_length, :a_length] = distance

        # assert np.array_equal(padded_distance[:a_length, :a_length], distance), \
        #     f"padded_distance:\n{padded_distance[:a_length, :a_length]}\n" \
        #     f"distance:\n{distance}"

        distances.append(padded_distance)

    distances = np.stack(distances)
    distances = torch.as_tensor(distances, dtype=torch.long)

    return distances
