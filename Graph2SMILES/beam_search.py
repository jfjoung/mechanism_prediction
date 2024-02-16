import argparse
import csv
import glob
import numpy as np
import time
import torch
import zipfile
from multiprocessing import Pool
from typing import Any, Dict, List
from utils import parsing
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from tqdm import tqdm
import tree_builder_ver2
import pickle
from utils.data_utils import canonicalize_smiles

def get_beam_search_parser():
    parser = argparse.ArgumentParser("beam search")
    parsing.add_common_args(parser)
    parsing.add_preprocess_args(parser)
    parsing.add_train_args(parser)
    parsing.add_predict_args(parser)

    return parser

class G2SHandler:
    """Graph2SMILES Handler for torchserve"""

    def __init__(self):
        self._context = None
        self.manifest = None
        self.initialized = False

        self.args = args
        self.device = None
        self.model = None
        self.vocab = None
        self.vocab_tokens = None
        self.distance_calculator = None
        self.p = Pool(args.num_cores)

    def overwrite_default_args(self):
        self.args.load_from = "model.850000_84.pt"
        self.args.vocab_file = "vocab.txt"
        # self.args.mask_rel_chirality = 1
        self.args.mask_rel_chirality = 0
        self.args.predict_batch_size = 16384
        self.args.beam_size = 30
        self.args.n_best = 30
        self.args.temperature = 1.0
        self.args.predict_min_len = 1
        self.args.predict_max_len = 512
        self.args.device = self.device

    def initialize(self): #, context):
        # self._context = context
        # self.manifest = context.manifest
        #
        # properties = context.system_properties

        # print(self.args.model_path)

        model_dir = self.args.model_path
        # print(glob.glob(f"{model_dir}/*"))
        # self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        #
        # with zipfile.ZipFile(model_dir + '/models.zip', 'r') as zip_ref:
        #     zip_ref.extractall(model_dir)
        # with zipfile.ZipFile(model_dir + '/utils.zip', 'r') as zip_ref:
        #     zip_ref.extractall(model_dir)

        from train import get_model
        from models.graph2smiles import Graph2SMILES
        from predict import get_predict_parser
        from utils import parsing
        from utils.ctypes_calculator import DistanceCalculator
        from utils.data_utils import load_vocab

        predict_parser = get_predict_parser()
        self.args, _ = predict_parser.parse_known_args()
        self.overwrite_default_args()

        checkpoint = model_dir + "/model.850000_84.pt"
        state = torch.load(checkpoint, map_location=self.device)
        pretrain_args = state["args"]
        pretrain_state_dict = state["state_dict"]

        # override
        pretrain_args.processed_data_path=self.args.model_path
        pretrain_args.model_path = self.args.model_path
        pretrain_args.load_from =checkpoint

        # for backward compatibility
        if not hasattr(pretrain_args, "shared_attention_layer"):
            pretrain_args.shared_attention_layer = 0
        if not hasattr(pretrain_args, "n_latent"):
            pretrain_args.n_latent = 1
        # pretrain_args.rel_pos_buckets -= 1
        pretrain_args.local_rank = -1
        # parsing.log_args(pretrain_args, phase="pretrained")

        model_class = Graph2SMILES

        self.args.processed_data_path = model_dir
        self.vocab = load_vocab(self.args)
        self.vocab_tokens = [k for k, v in sorted(self.vocab.items(), key=lambda tup: tup[1])]


        model, state = get_model(pretrain_args, model_class, self.vocab, self.device)
        # logging.info(model)
        if hasattr(model, "module"):
            model = model.module  # unwrap DDP model to enable accessing model func directly

        pretrain_state_dict = {k.replace("module.", ""): v for k, v in pretrain_state_dict.items()}
        model.load_state_dict(pretrain_state_dict)
        # print(f"Loaded pretrained state_dict from {checkpoint}")
        model.eval()
        self.model = model

        self.distance_calculator = DistanceCalculator()
        self.p = Pool()

        self.initialized = True

    def preprocess(self, data: List):
        from utils.data_utils import canonicalize_smiles, collate_graph_distances, \
            collate_graph_features, G2SBatch, len2idx
        from utils.preprocess_utils import get_graph_features_from_smi

        canonical_smiles = [canonicalize_smiles(smi) for smi in data]

        # ----------<adapted from preprocess.py>----------
        start = time.time()
        graph_features_and_lengths = self.p.imap(
            get_graph_features_from_smi, enumerate(canonical_smiles)
        )
        graph_features_and_lengths = list(graph_features_and_lengths)
        # print(f"Done graph featurization, time: {time.time() - start}. Collating and saving...")
        a_scopes, a_scopes_lens, b_scopes, b_scopes_lens, a_features, a_features_lens, \
            b_features, b_features_lens, a_graphs, b_graphs = zip(*graph_features_and_lengths)

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
        # ----------</adapted from preprocess.py>----------

        # ----------<adapted from data_utils.G2SDataset.__init__()>----------
        if self.args.mask_rel_chirality == 1:
            a_features[:, 6] = 2

        a_scopes_indices = len2idx(a_scopes_lens)
        b_scopes_indices = len2idx(b_scopes_lens)
        a_features_indices = len2idx(a_features_lens)
        b_features_indices = len2idx(b_features_lens)

        del a_scopes_lens, b_scopes_lens, a_features_lens, b_features_lens
        # ----------</adapted from data_utils.G2SDataset.__init__()>----------

        # ----------<adapted from data_utils.G2SDataset.batch()>----------
        # batching takes trivial time
        batch_sizes = []

        sample_size = 0
        max_batch_src_len = 0

        for smi in canonical_smiles:
            src_len = len(smi)
            max_batch_src_len = max(src_len, max_batch_src_len)
            if max_batch_src_len * (sample_size + 1) <= self.args.predict_batch_size:
                sample_size += 1
            else:
                batch_sizes.append(sample_size)
                sample_size = 1
                max_batch_src_len = src_len

        # lastly
        batch_sizes.append(sample_size)
        batch_sizes = np.array(batch_sizes)
        assert np.sum(batch_sizes) == len(canonical_smiles), \
            f"Size mismatch! Data size: {len(canonical_smiles)}, sum batch sizes: {np.sum(batch_sizes)}"

        batch_ends = np.cumsum(batch_sizes)
        batch_starts = np.concatenate([[0], batch_ends[:-1]])
        # ----------</adapted from data_utils.G2SDataset.batch()>----------

        # ----------<adapted from data_utils.G2SDataset.__getitem__()>----------
        g2s_batches = []
        for batch_start, batch_end in zip(batch_starts, batch_ends):
            data_indices = np.arange(batch_start, batch_end)
            graph_features = []
            a_lengths = []
            for data_index in data_indices:
                start, end = a_scopes_indices[data_index]
                a_scope = a_scopes[start:end]
                a_length = a_scope[-1][0] + a_scope[-1][1] - a_scope[0][0]

                start, end = b_scopes_indices[data_index]
                b_scope = b_scopes[start:end]

                start, end = a_features_indices[data_index]
                a_feature = a_features[start:end]
                a_graph = a_graphs[start:end]

                start, end = b_features_indices[data_index]
                b_feature = b_features[start:end]
                b_graph = b_graphs[start:end]

                graph_feature = (a_scope, b_scope, a_feature, b_feature, a_graph, b_graph)
                graph_features.append(graph_feature)
                a_lengths.append(a_length)

            fnode, fmess, agraph, bgraph, atom_scope, bond_scope = collate_graph_features(graph_features)
            distances = collate_graph_distances(graph_features, a_lengths, self.distance_calculator)

            g2s_batch = G2SBatch(
                fnode=fnode,
                fmess=fmess,
                agraph=agraph,
                bgraph=bgraph,
                atom_scope=atom_scope,
                bond_scope=bond_scope,
                tgt_token_ids=torch.tensor([0]),
                tgt_lengths=torch.tensor(a_lengths),
                distances=distances
            )
            g2s_batches.append(g2s_batch)
        # ----------</adapted from data_utils.G2SDataset.__getitem__()>----------

        return g2s_batches

    def inference(self, data: List) -> List[Dict[str, Any]]:
        # adapted from predict.get_predictions()
        from utils.data_utils import canonicalize_smiles

        results = []

        with torch.no_grad():
            for test_idx, test_batch in enumerate(data):
                test_batch.to(self.device)
                batch_predictions = self.model.predict_step(
                    reaction_batch=test_batch,
                    batch_size=test_batch.size,
                    beam_size=self.args.beam_size,
                    n_best=self.args.n_best,
                    temperature=self.args.temperature,
                    min_length=self.args.predict_min_len,
                    max_length=self.args.predict_max_len
                )

                for predictions, scores in zip(batch_predictions["predictions"], batch_predictions["scores"]):
                    valid_products = []
                    valid_scores = []
                    for prediction, score in zip(predictions, scores):
                        predicted_idx = prediction.detach().cpu().numpy()
                        score = score.item()
                        predicted_tokens = [self.vocab_tokens[idx] for idx in predicted_idx[:-1]]
                        smi = "".join(predicted_tokens)
                        smi = canonicalize_smiles(smi, trim=False, suppress_warning=True)

                        if not smi or smi in valid_products:
                            continue
                        else:
                            valid_products.append(smi)
                            valid_scores.append(score)

                    result = {
                        "products": valid_products,
                        "scores": valid_scores
                    }
                    results.append(result)

        return results

    def postprocess(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        return [data]

    # def handle(self, data, context) -> List[List[Dict[str, Any]]]:
    #     self._context = context
    #     # self.initialize()
    #     output = self.preprocess(data)
    #     output = self.inference(output)
    #     output = self.postprocess(output)
    #     # print(output)
    #
    #     return output

    def handle(self, data) -> List[List[Dict[str, Any]]]:
        output = self.preprocess(data)
        output = self.inference(output)
        output = self.postprocess(output)
        return output

global G_predictions

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
    # reactants, reagent, gt = test_row["unmapped_rxn"].strip().split(">")
    # k = canonicalize_smiles(reactants)
    k = reactants

    if k not in predictions:
        print(f"Product {reactants} not found in predictions (after canonicalization), skipping")
        return accuracy

    gt = canonicalize_smiles(gt)


    for j, prediction in enumerate(predictions[k]):
        prod_set = set(prediction.split('.'))
        # print(prod_set)
        if gt in prod_set:
            accuracy[j:] = 1.0
            break

    return accuracy


def score(args):
    global G_predictions
    n_best = 3

    # logging.info(f"Scoring predictions with model: {self.model_name}")
    #
    # # Load predictions and transform into a huge table {cano_prod: [cano_cand, ...]}
    # logging.info(f"Loading predictions from {self.output_file}")
    predictions = {}
    p = Pool(args.num_cores)

    saving_file = args.test_file[:-4] + '.pickle'

    with open(saving_file, 'rb') as file:
        prediction_reader = pickle.load(file)
        for result in tqdm(p.imap(pickle2kv,
                                  (prediction_row for prediction_row in prediction_reader))):
            k, v = result
            predictions[k] = v
    G_predictions = predictions
    p.close()
    p.join()
    p = Pool(args.num_cores)  # re-initialize to see the global variable

    # Results matching
    print(f"Matching against ground truth from {args.test_file}")
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
        print(f"Top {n + 1} accuracy: {mean_accuracies[n]}")


if __name__ == "__main__":
    # initialization ---------------------------------- args, logs and devices
    predict_parser = get_beam_search_parser()
    args = predict_parser.parse_args()
    handler = G2SHandler()
    handler.initialize()
    k = 3
    depth = 8

    # args.test_file.split(".")[0]
    saving_file = args.test_file.split(".")[0].split('/')[-1]+'.pickle'
    saving_path = args.test_output_path

    results = []
    for phase, fn in [("test", args.test_file)]:
        with open(fn, "r") as f, open(saving_path+'/'+saving_file, "wb") as f_out:
            csv_reader = csv.DictReader(f)
            for line in tqdm(csv_reader):
                rsmi, psmi = line['rxn_smiles'].split('>>')

                starting_node = tree_builder_ver2.Node([1, rsmi])
                terminated_node, continue_node = starting_node.get_topK_leafs(k)
                iter=0
                while len(terminated_node) <k:
                    smi=[node.plain_smiles for node in continue_node]
                    try:
                        outputs = handler.handle(smi)
                    except:
                        break

                    outputs = outputs[0]
                    for output, node in zip(outputs, continue_node):
                        child=[[sc, pred] for sc, pred in zip(output['scores'], output['products'])]
                        node.add(child)

                    terminated_node, continue_node = starting_node.get_topK_leafs(k)
                    depths = [node.depth for node in continue_node]
                    if any(x > depth for x in depths):
                        break
                    if iter>50:
                        break
                    iter+=1

                if not terminated_node:
                    terminated_node = [starting_node, starting_node, starting_node]

                pathways=[node for node in terminated_node]
                results.append(pathways)


            pickle.dump(results, f_out)
    score(args)
