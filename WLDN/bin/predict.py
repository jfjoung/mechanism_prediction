import os
import gzip
import argparse
from src.predictor import FWPredictor
import copy
from rdkit import Chem

def parse_arguments():
    parser = argparse.ArgumentParser(description='Command line forward predictor. Takes a single molecule and returns smiles')
    parser.add_argument('--model-dir', dest='model_dir', type=str, default='models')
    parser.add_argument('--model-name', dest='model_name', type=str, default='mech_pred')
    parser.add_argument('--hidden-bond-classifier', dest='hidden_bond_classifier', type=int, default=300,
                        help='Hidden size of the Dense layers of the bond classifier model')
    parser.add_argument('--hidden-cand-ranker', dest='hidden_cand_ranker', type=int, default=500,
                        help='Hidden size of the Dense layers of the cand ranker model')
    parser.add_argument('--depth', dest='depth', type=int, default=3)
    parser.add_argument('-s', '--smiles', dest='smiles', type=str, default=None)
    return parser.parse_args()

def adjust_probability(smiles_list, exclude_smiles):
    modified_list = copy.deepcopy(smiles_list)
    modified_list = [item for item in modified_list if item['smiles'] != exclude_smiles]
    total_prob = sum(item['prob'] for item in modified_list)
    for item in modified_list:
        item['prob'] /= total_prob

    return modified_list


def beam_search(args, smiles, predictor, beam_width=3, max_steps=7, max_iter=100):
    smol = Chem.MolFromSmiles(smiles, sanitize=False)
    for atom in smol.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(smol, isomericSmiles=False)
    rxn_dict = {1: {'reaction': [smiles], 'prob': [1], 'predicted_edits': [[]], 'score': [0], 'rank': [1]}}
    end_rxn_dict = {}

    iter=0
    # print('iter 0, rxn_dict is ', rxn_dict, '\n')
    while len(end_rxn_dict) < beam_width:
        iter+=1
        sorted_items = sorted(rxn_dict.items(), reverse=True)
        excutable_dict = dict(sorted_items[:beam_width])
        new_rxn_dict = {}
        for key, val in excutable_dict.items():
            if len(val['reaction']) == 1:    # At start of reaction, something should happen.
                res = predictor.predict_single(val['reaction'][-1])
                res = adjust_probability(res, smiles)
            elif len(val['reaction']) > max_steps: # If the reaction exceeds the max step, delete that node.
                del rxn_dict[key]
                continue
            elif val['predicted_edits'][-1] == []: # If the reaction is over, do not proceed to predict.
                continue
            else:
                res = predictor.predict_single(val['reaction'][-1])
                for r in res:

                    if r['smiles'] in val['reaction'][:-1]:  # If the same molecules are re-produced, delete it.
                        removing_smi=r['smiles']
                        res=adjust_probability(res, removing_smi)
                    elif val['reaction'][-1] == r['smiles']:
                        found = False
                        for edit_sets in val['predicted_edits']:
                            for edit in edit_sets:
                                if 'b' in edit[3]:
                                    found = True
                        if not found: # If the route consists of only protonation and deprotonation, delete it.
                            removing_smi = r['smiles']
                            res = adjust_probability(res, removing_smi)
            for r in res:
                new_reaction = val['reaction'] + [r['smiles']]
                new_rank = val['rank'] + [r['rank']]
                new_prob = val['prob'] + [r['prob']]
                new_predicted_edits = val['predicted_edits'] + [r['predicted_edits']]
                new_score = val['score'] + [r['score']]
                new_rxn_dict[key * r['prob']] = {'reaction': new_reaction,
                                                 'prob': new_prob,
                                                 'predicted_edits': new_predicted_edits,
                                                 'score': new_score,
                                                 'rank': new_rank}
            del rxn_dict[key]
        rxn_dict.update(new_rxn_dict)
        sorted_items = sorted(rxn_dict.items(), reverse=True)
        rxn_dict = dict(sorted_items)
        cheching_end_dict = dict(sorted_items[:beam_width])
        for key, val in cheching_end_dict.items():
            if val['reaction'][-1] == val['reaction'][-2]:
                end_rxn_dict[key]=val
        if iter == max_iter:
            break
    sorted_items = sorted(end_rxn_dict.items(), reverse=True)
    end_rxn_dict = dict(sorted_items[:beam_width])
    return end_rxn_dict


if __name__ == '__main__':
    '''
    eg command line run
    python predict.py --model_name uspto_500k --smiles "CCCCCCN.O=C(O)c1cccc(CBr)c1"

    '''
    args = parse_arguments()
    assert os.path.exists(args.model_dir), f'The specified model directory {args.model_dir} does not exist. Please use --model_dir flag to specify correct directory'

    predictor = FWPredictor(model_name=args.model_name, model_dir=args.model_dir, hidden_bond_classifier=args.hidden_bond_classifier,
                            hidden_candidate_ranker=args.hidden_cand_ranker, depth=args.depth)
    predictor.load_models()
    print(beam_search(args, args.smiles, predictor))
