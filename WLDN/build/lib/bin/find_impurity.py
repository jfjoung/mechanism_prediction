import os
import gzip
import argparse
from src.predictor import FWPredictor
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
from predict import beam_search
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator


def parse_arguments():
    parser = argparse.ArgumentParser(description='Command line forward predictor. Takes a single molecule and returns smiles')
    parser.add_argument('--model-dir', dest='model_dir', type=str, default='mech_pred')
    parser.add_argument('--model-name', dest='model_name', type=str, default='mech_pred')
    parser.add_argument('--hidden-bond-classifier', dest='hidden_bond_classifier', type=int, default=300,
                        help='Hidden size of the Dense layers of the bond classifier model')
    parser.add_argument('--hidden-cand-ranker', dest='hidden_cand_ranker', type=int, default=500,
                        help='Hidden size of the Dense layers of the cand ranker model')
    parser.add_argument('--depth', dest='depth', type=int, default=3)
    parser.add_argument('-s', '--smiles', dest='smiles', type=str, default=None)
    parser.add_argument('--file', dest='file', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    assert os.path.exists(args.model_dir), f'The specified model directory {args.model_dir} does not exist. Please use --model_dir flag to specify correct directory'
    predictor = FWPredictor(model_name=args.model_name, model_dir=args.model_dir, hidden_bond_classifier=args.hidden_bond_classifier,
                            hidden_candidate_ranker=args.hidden_cand_ranker, depth=args.depth)
    predictor.load_models()
    count = 0
    with open(args.file, 'r') as fid_in: #, open(args.file + '.result', 'w') as fid_out:
        num_of_find=0
        rank_list=[0, 0, 0, 0]
        for num, line in enumerate(fid_in):
            rsmi, psmi = line.strip().split(' ')
            if rsmi.endswith('.'):
                rsmi = rsmi.rstrip('.')
            result = beam_search(args, rsmi, predictor)
            true_set = set(psmi.split('.'))
            find=False
            prob=None
            rank=0
            for key, val in result.items():
                elem = [smi for smi in val['reaction'] if smi is not None]
                predict_set = set(elem[-1].split('.'))
                rank+=1
                if predict_set.issuperset(true_set):
                    find=True
                    num_of_find+=1
                    prob=key
                    rank_list[rank - 1]+=1
                    break
            if not find:
                rank=4
                rank_list[rank - 1] += 1
                prob = None

            if rank != 4:
                if len(result)>2:
                    reactions = [elem['reaction'] for elem in result.values()]
                    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=5)
                    compare_pair = [[reactions[0], reactions[1]], [reactions[0], reactions[2]],  [reactions[1], reactions[2]]]
                    for i, pair in enumerate(compare_pair):
                        reactant_0 = set(pair[0][0].split('.'))
                        set_product_0= set(pair[0][-1].split('.'))
                        set_product_1 = set(pair[1][-1].split('.'))
                        intersect = set_product_0 & set_product_1
                        product_0 = '.'.join([product for product in pair[0][-1].split('.') if product not in intersect])
                        product_1 = '.'.join([product for product in pair[1][-1].split('.') if product not in intersect])

                        intersect_r0 = set_product_0 & reactant_0
                        intersect_r1 = reactant_0 & set_product_1
                        reactant_00 = '.'.join([product for product in pair[0][0].split('.') if product not in intersect_r0])
                        reactant_01 = '.'.join([product for product in pair[0][0].split('.') if product not in intersect_r1])
                        product_00 = '.'.join([product for product in pair[0][-1].split('.') if product not in intersect_r0])
                        product_11 = '.'.join([product for product in pair[1][-1].split('.') if product not in intersect_r1])
                        mol_pair = [Chem.MolFromSmiles(reactant_00),Chem.MolFromSmiles(product_00), Chem.MolFromSmiles(product_0),
                                    Chem.MolFromSmiles(reactant_01),Chem.MolFromSmiles(product_11), Chem.MolFromSmiles(product_1)]
                        fps = [fpgen.GetSparseCountFingerprint(x) for x in mol_pair]

                        # print('reactant & product 0 ',DataStructs.TanimotoSimilarity(fps[0], fps[1]))
                        # print('reactant & product 1 ', DataStructs.TanimotoSimilarity(fps[3], fps[4]))
                        # print('product0 & product 1 ', DataStructs.TanimotoSimilarity(fps[2], fps[5]))

                        sim = DataStructs.TanimotoSimilarity(fps[2], fps[5])

                        if 0.5 > sim > 0.0:
                            print(result)
                            print(i,sim)
                            count +=1
                            break
            if count == 100:
                break

            # print(result)
            # fid_out.write('{} {} {} {}\n'.format(rsmi, psmi, rank, prob))
            # print('top 1: {} ({:.0f}%), top 2: {} ({:.0f}%), top 3: {} ({:.0f}%), and {} ({:.0f}%) products are not found out of {}'.format(rank_list[0],rank_list[0]/(num+1)*100, rank_list[1],rank_list[1]/(num+1)*100, rank_list[2],rank_list[2]/(num+1)*100, rank_list[3],rank_list[3]/(num+1)*100, num+1))
            # break
    # print('-'*50)
    # print('Prediction is over')
    # print('Final performance is ')
    # print('top 1: {}, top 2: {}, top 3: {}, and {} products are not found out of {}'.format(rank_list[0], rank_list[1], rank_list[2],
    #                                                                                         rank_list[3], num + 1))
