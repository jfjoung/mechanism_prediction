import os
import gzip
import argparse
from src.predictor import FWPredictor
import copy
from rdkit import Chem
from predict import beam_search
import os

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
    parser.add_argument('--dir', dest='dir', type=str, default=None)
    return parser.parse_args()



def get_files_with_extension(directory, extension):
    file_list = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            file_path = os.path.join(directory, filename)
            file_list.append((filename, os.path.getsize(file_path)))
    return file_list


if __name__ == '__main__':

    args = parse_arguments()
    assert os.path.exists(args.model_dir), f'The specified model directory {args.model_dir} does not exist. Please use --model_dir flag to specify correct directory'
    predictor = FWPredictor(model_name=args.model_name, model_dir=args.model_dir, hidden_bond_classifier=args.hidden_bond_classifier,
                            hidden_candidate_ranker=args.hidden_cand_ranker, depth=args.depth)
    predictor.load_models()
    #
    proc_files = get_files_with_extension(args.dir, ".proc")
    sorted_proc_files = sorted(proc_files, key=lambda x: x[1], reverse=True)
    # print(sorted_proc_files[:200])

    sorted_proc_files=sorted_proc_files[65:]
    for file, size in sorted_proc_files:
        with open(args.dir+'/'+file, 'r') as fid_in, open(args.dir+'/'+file + '.result', 'w') as fid_out:
            num_of_find=0
            rank_list=[0, 0, 0, 0]
            for num, line in enumerate(fid_in):
                try:
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
                    fid_out.write('{} {} {} {}\n'.format(rsmi, psmi, rank, prob))
                    print('{}, top 1: {} ({:.0f}%), top 2: {} ({:.0f}%), top 3: {} ({:.0f}%), and failed: {} ({:.0f}%) out of {}'.format(file, rank_list[0],rank_list[0]/(num+1)*100, rank_list[1],rank_list[1]/(num+1)*100, rank_list[2],rank_list[2]/(num+1)*100, rank_list[3],rank_list[3]/(num+1)*100, num+1))
                except:
                    predictor.load_models()

        print('-'*50)
        print('Prediction is over')
        print('Final performance of {} is '.format(file))
        print('top 1: {}, top 2: {}, top 3: {}, and failed: {} out of {}'.format(rank_list[0], rank_list[1], rank_list[2],
                                                                                                rank_list[3], num + 1))
        with open(args.dir + '/result/{}.txt'.format(file), 'a') as fid_result:
            fid_result.write('total_reaction: {}, top 1: {} ({:.0f}%), top 2: {} ({:.0f}%), top 3: {} ({:.0f}%), and failed: {} ({:.0f}%)'.format(num+1,
                                                                                                                                                rank_list[0],rank_list[0]/(num+1)*100,
                                                                                                                                                rank_list[1],rank_list[1]/(num+1)*100,
                                                                                                                                                rank_list[2],rank_list[2]/(num+1)*100,
                                                                                                                                                rank_list[3],rank_list[3]/(num+1)*100))
        print('Writing was successful')

