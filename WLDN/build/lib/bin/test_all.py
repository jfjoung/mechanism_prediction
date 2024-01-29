import os
import sys
import time
import argparse
import pandas as pd
from src.test_models import gen_cands_detailed_testing, test_wln_diffnet
from src.eval_by_smiles import eval_by_smiles

"""
example commands:
python bin/test_all.py --nproc 5 --test data/test_trunc.txt.proc --model-dir testing --model-name testing --hidden-cand-ranker 500
--depth 3 --batch-size 10
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test the forward reaction predictor for integration with ASKCOS')
    parser.add_argument('--model-dir', dest='model_dir', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models'),
                        help='Directory that the model will be loaded from')
    parser.add_argument('--model-name', dest='model_name', type=str, default='wln_fw_pred',
                        help='Name of the model.')
    parser.add_argument('--test', dest='test', type=str, default='../data/test.txt.proc',
                        help='Path to the test data')
    parser.add_argument('--reagents', dest='reagents', action='store_true',
                        help='To use reagents (ie molecules that do not contribute atoms to product) during training')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=10,
                        help='Size of batches for testing the WLN_Core model')
    parser.add_argument('--resume', dest='resume', type=int, default=0,
                        help='Resume on step. 0=test all, 1=start from reactive bond prediction,'
                        '2=start from candidate products construction, 3=only evaluate ranked candidates by smiles')
    return parser.parse_args()


def print_time(task_name, t0):
    new_t0 = time.time()
    print('')
    print('*' * 100)
    print(f'{task_name}: {new_t0-t0:.2f} seconds elapsed')
    print('*' * 100)
    print('')
    return new_t0


if __name__ == '__main__':
    args = parse_arguments()
    t0 = time.time()

    if args.resume in range(-1,2):
        t0 = print_time('Predicting reactive bonds, step 1 of 3', t0)
        gen_cands_detailed_testing(model_name=args.model_name, model_dir=args.model_dir, test=args.test,
                            batch_size=args.batch_size, reagents=args.reagents)
    if args.resume in range(-1,3):
        t0 = print_time('Constructing and ranking candidate products, step 2 of 3', t0)
        test_wln_diffnet(batch_size=1, model_name=args.model_name, model_dir=args.model_dir,
                     test=args.test)
    if args.resume in range(-1,4):
        t0 = print_time('Evaluating ranked candidates by smiles, step 3 of 3'   , t0)
        eval_by_smiles(pred_path=f'{args.model_dir}/test_{args.model_name}.predictions.txt', gold_path=args.test)

    t0 = print_time('FINISHED TESTING ALL OF THE MODULES', t0)

    sys.exit()
