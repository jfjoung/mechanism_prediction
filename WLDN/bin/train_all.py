import os
import sys
import time
import argparse
import pandas as pd
from src.train_classifier_core import train_wln_core
from src.gen_cand_score import gen_cands_detailed
from src.train_wldiff import wln_diffnet

"""
example commands:
python bin/train_all.py --cutoff -1 --nproc 5 --hidden 50 \
--train data/train_trunc.txt.proc --valid data/valid_trunc.txt.proc \
 --test data/test_trunc.txt.proc --model-name testing
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the forward reaction predictor for integration with ASKCOS')
    parser.add_argument('--nproc', dest='nproc', type=int, default=1,
                        help='Number of Processors to use for generation of input graphs')
    parser.add_argument('--model-dir', dest='model_dir', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models'),
                        help='Directory that the model will be saved in. If using Docker/Singularity be sure this is in the mounted volume')
    parser.add_argument('--train', dest='train', type=str, default='../data/train.txt.proc',
                        help='Path to the training data')
    parser.add_argument('--valid', dest='valid', type=str, default='../data/valid.txt.proc',
                        help='Path to the validation data')
    parser.add_argument('--test', dest='test', type=str, default='../data/test.txt.proc',
                        help='Path to the test data')
    parser.add_argument('--reagents', dest='reagents', action='store_true',
                        help='To use reagents (ie molecules that do not contribute atoms to product) during training')
    parser.add_argument('--model-name', dest='model_name', type=str, default='mech_pred',
                        help='Name of the model.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=10,
                        help='Size of batches for training the WLN_Core model')
    parser.add_argument('--hidden-bond-classifier', dest='hidden_bond_classifier', type=int, default=300,
                        help='Hidden size of the Dense layers of the bond classifier model')
    parser.add_argument('--hidden-cand-ranker', dest='hidden_cand_ranker', type=int, default=500,
                        help='Hidden size of the Dense layers of the cand ranker model')
    parser.add_argument('--depth', dest='depth', type=int, default=3,
                        help='Depth of the graph convolutions, similar to fingerprint radius')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10,
                        help='Number of epochs to train. Early stopping is enabled so use the "--early-stopping flag to modify the patience')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--early-stopping', dest='early_stopping', type=int, default=2,
                        help='How many epochs to run until there is no more progress in minimizing loss (only the best epoch is saved)')
    parser.add_argument('--clipnorm' , dest='clipnorm', type=float, default=5.0,
                        help='Gradient clipping value')
    parser.add_argument('--cutoff' , dest='cutoff', type=float, default=0.6,
                        help='cutoff for accuracy before training candidate ranker model')
    parser.add_argument('--resume', dest='resume', type=int, default=0,
                        help='Resume on step. 0=train all (default), 1=start from core model training,'
                        '2=start from candidate products generation, 3=start from candidate ranker model')
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
    print(args)
    t0 = time.time()

    if args.resume in range(0,2):
        t0 = print_time('step 1 of 3', t0)
        t0 = print_time('Training the core predictor', t0)
        model = train_wln_core(train=args.train, valid=args.valid, reagents=args.reagents, model_dir=args.model_dir, model_name=args.model_name,
                        hidden=args.hidden_bond_classifier, learning_rate=args.learning_rate, early_stopping=args.early_stopping, epochs=args.epochs,
                        clipnorm=args.clipnorm, workers=args.nproc, batch_size=args.batch_size)
        t0 = print_time('FINISHED TRAINING THE CORE PREDICTOR', t0)
    if args.resume in range(0,3):
        t0 = print_time('step 2 of 3', t0)
        t0 = print_time('Evaluating and generating detailed predictions', t0)
        gen_cands_detailed(model_name=args.model_name, model_dir=args.model_dir, train=args.train, valid=args.valid,
                        test=args.test, cutoff=args.cutoff, workers=args.nproc, reagents=args.reagents, batch_size=args.batch_size)
        t0 = print_time('FINISHED GENERATING DETAILED PREDICTIONS', t0)
    if args.resume in range(0,4):
        t0 = print_time('step 3 of 3', t0)
        t0 = print_time('Training of the candidate ranker', t0)
        #INFO batch-size has to = 1 for the diffnet (default kwarg in 'wln_diffnet')
        wln_diffnet(train=args.train, valid=args.valid, model_dir=args.model_dir, model_name=args.model_name,
                    test=args.test, hidden=args.hidden_cand_ranker, learning_rate=args.learning_rate, early_stopping=args.early_stopping,
                    clipnorm=args.clipnorm, epochs=args.epochs, workers=args.nproc)

        t0 = print_time('FINISHED TRAINING THE CANDIDATE RANKER', t0)

    #There is a bug in multiprocessing where the script does not exit after finishing. Does
    # not affect the training but in a HPC environment it would have to wait till timeout before
    # the job is killed
    sys.exit()
