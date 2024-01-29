import os
import gzip
import argparse
from src.predictor import FWPredictor

def parse_arguments():
    parser = argparse.ArgumentParser(description='Command line method for getting the models trainable parameters')
    parser.add_argument('--model-dir', dest='model_dir', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models'))
    parser.add_argument('--model-name', dest='model_name', type=str, default='mech_pred')
    parser.add_argument('--hidden', dest='hidden', type=int, default=300)
    parser.add_argument('--depth', dest='depth', type=int, default=3)
    return parser.parse_args()

if __name__ == '__main__':
    '''
    eg command line run
    python get_model_params.py --model_name uspto_500k

    '''
    args = parse_arguments()
    assert os.path.exists(args.model_dir), f'The specified model directory {args.model_dir} does not exist. Please use --model_dir flag to specify correct directory'

    predictor = FWPredictor(model_name=args.model_name, model_dir=args.model_dir, hidden=args.hidden,
                            depth=args.depth)
    predictor.load_models()

    print('Classifier summary')
    predictor.core_model.summary()
    print('=' * 100)
    predictor.diff_model.summary()
