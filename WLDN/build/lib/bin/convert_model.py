"""
Script to convert the model from the keras form to the tensorflow saved model format for
production using tensorflow serving.
"""
import argparse
import os
from src.predictor import FWPredictor


def parse_arguments():
    parser = argparse.ArgumentParser(description='Command line forward predictor. Takes a single molecule and returns smiles')
    parser.add_argument('--model-dir', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models'))
    parser.add_argument('--model-name', type=str, default='wln_fw_pred')
    parser.add_argument('--output-dir', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models'))
    parser.add_argument('--hidden-bond-classifier', type=int, default=300)
    parser.add_argument('--hidden-candidate-ranker', type=int, default=500)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--cutoff', type=int, default=1500)
    parser.add_argument('--output-dim', type=int, default=5)
    parser.add_argument('--core-size', type=int, default=16)

    return parser.parse_args()


if __name__ == '__main__':
    """
    eg command line run
    python save_model.py --model_name uspto_500k
    """
    args = parse_arguments()
    assert os.path.exists(args.model_dir), f'The specified model directory {args.model_dir} does not exist. Please use --model_dir flag to specify correct directory'

    predictor = FWPredictor(
        model_name=args.model_name,
        model_dir=args.model_dir,
        hidden_bond_classifier=args.hidden_bond_classifier,
        hidden_candidate_ranker=args.hidden_candidate_ranker,
        depth=args.depth,
        cutoff=args.cutoff,
        output_dim=args.output_dim,
        core_size=args.core_size
    )
    predictor.load_models()
    predictor.save_model(args.output_dir)
