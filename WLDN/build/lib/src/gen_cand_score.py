import json
import math

import tensorflow as tf
from tensorflow.keras import backend as K

from WLN.data_loading import Graph_DataLoader
from WLN.metrics import wln_loss, top_10_acc, top_20_acc, top_100_acc
from WLN.models import WLNPairwiseAtomClassifier
from graph_utils.ioutils_direct import INVALID_BOND, bindex_to_o,hindex_to_o,fcindex_to_o, nbos, reactant_tracking, n_hydro, n_formal

'''
 Run test set then output candidates.
'''

def gen_core_preds(model, filename, data_gen, nk=80):
    """
    Generates candidates for train/valid/test (ie batch)
    """

    print(f'Generating candidates for {filename}')

    with open(filename, 'w') as f:
        for it, batch in enumerate(data_gen):
            graph, bond_labels, sp_labels, all_ratoms, all_rbonds, rxn_smiles, labels = batch
            scores = model.predict_on_batch(graph)
            bmask = K.cast(K.equal(bond_labels, INVALID_BOND), dtype='float32') * 10000
            topk_scores, topk = tf.math.top_k(scores - bmask,  k=nk)
            topk = K.eval(topk)
            total_dim = bond_labels.shape[1] #dynamic depending on number of atoms in molecules
            num_atom = int((-(n_hydro+n_formal) + math.sqrt((n_hydro+n_formal)**2 - 4*nbos*(-total_dim)))/(2*nbos))
            bond_end = num_atom * num_atom * nbos
            H_end = num_atom * n_hydro
            FC_end = num_atom * n_formal

            #do not use batch_size because not all batches will be same size if there is not an even number of training examples
            for i in range(len(labels)):
                ratoms = all_ratoms[i]
                rbonds = all_rbonds[i]
                f.write(f'{rxn_smiles[i]} {labels[i]} ')
                for j in range(nk):
                    k = topk[i,j] # index that must be converted to (x, y, t) tuple
                    if k < bond_end:
                        bindex = k % nbos
                        y = ((k - bindex) / nbos) % num_atom + 1
                        x = (k - bindex - (y - 1) * nbos) / num_atom / nbos + 1
                        bo = bindex_to_o[bindex]
                        if x < y and x in ratoms and y in ratoms and (x, y, bo) not in rbonds:
                            f.write(f'B {x}-{y}-{bo:.1f} ')
                            f.write(f'{topk_scores[i,j]} ')

                    elif k >= bond_end and k < bond_end + H_end:
                        k -= bond_end
                        H_index = k % n_hydro
                        x = int((k - H_index) / n_hydro + 1)
                        h_o = hindex_to_o[H_index]
                        if x in ratoms:
                            f.write(f'H {x}:{h_o:.1f} ')
                            f.write(f'{topk_scores[i,j]} ')

                    elif k >= bond_end + H_end:
                        k -= (bond_end + H_end)
                        FC_index = k % n_formal
                        x = int((k - FC_index) / n_formal + 1)
                        fc_o = fcindex_to_o[FC_index]
                        if x in ratoms:
                            f.write(f'F {x}:{fc_o:.1f} ')
                            f.write(f'{topk_scores[i,j]} ')

            # original codes
                #     bindex = k % nbos
                #     y = ((k - bindex) / nbos) % cur_dim + 1
                #     x = (k - bindex - (y-1) * nbos) / cur_dim / nbos + 1
                #     bo = bindex_to_o[bindex]
                #     # Only allow atoms from reacting molecules to be part of the prediction,
                #     if x < y and x in ratoms and y in ratoms and (x, y, bo) not in rbonds:
                #         f.write(f'{x}-{y}-{bo:.1f} ')
                #         f.write(f'{topk_scores[i,j]} ')
                f.write('\n') # new l

    return True

def gen_cand_single(scores, nk=80, smiles=None, reagents=False):
    """
    Generates candidates for inference (ie single molecule)
    """
    # print('scores ', len(scores))
    topk_scores, topk = tf.nn.top_k(scores, k=nk)
    # print('topk ', len(topk))
    topk = topk.numpy()
    # print('topk ', len(topk))

    total_dim = len(scores)  # dynamic depending on number of atoms in molecules
    num_atom = int((-(n_hydro + n_formal) + math.sqrt((n_hydro + n_formal) ** 2 - 4 * nbos * (-total_dim))) / (2 * nbos))
    bond_end = num_atom * num_atom * nbos
    H_end = num_atom * n_hydro
    FC_end = num_atom * n_formal

    cand_bonds = []
    cand_H=[]
    cand_FC=[]

    ratoms = None

    for j in range(nk):
        k = topk[j]

        if k < bond_end:
            bindex = k % nbos
            y = ((k - bindex) / nbos) % num_atom + 1
            x = (k - bindex - (y - 1) * nbos) / num_atom / nbos + 1
            bo = bindex_to_o[bindex]
            cand_bonds.append((int(x) - 1, int(y) - 1, bo, float(topk_scores[j])))

        elif k >= bond_end and k < bond_end + H_end:
            k -= bond_end
            H_index = k % n_hydro
            x = int((k - H_index) / n_hydro + 1)
            h_o = hindex_to_o[H_index]
            cand_H.append((int(x) - 1, h_o, float(topk_scores[j])))

        elif k >= bond_end + H_end:
            k -= (bond_end + H_end)
            FC_index = k % n_formal
            x = int((k - FC_index) / n_formal + 1)
            fc_o = fcindex_to_o[FC_index]
            cand_FC.append((int(x) - 1, fc_o, float(topk_scores[j])))

    if len(cand_bonds)<5:
        topk_scores, topk = tf.nn.top_k(scores, k=nk+100)
        topk = topk.numpy()
        for j in range(100):
            k = topk[j+nk]
            if k < bond_end:
                bindex = k % nbos
                y = ((k - bindex) / nbos) % num_atom + 1
                x = (k - bindex - (y - 1) * nbos) / num_atom / nbos + 1
                bo = bindex_to_o[bindex]
                cand_bonds.append((int(x) - 1, int(y) - 1, bo, float(topk_scores[j])))
            if len(cand_bonds) >=5:
                # cand_H=cand_H[:6]
                # cand_FC = cand_FC[:6]
                break

    return cand_bonds, cand_H, cand_FC

def gen_cands_detailed(model=None, model_name=None,  model_dir='models', train=None, valid=None, test=None,
                    batch_size=10, cutoff=0.05, use_multiprocessing=True, workers=1, reagents=False):

    assert model or model_name, 'Either a model or model_name (path) needs to be provided'
    assert train and valid and test, 'Please provide the training, validation, and test sets'

    #Hardcoded from rexgen_direct DO NOT CHANGE
    NK3 = 80
    NK2 = 40
    NK1 = 20
    NK0 = 16
    NK = 12

    test_gen = Graph_DataLoader(test, batch_size)

    #so that we can start from a filename instead of a model object and resume training
    if model_name and not model:
        params_file = f'{model_dir}/{model_name}_core-params.txt'
        core_model = f'{model_dir}/{model_name}_core-weights.hdf5'
        try:
            with open(params_file, 'r') as f:
                model_params = json.loads(f.read())
        except:
            print('!' * 100)
            print('No Params file, will use default params for loading model. Warning: this will not work if user has changed default training parameters')
            print('!' * 100)
            model_params = {}

        hidden = model_params.get('hidden', 300)
        depth = model_params.get('depth', 3)
        output_dim = model_params.get('output_dim', 5)
        model = WLNPairwiseAtomClassifier(hidden, depth, output_dim)
        model.compile(
            loss=wln_loss(batch_size),
            metrics=[
                top_10_acc,
                top_20_acc,
                top_100_acc,
            ],
        )
        model.fit_generator(test_gen, steps_per_epoch=1, epochs=1, verbose=0)
        model.load_weights(core_model)

    print('~' * 100)
    print('Evaluating model performance')

    performance = model.evaluate_generator(test_gen, use_multiprocessing=use_multiprocessing, workers=workers)
    performance = dict(zip(model.metrics_names, performance))

    print('Performance for model on test set:')
    print(performance)
    print('~' * 100)

    #ensure that the performance on test set is high enough to continue
    assert performance.get('top_100_acc', 0.0) > cutoff, \
                        f'The top 100 accuracy for the supplied test set is below the threshold that is desired for continuing the training process \nHere are the current performance metrics {performance}'

    train_gen = Graph_DataLoader(train, batch_size, detailed=True, reagents=reagents)
    val_gen = Graph_DataLoader(valid, batch_size, detailed=True, reagents=reagents)

    detailed_file = f'{model_dir}/train_{model_name}.cbond_detailed.txt'
    train_preds = gen_core_preds(model, detailed_file, train_gen, nk=NK3) #DO NOT CHANGE NK

    detailed_file1 = f'{model_dir}/valid_{model_name}.cbond_detailed.txt'
    val_preds = gen_core_preds(model, detailed_file1, val_gen, nk=NK3)

    assert train_preds and val_preds, 'Predictions for either training set or validation set failed'

    print(f'Detailed output written to file {detailed_file} and {detailed_file1}')
    return model

if __name__ == '__main__':
    # gen_cands_detailed(model_name='uspto_500k', train='/work/data/train_trunc.txt.proc', valid='/work/data/valid_trunc.txt.proc',
    #                     test='/work/data/test_trunc.txt.proc', cutoff=-1, workers=6)
    gen_cands_detailed(model_name='mech_pred', train='data/training_set_trunc.txt.proc', valid='data/validation_set_trunc.txt.proc',
                        test='data/test_set_trunc.txt.proc', cutoff=-1, workers=6)
