import json
import numpy as np

from WLN.data_loading import Graph_DataLoader, Candidate_DataLoader
from WLN.metrics import wln_loss, top_10_acc, top_20_acc, top_100_acc
from WLN.models import WLNCandidateRanker, WLNPairwiseAtomClassifier
from src.gen_cand_score import gen_core_preds
from src.utils import enumerate_outcomes
from graph_utils.ioutils_direct import smiles2graph_list_bin
from graph_utils.mol_graph_useScores import smiles2graph as s2g_edit


def gen_cands_detailed_testing(core_model=None, model_name=None,  model_dir='models', test=None,
                    batch_size=10, reagents=False):

    assert core_model or model_name, 'Either a model or model_name (path) needs to be provided'
    assert test, 'Please provide the test set'

    #Hardcoded from rexgen_direct DO NOT CHANGE
    NK3 = 80
    NK2 = 40
    NK1 = 20
    NK0 = 16
    NK = 12

    test_gen = Graph_DataLoader(test, batch_size, detailed=True, shuffle=False, reagents=reagents)

    if model_name and not core_model:
        params_file = f'{model_dir}/{model_name}_core-params.txt'
        core_model_path = f'{model_dir}/{model_name}_core-weights.hdf5'
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
        core_model = WLNPairwiseAtomClassifier(hidden, depth, output_dim)

        # Hardcode a smiles to init the model params
        init_smi = '[CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14]'
        reactant_smi = init_smi.split('>')[0]
        core_input = list(smiles2graph_list_bin([reactant_smi], idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1))
        core_model(core_input)
        core_model.load_weights(core_model_path)
        print('**********MODEL LOADED SUCCESSFULLY**********')

    detailed_file = f'{model_dir}/test_{model_name}.cbond_detailed.txt'
    gen_core_preds(core_model, detailed_file, test_gen, nk=NK3)

    print(f'Detailed output written to file {detailed_file}')


def test_wln_diffnet(test=None, batch_size=1, model_name='wln_diffnet', model_dir='models', core_size=16):

    """
    Tests the candidate ranker.

    Args:
        test (str): path to test data
        model_name (str): the name of your model will be saved in models folder
        model_dir (str): path to the folder where the models and intermediate output will be written
        batch_size (int): batch size for training, 1 is currently the only batch_size supported
    """

    init_smi = '[CH3:1][c:2]1[cH:3][cH:4][c:5]([C:19](=[O:20])[NH:21][CH:22]2[CH2:23][CH2:24]2)[cH:6][c:7]1-[c:8]1[cH:9][cH:10][c:11]2[c:12]([cH:13][cH:14][nH:15][c:16]2=[O:17])[cH:18]1.[Cl:25][CH2:26][c:27]1[cH:28][cH:29][cH:30][cH:31][n:32]1.[ClH:33]>>[CH3:1][c:2]1[cH:3][cH:4][c:5]([C:19](=[O:20])[NH:21][CH:22]2[CH2:23][CH2:24]2)[cH:6][c:7]1-[c:8]1[cH:9][cH:10][c:11]2[c:12]([cH:13][cH:14][nH+:15]([CH2:26][c:27]3[cH:28][cH:29][cH:30][cH:31][n:32]3)[c:16]2=[O:17])[cH:18]1.[Cl-:25].[ClH:33]'
    init_cand ='B 15.0-26.0-1.0 7.199562072753906 B 25.0-26.0-0.0 6.26941442489624 F 25:-1.0 0.05872488021850586 F 15:1.0 -3.022561550140381 H 33:-1.0 -3.444171905517578 F 33:1.0 -3.9797775745391846 F 33:-1.0 -4.1625075340271 H 33:1.0 -5.541978359222412 F 15:-1.0 -6.293914794921875 F 21:1.0 -6.467335224151611 F 14:1.0 -6.4888200759887695 H 15:-1.0 -6.95046329498291 F 25:1.0 -7.239377975463867 F 12:1.0 -7.790594100952148 F 2:1.0 -7.819883346557617 F 7:1.0 -8.029661178588867 F 27:1.0 -8.144944190979004 H 21:-1.0 -8.354864120483398 B 21.0-26.0-1.0 -8.513795852661133 F 30:1.0 -8.55181884765625 F 32:1.0 -8.565037727355957 F 8:1.0 -8.647730827331543 F 17:-1.0 -8.66666030883789 F 13:1.0 -8.788835525512695 H 25:1.0 -8.793380737304688 F 26:1.0 -8.812108993530273 F 10:1.0 -8.912935256958008 F 20:-1.0 -9.082693099975586 F 9:1.0 -9.143857955932617 F 3:1.0 -9.216744422912598 F 4:1.0 -9.270949363708496 F 17:1.0 -9.381104469299316 F 29:1.0 -9.453511238098145 H 30:1.0 -9.54365348815918 H 8:1.0 -9.661615371704102 H 20:1.0 -9.682453155517578 H 21:1.0 -9.772052764892578 F 31:1.0 -9.853240966796875 F 18:1.0 -9.959197044372559 F 13:-1.0 -10.120277404785156 F 32:-1.0 -10.400331497192383 F 28:1.0 -10.417074203491211 F 6:1.0 -10.45183277130127 H 13:1.0 -10.484968185424805 F 1:-1.0 -10.54159927368164 H 25:-1.0 -10.637802124023438 F 26:-1.0 -10.676332473754883 H 29:1.0 -10.681378364562988 H 10:1.0 -10.725515365600586 F 20:1.0 -10.847947120666504 H 15:1.0 -10.875126838684082 H 3:1.0 -11.039789199829102 H 26:-1.0 -11.251022338867188 F 6:-1.0 -11.265373229980469 H 6:-1.0 -11.298473358154297 F 21:-1.0 -11.318930625915527 H 1:-1.0 -11.341318130493164 F 18:-1.0 -11.45108413696289 H 14:1.0 -11.480339050292969 H 1:1.0 -11.588715553283691 H 23:1.0 -11.596807479858398 H 24:1.0 -11.596807479858398 H 13:-1.0 -11.606613159179688 F 11:1.0 -11.619236946105957 H 7:1.0 -11.68469524383545 F 30:-1.0 -11.706260681152344 F 28:-1.0 -11.712970733642578 H 9:1.0 -11.994329452514648 H 19:1.0 -12.01513671875 F 7:-1.0 -12.109963417053223 F 5:1.0 -12.162993431091309 F 3:-1.0 -12.164909362792969 H 5:1.0 -12.222097396850586 F 1:1.0 -12.237168312072754 H 18:-1.0 -12.318384170532227 H 18:1.0 -12.39942741394043 H 31:1.0 -12.439037322998047'
    reactant_smi = init_smi.split('>')[0]
    parts = init_cand.split()
    # cand_bonds = [cand_bonds[i].split('-') + [cand_bonds[i + 1]] for i in range(0, len(cand_bonds), 2)]
    # cand_bonds = [(int(float(x)), int(float(y)), float(t), float(s)) for x, y, t, s in cand_bonds]

    cand_type_tmp, cand_change_tmp, cand_prob_tmp = [], [], []
    cand_type_tmp.append(parts[0::3])
    cand_change_tmp.append(parts[1::3])
    cand_prob_tmp.append(parts[2::3])
    cand_bond, cand_h, cand_f = [], [], []
    for cand_type, cand, prob in zip(cand_type_tmp, cand_change_tmp, cand_prob_tmp):
        b_indices = [index for index, letter in enumerate(cand_type[:core_size]) if letter == 'B']
        h_indices = [index for index, letter in enumerate(cand_type[:core_size]) if letter == 'H']
        f_indices = [index for index, letter in enumerate(cand_type[:core_size]) if letter == 'F']
        bonds = []
        for i in b_indices:
            x, y, t = cand[i].split('-')
            x, y = tuple(sorted([int(float(x)) - 1, int(float(y)) - 1]))
            bonds.append((x, y, float(t), float(prob[i])))
        cand_bond.append(bonds)

        Hs = []
        for i in h_indices:
            x, t = cand[i].split(':')
            x = int(float(x)) - 1
            Hs.append((x, float(t), float(prob[i])))
        cand_h.append(Hs)

        FCs = []
        for i in f_indices:
            x, t = cand[i].split(':')
            x = int(float(x)) - 1
            FCs.append((x, float(t), float(prob[i])))
        cand_f.append(FCs)
    diff_input, _ = s2g_edit(reactant_smi, None,cand_bond[0],cand_h[0], cand_f[0],None, None, None, cutoff=150,
                                 idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1, testing=True)

    # diff_input, _ = s2g_edit(reactant_smi, None, cand_bonds, None, cutoff=150, core_size=16,
    #                              idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1, testing=True)
    diff_input = [np.expand_dims(i, axis=0) for i in diff_input]

    params_file = f'{model_dir}/{model_name}_diffnet-params.txt'
    diff_model_path = f'{model_dir}/{model_name}_diffnet-weights.hdf5'
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

    diff_model = WLNCandidateRanker(hidden, depth)
    diff_model(diff_input)
    diff_model.load_weights(diff_model_path)
    print('**********MODEL LOADED SUCCESSFULLY**********')

    assert test, 'Please specify the test set'

    test_detailed = f'{model_dir}/test_{model_name}.cbond_detailed.txt'

    test_gen = Candidate_DataLoader(test_detailed, batch_size, cutoff=1500, core_size=16, testing=True)

    assert len(test_gen) > 0, f'Test set has {len(test_gen)} examples, has to be greater than 0'

    assert model_name and model_dir, 'Model name and directory must be provided!'

    pred_path = f'{model_dir}/test_{model_name}'
    # i=0
    with open(pred_path + '.predictions.txt', 'w') as fpred:
        for batch in test_gen:
            inputs, conf, rsmi = batch
            score = diff_model.predict(inputs)

            outcomes = enumerate_outcomes(rsmi, conf, score)
            # print(rsmi)
            # print(conf)
            # print(score)
            for outcome in outcomes:
                # print(outcome['predicted_edits'])
                # print(outcome['rank'])
                # print(outcome['prob'], '\n')
                if outcome['predicted_edits']:
                    for x, y, t, l in outcome['predicted_edits']:
                        fpred.write("{}-{}-{}-{} ".format(x, y, t, l))
                else:
                    fpred.write("END")
                fpred.write(' | ')
            fpred.write(' \n')
            # i+=1
            # if i == 12:
            #     break


if __name__ == '__main__':
    gen_cands_detailed_testing(model_name='mech_pred', model_dir='mech_pred',
                        test='data/test_mini.txt.proc', batch_size=10, reagents=False)
    test_wln_diffnet(batch_size=1, model_name='mech_pred', model_dir='mech_pred',
                     test='data/test_mini.txt.proc')
