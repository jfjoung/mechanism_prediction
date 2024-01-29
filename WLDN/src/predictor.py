import os

import numpy as np
from rdkit import Chem

from WLN.models import WLNCandidateRanker, WLNPairwiseAtomClassifier
from graph_utils.ioutils_direct import smiles2graph_list_bin
from graph_utils.mol_graph_useScores import smiles2graph as s2g_edit
from src.gen_cand_score import gen_cand_single
from src.utils import set_map, enumerate_outcomes


class FWPredictor:
    """
    Class for making predictions.

    Args:
        model_name (str): the name of your model will be saved in models folder
        model_dir (str): path to the folder where the models and intermediate output will be written
        hidden_bond_classifier (int): the number of hidden layers in the WLN network of the bond classifier
        hidden_cand_ranker (int): the number of hidden layers in the WLN network of the candidate ranker
        depth (int): the depth the WLN graph convolutions will traverse
        cutoff (int): how many molecules to run through the candidate ranker

    Returns:
        model: the trained model, the history can be accessed with model.history
    """
    def __init__(self, model_name=None, model_dir=None, hidden_bond_classifier=300, hidden_candidate_ranker=500, depth=3,
                cutoff=1500, output_dim=5, core_size=16, debug=False, reagents=False):

        self.model_name = model_name
        self.cutoff = cutoff
        self.core_size = core_size
        self.model_dir = model_dir
        self.hidden_bond_classifier = hidden_bond_classifier
        self.hidden_cand_ranker = hidden_candidate_ranker
        self.depth = depth
        self.output_dim = output_dim
        self.debug = debug
        self.reagents = reagents

    def load_models(self):

        # Hardcode a smiles to init the model params
        init_smi = '[CH3:1][c:2]1[cH:3][cH:4][c:5]([C:19](=[O:20])[NH:21][CH:22]2[CH2:23][CH2:24]2)[cH:6][c:7]1-[c:8]1[cH:9][cH:10][c:11]2[c:12]([cH:13][cH:14][nH:15][c:16]2=[O:17])[cH:18]1.[Cl:25][CH2:26][c:27]1[cH:28][cH:29][cH:30][cH:31][n:32]1.[ClH:33]>>[CH3:1][c:2]1[cH:3][cH:4][c:5]([C:19](=[O:20])[NH:21][CH:22]2[CH2:23][CH2:24]2)[cH:6][c:7]1-[c:8]1[cH:9][cH:10][c:11]2[c:12]([cH:13][cH:14][nH+:15]([CH2:26][c:27]3[cH:28][cH:29][cH:30][cH:31][n:32]3)[c:16]2=[O:17])[cH:18]1.[Cl-:25].[ClH:33]'
        init_label = '12-13-0.0;12-15-1.0'
        init_cand ='B 15.0-26.0-1.0 7.199562072753906 B 25.0-26.0-0.0 6.26941442489624 F 25:-1.0 0.05872488021850586 F 15:1.0 -3.022561550140381 H 33:-1.0 -3.444171905517578 F 33:1.0 -3.9797775745391846 F 33:-1.0 -4.1625075340271 H 33:1.0 -5.541978359222412 F 15:-1.0 -6.293914794921875 F 21:1.0 -6.467335224151611 F 14:1.0 -6.4888200759887695 H 15:-1.0 -6.95046329498291 F 25:1.0 -7.239377975463867 F 12:1.0 -7.790594100952148 F 2:1.0 -7.819883346557617 F 7:1.0 -8.029661178588867 F 27:1.0 -8.144944190979004 H 21:-1.0 -8.354864120483398 B 21.0-26.0-1.0 -8.513795852661133 F 30:1.0 -8.55181884765625 F 32:1.0 -8.565037727355957 F 8:1.0 -8.647730827331543 F 17:-1.0 -8.66666030883789 F 13:1.0 -8.788835525512695 H 25:1.0 -8.793380737304688 F 26:1.0 -8.812108993530273 F 10:1.0 -8.912935256958008 F 20:-1.0 -9.082693099975586 F 9:1.0 -9.143857955932617 F 3:1.0 -9.216744422912598 F 4:1.0 -9.270949363708496 F 17:1.0 -9.381104469299316 F 29:1.0 -9.453511238098145 H 30:1.0 -9.54365348815918 H 8:1.0 -9.661615371704102 H 20:1.0 -9.682453155517578 H 21:1.0 -9.772052764892578 F 31:1.0 -9.853240966796875 F 18:1.0 -9.959197044372559 F 13:-1.0 -10.120277404785156 F 32:-1.0 -10.400331497192383 F 28:1.0 -10.417074203491211 F 6:1.0 -10.45183277130127 H 13:1.0 -10.484968185424805 F 1:-1.0 -10.54159927368164 H 25:-1.0 -10.637802124023438 F 26:-1.0 -10.676332473754883 H 29:1.0 -10.681378364562988 H 10:1.0 -10.725515365600586 F 20:1.0 -10.847947120666504 H 15:1.0 -10.875126838684082 H 3:1.0 -11.039789199829102 H 26:-1.0 -11.251022338867188 F 6:-1.0 -11.265373229980469 H 6:-1.0 -11.298473358154297 F 21:-1.0 -11.318930625915527 H 1:-1.0 -11.341318130493164 F 18:-1.0 -11.45108413696289 H 14:1.0 -11.480339050292969 H 1:1.0 -11.588715553283691 H 23:1.0 -11.596807479858398 H 24:1.0 -11.596807479858398 H 13:-1.0 -11.606613159179688 F 11:1.0 -11.619236946105957 H 7:1.0 -11.68469524383545 F 30:-1.0 -11.706260681152344 F 28:-1.0 -11.712970733642578 H 9:1.0 -11.994329452514648 H 19:1.0 -12.01513671875 F 7:-1.0 -12.109963417053223 F 5:1.0 -12.162993431091309 F 3:-1.0 -12.164909362792969 H 5:1.0 -12.222097396850586 F 1:1.0 -12.237168312072754 H 18:-1.0 -12.318384170532227 H 18:1.0 -12.39942741394043 H 31:1.0 -12.439037322998047'
        # cand_bonds = init_cand.split()
        # cand_bonds = [cand_bonds[i].split('-') + [cand_bonds[i + 1]] for i in range(0, len(cand_bonds), 2)]
        # cand_bonds = [(int(float(x)), int(float(y)), float(t), float(s)) for x, y, t, s in cand_bonds]

        parts = init_cand.split()
        cand_type_tmp, cand_change_tmp, cand_prob_tmp = [], [], []
        cand_type_tmp.append(parts[0::3])
        cand_change_tmp.append(parts[1::3])
        cand_prob_tmp.append(parts[2::3])
        cand_bond, cand_h, cand_f = [], [], []
        for cand_type, cand, prob in zip(cand_type_tmp, cand_change_tmp, cand_prob_tmp):
            b_indices = [index for index, letter in enumerate(cand_type[:self.core_size]) if letter == 'B']
            h_indices = [index for index, letter in enumerate(cand_type[:self.core_size]) if letter == 'H']
            f_indices = [index for index, letter in enumerate(cand_type[:self.core_size]) if letter == 'F']
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

        # Still have to run one molecule through to init the model, then replace weights with the saved paramaters
        reactant_smi = init_smi.split('>')[0]
        core_input = list(smiles2graph_list_bin([reactant_smi], idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1))
        # diff_input, _ = s2g_edit(reactant_smi, None, cand_bonds, None, cutoff=self.cutoff, core_size=self.core_size,
        #                          idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1, testing=True)
        diff_input, _ = s2g_edit(reactant_smi, None, cand_bond[0], cand_h[0], cand_f[0], None, None, None, cutoff=self.cutoff,
                                 idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1, testing=True)
        diff_input = [np.expand_dims(i, axis=0) for i in diff_input]

        core_model_path = f'{self.model_dir}/{self.model_name}_core-weights.hdf5'
        self.core_model = WLNPairwiseAtomClassifier(self.hidden_bond_classifier, self.depth, output_dim=self.output_dim)
        self.core_model(core_input)
        self.core_model.load_weights(core_model_path)

        diff_model_path = f'{self.model_dir}/{self.model_name}_diffnet-weights.hdf5'
        self.diff_model = WLNCandidateRanker(self.hidden_cand_ranker, self.depth)
        self.diff_model(diff_input)
        self.diff_model.load_weights(diff_model_path)
        print('**********MODELS LOADED SUCCESSFULLY**********')

    def predict_single(self, smi):
        if '>' not in smi:
            reactant_smiles, _ = set_map(smi)
        else:
            reactant_smiles = smi.split('>')[0]
            if any(not a.HasProp('molAtomMapNumber') for a in Chem.MolFromSmiles(reactant_smiles).GetAtoms()):
                reactant_smiles, _ = set_map(reactant_smiles)
            else:
                pass

        graph_inputs = list(smiles2graph_list_bin([reactant_smiles], idxfunc=lambda x:x.GetIntProp('molAtomMapNumber') - 1))
        score = self.core_model.predict(graph_inputs)
        self.score = score
        cand_bonds, cand_h, cand_fc = gen_cand_single(score[0], smiles=reactant_smiles, reagents=self.reagents, nk=self.core_size)

        graph_inputs, conf = s2g_edit(reactant_smiles, None, cand_bonds, cand_h, cand_fc, None, None, None, cutoff=self.cutoff, \
                                       idxfunc=lambda x:x.GetIntProp('molAtomMapNumber') - 1, testing=True)

        inputs = (np.expand_dims(graph_inputs[0], axis=0),
            np.expand_dims(graph_inputs[1], axis=0),
            np.expand_dims(graph_inputs[2], axis=0),
            np.expand_dims(graph_inputs[3], axis=0),
            np.expand_dims(graph_inputs[4], axis=0),
            np.expand_dims(graph_inputs[5], axis=0))

        score = self.diff_model.predict(inputs)
        outcomes = enumerate_outcomes(reactant_smiles, conf, score)

        if self.debug:
            return outcomes, smi, sorted(cand_bonds, key=lambda x : x[3], reverse=True)
        else:
            return outcomes

    def save_model(self, output_dir):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.core_model.save(f'{output_dir}/{self.model_name}_core/1', save_format='tf')
        self.diff_model.save(f'{output_dir}/{self.model_name}_diffnet/1', save_format='tf')


if __name__ == '__main__':

    #For debugging
    predictor = FWPredictor(model_name='mech_pred', model_dir='mech_pred')
    predictor.load_models()
    # res = predictor.predict_single('CCOCC.C=C(C1Cc2c(O1)ccc(C(C)(C)C)c2)O.BrBr')
    # res = predictor.predict_single('Br[Br+][Fe-](Br)(Br)Br.Cc1cc(Cl)c(Cl)cc1')
    res = predictor.predict_single('CC1CCC(O1)CO.Cc2ccc(S(=O)(Cl)=O)cc2.CCN(CC)CC.ClCCl')

    print(list(zip([x.get('smiles', ['R']) for x in res], \
      [x.get('prob', -1) for x in res])))
