import rdkit
from rdkit import Chem
from molvs import Standardizer
from rdkit.Chem import AllChem
from optparse import OptionParser
# from rdkit import RDLogger
# lg = RDLogger.logger()
# lg.setLevel(4)

'''
This script evaluates the quality of predictions from the rank_diff_wln model by applying the predicted
graph edits to the reactants, cleaning up the generated product, and comparing it to what was recorded
as the true (major) product of that reaction
'''

BOND_TYPE = [0, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPE[0],
    1.0: BOND_TYPE[1],
    2.0: BOND_TYPE[2],
    3.0: BOND_TYPE[3],
    1.5: BOND_TYPE[4],
}

hydro_to_index= {-1.0:0, 1.0:1}
formal_to_index= {-2.0:0, -1.0:1, 1.0:2, 2.0:3}
hindex_to_o = {val:key for key, val in hydro_to_index.items()}
fcindex_to_o = {val:key for key, val in formal_to_index.items()}

def edit_mol_smiles(rmol, edits):
    new_mol = Chem.RWMol(rmol)
    new_mol.UpdatePropertyCache(strict=False)
    #     [a.SetNumExplicitHs(0) for a in new_mol.GetAtoms()]
    amap = {}
    val_map = {}
    for atom in rmol.GetAtoms():
        amap[atom.GetIntProp('molAtomMapNumber') - 1] = atom.GetIdx()

    if edits:
        for x, y, t, l in edits:
            if l == 'b':
                bond = new_mol.GetBondBetweenAtoms(amap[x], amap[y])
                if bond is not None:
                    new_mol.RemoveBond(amap[x], amap[y])
                if t > 0:
                    new_mol.AddBond(amap[x], amap[y], BOND_TYPE[int(t)])  # It was BOND_FLOAT_TO_TYPE[t]
            elif l == 'h':
                a1 = new_mol.GetAtomWithIdx(amap[x])
                old_Hs = a1.GetNumExplicitHs()
                a1.SetNumExplicitHs(int(old_Hs + hindex_to_o[t]))
            elif l == 'f':
                a1 = new_mol.GetAtomWithIdx(amap[x])
                old_FCs = a1.GetFormalCharge()
                # print(old_FCs)
                # print(t)
                # print(fcindex_to_o[t])
                a1.SetFormalCharge(int(old_FCs + fcindex_to_o[t]))
    else: pass

    pred_mol = new_mol.GetMol()
    if Chem.rdmolops.DetectChemistryProblems(pred_mol):
        return None  # To remove invalid molecules

    for atom in pred_mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')

    pred_smiles = Chem.MolToSmiles(pred_mol, isomericSmiles=False)
    # pred_list = pred_smiles.split('.')
    # pred_mols = []
    # for pred_smiles in pred_list:
    #     mol = Chem.MolFromSmiles(pred_smiles, sanitize=False)
    #     mol.UpdatePropertyCache(strict=False)
    #     if mol is None:
    #         continue
    #     pred_mols.append(mol)
    # pred_smiles = [Chem.MolToSmiles(pred_mol, isomericSmiles=False) for pred_mol in pred_mols if pred_mol is not None]
    return pred_smiles

def eval_by_smiles(pred_path, gold_path, singleonly=False, bonds_as_doubles=False):
    fpred = open(pred_path)
    fgold = open(gold_path)
    feval = open(pred_path + '.eval_by_smiles', 'w')

    print('## Bond types in output files are doubles? {}'.format(bonds_as_doubles))

    idxfunc = lambda a: a.GetIntProp('molAtomMapNumber') - 1
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                  Chem.rdchem.BondType.AROMATIC]
    bond_types_as_double = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}

    # Define a standardization procedure so we can evaluate based on...
    # a) RDKit-sanitized equivalence, and
    # b) MOLVS-sanitized equivalence
    standardizer = Standardizer()
    standardizer.prefer_organic = True

    def sanitize_smiles(smi, largest_fragment=False):
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return smi
        try:
            mol = standardizer.standardize(mol) # standardize functional group reps
            if largest_fragment:
                mol = standardizer.largest_fragment(mol) # remove product counterions/salts/etc.
            mol = standardizer.uncharge(mol) # neutralize, e.g., carboxylic acids
        except Exception:
            pass
        return Chem.MolToSmiles(mol, isomericSmiles=False)


    try:
        i = 0
        n,top1,top2,top3,top5,gfound = 0,0,0,0,0,0
        # top1_sani, top2_sani, top3_sani, top5_sani, gfound_sani = 0, 0, 0, 0, 0
        for line in fpred:
            thisrow = []
            line = line.strip('\r\n |')
            gold = fgold.readline()
            rex,gedits = gold.split()
            r,_,p = rex.split('>')

            if singleonly and '.' in p:
                continue

            rmol = Chem.MolFromSmiles(r, sanitize=False)
            pmol = Chem.MolFromSmiles(p, sanitize=False)

            thisrow.append(r)
            thisrow.append(p)

            # Save pbond information
            pbonds = {}
            for bond in pmol.GetBonds():
                a1 = idxfunc(bond.GetBeginAtom())
                a2 = idxfunc(bond.GetEndAtom())
                t = bond_types.index(bond.GetBondType())
                pbonds[(a1, a2)] = pbonds[(a2, a1)] = t + 1

            for atom in pmol.GetAtoms():
                atom.ClearProp('molAtomMapNumber')

            psmiles = Chem.MolToSmiles(pmol, isomericSmiles=False)
            # psmiles_sani = set(sanitize_smiles(psmiles, True).split('.'))
            psmiles = set(psmiles.split('.'))

            thisrow.append('.'.join(psmiles))
            # thisrow.append('.'.join(psmiles_sani))


            ########### Use *true* edits to try to recover product
            ## x, y, t, l style
            gold_bond, gold_h, gold_f = [], [], []
            bond_edits, h_edits, f_edits = gedits.split('/')

            cbonds = []
            if bond_edits:
                for bond in bond_edits.split(';'):
                        x, y, t = bond.split('-')
                        x, y, t = int(x), int(y), float(t)
                        cbonds.append((x - 1, y - 1, bond_types_as_double[t], 'b'))
            if h_edits:
                for h in h_edits.split(';'):
                    x, t = h.split(':')
                    x, t = int(x), int(t)
                    cbonds.append((x - 1, x - 1, hydro_to_index[t], 'h'))
            if f_edits:
                for f in f_edits.split(';'):
                    x, t = f.split(':')
                    x, t = int(x), int(t)
                    cbonds.append((x - 1, x - 1, formal_to_index[t], 'f'))
            # Generate products by modifying reactants with predicted edits.
            pred_smiles = edit_mol_smiles(rmol, cbonds)
            # print(pred_smiles)

            if pred_smiles:
                pred_smiles = set(pred_smiles.split('.'))
            else: pred_smiles = set()
            if not psmiles <= pred_smiles:
                print('\nwarn: could not regenerate product {}'.format(psmiles))
                print(r)
                print(p)
                print(gedits)
                print(cbonds)
                print('pred_smiles: {}'.format(pred_smiles))
            else:
                gfound += 1
                # gfound_sani += 1

            ########### Now use candidate edits to try to recover product

            rk,rk_sani = 11,11
            pred_smiles_list = []
            pred_smiles_sani_list = []
            ctr = 0
            # print(line.split('|'))
            for idx,edits in enumerate(line.split('|')):
                prev_len_pred_smiles = len(set(pred_smiles_list))
                couldnt_find_smiles = True
                cbonds = []
                if 'END' in edits:
                    pass
                else:
                    for edit in edits.split():
                        x,y,t,l = edit.split('-')
                        x,y,t=float(x), float(y),float(t)
                        if l == 'b':
                            if bonds_as_doubles:
                                x,y,t = int(x), int(y), bond_types_as_double[float(t)]
                            else:
                                x,y,t = int(x),int(y),int(t)
                            cbonds.append((x,y,t,l))
                        elif l == 'h':
                            x, t = int(x), int(t)
                            cbonds.append((x, x, t, 'h'))
                        elif l == 'f':
                            x, t = int(x), int(t)
                            cbonds.append((x, x, t, 'f'))
                #Generate products by modifying reactants with predicted edits.
                try:
                    pred_smiles = edit_mol_smiles(rmol, cbonds)
                    pred_smiles = set(pred_smiles.split('.'))
                    pred_smiles = set(pred_smiles)
                except: pred_smiles=set()

                if psmiles <= pred_smiles:
                    rk = min(rk, ctr + 1)
                pred_smiles_list.append('.'.join(pred_smiles))


                # If we failed to come up with a new candidate, don't increment the counter!
                if len(set(pred_smiles_list)) > prev_len_pred_smiles:
                    ctr += 1

            n += 1.0
            if rk == 1: top1 += 1
            if rk <= 2: top2 += 1
            if rk <= 3: top3 += 1
            if rk <= 5: top5 += 1
            # if rk_sani == 1: top1_sani += 1
            # if rk_sani <= 2: top2_sani += 1
            # if rk_sani <= 3: top3_sani += 1
            # if rk_sani <= 5: top5_sani += 1

            thisrow.append(rk)
            while len(pred_smiles_list) < 10:
                pred_smiles_list.append('n/a')
            thisrow.extend(pred_smiles_list)

            print('[strict]  acc@1: %.4f, acc@2: %.4f, acc@3: %.4f, acc@5: %.4f (after seeing %d) gfound = %.4f' % (top1 / n, top2 / n, top3 / n, top5 / n, n, gfound / n))
            # print('[molvs]   acc@1: %.4f, acc@2: %.4f, acc@3: %.4f, acc@5: %.4f (after seeing %d) gfound = %.4f' % (top1_sani / n, top2_sani / n, top3_sani / n, top5_sani / n, n, gfound_sani / n))
            feval.write('\t'.join([str(x) for x in thisrow]) + '\n')
            # i+=1
            # if i == 4:
            #     break

    finally:
        fpred.close()
        fgold.close()
        feval.close()

if __name__ == "__main__":

    # parser = OptionParser()
    # parser.add_option("-t", "--pred", dest="pred_path") # file containing predicted edits
    # parser.add_option("-g", "--gold", dest="gold_path") # file containing true edits
    # parser.add_option("-s", "--singleonly", dest="singleonly", default=False) # only compare single products
    # parser.add_option("--bonds_as_doubles", dest="bonds_as_doubles", default=False) # bond types are doubles, not indices
    # opts,args = parser.parse_args()

    eval_by_smiles(pred_path=f'mech_pred/test_mech_pred.predictions.txt', gold_path='data/test_mini.txt.proc')
    # eval_by_smiles(pred_path=opts.pred, gold_path=opts.gold, singleonly=opts.singleonly, bonds_as_doubles=opts.bonds_as_doubles)
