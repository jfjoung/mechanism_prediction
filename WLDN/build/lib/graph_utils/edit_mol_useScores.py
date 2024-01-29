import rdkit
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
lg = RDLogger.logger()
lg.setLevel(4)

BOND_TYPE = [0, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPE[0],
    1.0: BOND_TYPE[1],
    2.0: BOND_TYPE[2],
    3.0: BOND_TYPE[3],
    1.5: BOND_TYPE[4],
}
def get_product_smiles(rmol, edits, tatoms):
    smiles = edit_mol(rmol, edits, tatoms)
    if len(smiles) != 0: return smiles
    try:
        Chem.Kekulize(rmol)
    except Exception as e:
        return smiles
    return edit_mol(rmol, edits, tatoms)

def edit_mol(rmol, edits, tatoms):
    new_mol = Chem.RWMol(rmol)
    new_mol.UpdatePropertyCache(strict=False)
#     [a.SetNumExplicitHs(0) for a in new_mol.GetAtoms()]
    amap = {}
    val_map = {}
    for atom in rmol.GetAtoms():
        amap[atom.GetIntProp('molAtomMapNumber') - 1] = atom.GetIdx()

    for x,y,t,v,l in edits:
        if l =='b':
            bond = new_mol.GetBondBetweenAtoms(amap[x],amap[y])
            if bond is not None:
                new_mol.RemoveBond(amap[x],amap[y])
            if t > 0:
                new_mol.AddBond(amap[x],amap[y],BOND_FLOAT_TO_TYPE[t])
                new_val = {a: (new_mol.GetAtomWithIdx(amap[a]).GetTotalDegree(), new_mol.GetAtomWithIdx(amap[a]).GetExplicitValence(), new_mol.GetAtomWithIdx(amap[a]).GetFormalCharge()) for a in [x,y]}
        elif l == 'h':
            a1=new_mol.GetAtomWithIdx(amap[x])
            old_Hs=a1.GetNumExplicitHs()
            a1.SetNumExplicitHs(int(old_Hs+t))
        elif l == 'f':
            a1=new_mol.GetAtomWithIdx(amap[x])
            old_FCs=a1.GetFormalCharge()
            a1.SetFormalCharge(int(old_FCs+t))

    pred_mol = new_mol.GetMol()
    pred_smiles = Chem.MolToSmiles(pred_mol, isomericSmiles=False)
    pred_list = pred_smiles.split('.')
    pred_mols = []
    for pred_smiles in pred_list:
        mol = Chem.MolFromSmiles(pred_smiles, sanitize=False)
        mol.UpdatePropertyCache(strict=False)
        if mol is None:
            continue
        atom_set = set([atom.GetIntProp('molAtomMapNumber') - 1 for atom in mol.GetAtoms()])

        if len(atom_set & tatoms) == 0:
            continue
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        pred_mols.append(mol)

    return '.'.join( sorted([Chem.MolToSmiles(pred_mol, isomericSmiles=False) for pred_mol in pred_mols]) )

# if __name__ == '__main__':
#   mol = Chem.MolFromSmiles('[CH2:42]1[O:43][CH2:44][CH2:45][N:46]2[CH:47]1[CH:48]([CH2:52][OH:53])[CH2:49][CH2:50][CH2:51]2.[OH:1][CH:2]1[CH:3]([OH:4])[CH2:5][CH2:6][N:7]([CH:8]([CH3:9])[CH2:10][N:11]2[CH2:12][CH2:13][CH:14]([NH:17][C:18](=[O:19])[c:20]3[nH:21][c:22]4[cH:23][cH:24][cH:25][c:26]([O:29][CH2:30][c:31]5[cH:32][o:33][c:34]6[c:35]5[cH:36][c:37]([Cl:40])[cH:38][cH:39]6)[c:27]4[cH:28]3)[CH2:15][CH2:16]2)[CH2:41]1')
#   edits =
