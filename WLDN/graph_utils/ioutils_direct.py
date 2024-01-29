
import rdkit.Chem as Chem
from graph_utils.mol_graph import *
import numpy as np

BOND_TYPE = [0, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
bo_to_index  = {0.0: 0, 1.0:1, 2.0:2, 3.0:3, 1.5:4}
bindex_to_o = {val:key for key, val in bo_to_index.items()}
nbos = len(bo_to_index)
binary_fdim = 5 + bond_fdim
INVALID_BOND = -1

hydro_to_index= {-1.0:0, 1.0:1}
n_hydro=len(hydro_to_index)
formal_to_index= {-2.0:0, -1.0:1, 1.0:2, 2.0:3}
n_formal=len(formal_to_index)

hindex_to_o = {val:key for key, val in hydro_to_index.items()}
fcindex_to_o = {val:key for key, val in formal_to_index.items()}

def get_bin_feature(r, max_natoms):
    '''
    This function is used to generate descriptions of atom-atom relationships, including
    the bond type between the atoms (if any) and whether they belong to the same molecule.
    It is used in the global attention mechanism.
    '''
    comp = {}
    for i, s in enumerate(r.split('.')):
        mol = Chem.MolFromSmiles(s, sanitize=False)
        for atom in mol.GetAtoms():
            comp[atom.GetIntProp('molAtomMapNumber') - 1] = i
    n_comp = len(r.split('.'))
    rmol = Chem.MolFromSmiles(r, sanitize=False)
    n_atoms = rmol.GetNumAtoms()
    bond_map = {}
    for bond in rmol.GetBonds():
        a1 = bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1
        a2 = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
        bond_map[(a1,a2)] = bond_map[(a2,a1)] = bond

    features = []
    for i in range(max_natoms):
        for j in range(max_natoms):
            f = np.zeros((binary_fdim,))
            if i >= n_atoms or j >= n_atoms or i == j:
                features.append(f)
                continue
            if (i,j) in bond_map:
                bond = bond_map[(i,j)]
                f[1:1+bond_fdim] = bond_features(bond)
            else:
                f[0] = 1.0
            f[-4] = 1.0 if comp[i] != comp[j] else 0.0
            f[-3] = 1.0 if comp[i] == comp[j] else 0.0
            f[-2] = 1.0 if n_comp == 1 else 0.0
            f[-1] = 1.0 if n_comp > 1 else 0.0
            features.append(f)
    return np.vstack(features).reshape((max_natoms,max_natoms,binary_fdim))

def get_changing_label(r, edits, max_natoms):
    rmol = Chem.MolFromSmiles(r, sanitize=False)
    n_atoms = rmol.GetNumAtoms()
    rmap = np.zeros((max_natoms, max_natoms, nbos))
    hmap = np.zeros((max_natoms, n_hydro))
    fmap = np.zeros((max_natoms, n_formal))

    bond_edit = edits.split('/')[0]
    hydrogen_edit = edits.split('/')[1]
    formal_edit = edits.split('/')[2]

    if bond_edit:
        for s in bond_edit.split(';'):
            a1, a2, bo = s.split('-')
            x = min(int(a1) - 1, int(a2) - 1)
            y = max(int(a1) - 1, int(a2) - 1)
            z = bo_to_index[float(bo)]
            rmap[x, y, z] = rmap[y, x, z] = 1

    bond_labels = []
    bond_sp_labels = []
    for i in range(max_natoms):
        for j in range(max_natoms):
            for k in range(len(bo_to_index)):
                if i == j or i >= n_atoms or j >= n_atoms:
                    bond_labels.append(INVALID_BOND)  # mask
                else:
                    bond_labels.append(rmap[i, j, k])
                    if rmap[i, j, k] == 1:
                        bond_sp_labels.append(i * max_natoms * nbos + j * nbos + k)
                        # TODO: check if this is consistent with how TF does flattening

    bond_labels = np.array(bond_labels, dtype='float32')

    if hydrogen_edit:
        for s in hydrogen_edit.split(';'):
            a1, h = s.split(':')
            x = int(a1) - 1
            z = hydro_to_index[float(h)]
            hmap[x, z] = 1
    hydrogen_labels = []
    hydrogen_sp_labels = []
    for i in range(max_natoms):
        for j in range(len(hydro_to_index)):
            if i >= n_atoms:
                hydrogen_labels.append(INVALID_BOND)  # mask
            else:
                hydrogen_labels.append(hmap[i, j])
                if hmap[i, j] == 1:
                    hydrogen_sp_labels.append(max_natoms * max_natoms * nbos + i * n_hydro + j)

    hydrogen_labels = np.array(hydrogen_labels, dtype='float32')

    if formal_edit:
        for s in formal_edit.split(';'):
            a1, f = s.split(':')
            x = int(a1) - 1
            z = formal_to_index[float(f)]
            fmap[x, z] = 1
    formal_labels = []
    formal_sp_labels = []
    for i in range(max_natoms):
        for j in range(len(formal_to_index)):
            if i >= n_atoms:
                formal_labels.append(INVALID_BOND)  # mask
            else:
                formal_labels.append(fmap[i, j])
                if fmap[i, j] == 1:
                    formal_sp_labels.append(max_natoms * max_natoms * nbos + max_natoms * n_hydro + i * n_formal + j)

    formal_labels = np.array(formal_labels, dtype='float32')

    return np.concatenate((bond_labels, hydrogen_labels, formal_labels)), np.concatenate(
        (bond_sp_labels, hydrogen_sp_labels, formal_sp_labels))

def binary_features_batch(r_list):
    mol_list = []
    max_natoms = 0
    for r in r_list:
        rmol = Chem.MolFromSmiles(r, sanitize=False)
        if rmol.GetNumAtoms() > max_natoms:
            max_natoms = rmol.GetNumAtoms()
    features = []
    #Original
    # for r in r_list:
    #     features.append(get_bin_feature(r,max_natoms))
    #Joonyoung edits
    for r in r_list:
        try:
            features.append(get_bin_feature(r,max_natoms))
        except:
            print(r)
    #Joonyong edit end
    return np.array(features)

def reactant_tracking(rxn_list, hard=False):
    '''
    hard = whether to allow reagents/solvents to contribute atoms
    '''
    all_ratoms = []
    all_rbonds = []
    for i in rxn_list:
        react,_,p = i.split('>')
        pmol = Chem.MolFromSmiles(p, sanitize=False)
        patoms = set([atom.GetIntProp('molAtomMapNumber') for atom in pmol.GetAtoms()])
        mapnum = max(patoms) + 1

        # ratoms, rbonds keep track of what parts of the reactant molecules are involved in the reaction
        ratoms = []; rbonds = []
        new_mapnums = False
        react_new = []
        for x in react.split('.'):
            xmol = Chem.MolFromSmiles(x, sanitize=False)
            xatoms = [atom.GetIntProp('molAtomMapNumber') for atom in xmol.GetAtoms()]
            if len(set(xatoms) & patoms) > 0 or hard:
                ratoms.extend(xatoms)
                rbonds.extend([
                    tuple(sorted([b.GetBeginAtom().GetIntProp('molAtomMapNumber'), b.GetEndAtom().GetIntProp('molAtomMapNumber')]) + [b.GetBondTypeAsDouble()]) \
                    for b in xmol.GetBonds()
                ])
        all_ratoms.append(ratoms)
        all_rbonds.append(rbonds)

    return all_ratoms, all_rbonds


def get_feature_batch(r_list):
    max_natoms = 0
    for r in r_list:
        rmol = Chem.MolFromSmiles(r, sanitize=False)
        if rmol.GetNumAtoms() > max_natoms:
            max_natoms = rmol.GetNumAtoms()

    features = []
    for r in r_list:
        features.append(get_bin_feature(r,max_natoms))
    return np.array(features)

def smiles2graph_list_bin(smiles_list, idxfunc=lambda x:x.GetIdx()):
    res = list(map(lambda x:smiles2graph(x,idxfunc), smiles_list))
    fatom_list, fbond_list, gatom_list, gbond_list, nb_list = zip(*res)
    return pack2D(fatom_list), pack2D(fbond_list), pack2D_withidx(gatom_list), pack2D_withidx(gbond_list), pack1D(nb_list), get_mask(fatom_list), binary_features_batch(smiles_list)

# if __name__ == '__main__':
#     smiles = ['CC']#, 'CCOCO']
#     print(binary_features_batch(smiles))
