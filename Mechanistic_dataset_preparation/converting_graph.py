import rdkit
import rdkit.Chem as Chem
import numpy as np
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

import tensorflow.compat.v1 as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers

import math
from collections import defaultdict
import random
from tqdm import tqdm
import os

### graph_utils/mol_graph.py

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 7 + 6 + 7 + 9 + 1+3
bond_fdim = 6 #if the number of bond features change this needs to be changed in WLN layer fbond_nei shape
max_nb = 10

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list)
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
            + onek_encoding_unk(atom.GetExplicitValence(), [0,1,2,3,4,5,6])  ## Add 0 valance for ion
            + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])
            + onek_encoding_unk(atom.GetFormalCharge(), [-4,-3,-2,-1,0,1,2,3,4])  ## Add formal charge
            + onek_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4,5,6])    ## Add the number of hydrogen
            + onek_encoding_unk(atom.GetNumRadicalElectrons(), [0,1,2])   ## Add the number of radical electrons
            + [atom.GetIsAromatic()], dtype=np.float32)
            

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)

def smiles2graph(smiles, idxfunc=lambda x:x.GetIdx()):
    mol = Chem.MolFromSmiles(smiles,sanitize=False) # Sanitization should be False in case of unstable intermediates
    Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
    if not mol:
        raise ValueError("Could not parse smiles string:", smiles)

    n_atoms = mol.GetNumAtoms()
    n_bonds = max(mol.GetNumBonds(), 1)
    fatoms = np.zeros((n_atoms, atom_fdim))
    fbonds = np.zeros((n_bonds, bond_fdim))
    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)

    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        if idx >= n_atoms:
            raise Exception(smiles)
        fatoms[idx] = atom_features(atom)

    for bond in mol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        idx = bond.GetIdx()
        if num_nbs[a1] == max_nb or num_nbs[a2] == max_nb:
            raise Exception(smiles)
        atom_nb[a1,num_nbs[a1]] = a2
        atom_nb[a2,num_nbs[a2]] = a1
        bond_nb[a1,num_nbs[a1]] = idx
        bond_nb[a2,num_nbs[a2]] = idx
        num_nbs[a1] += 1
        num_nbs[a2] += 1
        fbonds[idx] = bond_features(bond)
    return fatoms, fbonds, atom_nb, bond_nb, num_nbs

def pack2D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i,0:n,0:m] = arr
    return a

def pack2D_withidx(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M, 2))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i,0:n,0:m,0] = i
        a[i,0:n,0:m,1] = arr
    return a

def pack1D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        a[i,0:n] = arr
    return a

def get_mask(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        for j in range(arr.shape[0]):
            a[i][j] = 1
    return a

def smiles2graph_list(smiles_list, idxfunc=lambda x:x.GetIntProp('molAtomMapNumber') - 1):
    '''
    This function prepares all of the model inputs needed to process one batch and
    pads them as needed (because not all examples will have the same number of atoms)
    '''
    res = list(map(lambda x:smiles2graph(x,idxfunc), smiles_list))
    fatom_list, fbond_list, gatom_list, gbond_list, nb_list = zip(*res)
    return pack2D(fatom_list), pack2D(fbond_list), pack2D_withidx(gatom_list), pack2D_withidx(gbond_list), pack1D(nb_list), get_mask(fatom_list)

## wln-keras-fw/data_prep/prep_data.py

def get_bond_edits(rxn_smi):
    
    reactants = Chem.MolFromSmiles(rxn_smi.split('>')[0],sanitize=False)
    products  = Chem.MolFromSmiles(rxn_smi.split('>')[2],sanitize=False)

    conserved_maps = [a.GetIntProp('molAtomMapNumber') for a in reactants.GetAtoms() if a.GetIntProp('molAtomMapNumber')]
    bond_changes = set()

    bonds_prev = {}
    for bond in reactants.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetIntProp('molAtomMapNumber'), bond.GetEndAtom().GetIntProp('molAtomMapNumber')])
        bonds_prev['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
    bonds_new = {}
    for bond in products.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetIntProp('molAtomMapNumber'), bond.GetEndAtom().GetIntProp('molAtomMapNumber')])
        if (nums[0] not in conserved_maps) or (nums[1] not in conserved_maps): continue
        bonds_new['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()

    for bond in bonds_prev:
        if bond not in bonds_new:
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], 0.0)) # lost bond
        else:
            if bonds_prev[bond] != bonds_new[bond]:
                bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond])) # changed bond
    for bond in bonds_new:
        if bond not in bonds_prev:
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))  # new bond

    return bond_changes

def get_changed_formal(rxn_smi):
    
    reactants = Chem.MolFromSmiles(rxn_smi.split('>')[0],sanitize=False)
    products  = Chem.MolFromSmiles(rxn_smi.split('>')[2],sanitize=False)
    
    conserved_maps = [a.GetAtomMapNum() for a in products.GetAtoms() if a.HasProp('molAtomMapNumber')]
    formal_changes = set() # keep track of formal charge changes

    # Look at changed bonds
    formal_prev = {}
    for atom in reactants.GetAtoms():
        if atom.GetAtomMapNum() not in conserved_maps: continue
        formal_prev['{}'.format(atom.GetAtomMapNum())] = atom.GetFormalCharge()
        
    formal_new = {}
    for atom in products.GetAtoms():      
        formal_new['{}'.format(atom.GetAtomMapNum())] = atom.GetFormalCharge()
        
    for atom in formal_new:
        if formal_prev[atom] != formal_new[atom]:
            formal_changes.add((atom, formal_new[atom]-formal_prev[atom])) # changed formal charge
    return formal_changes

def get_changed_hydrogens(rxn_smi):
    
    reactants = Chem.MolFromSmiles(rxn_smi.split('>')[0],sanitize=False)
    products  = Chem.MolFromSmiles(rxn_smi.split('>')[2],sanitize=False)
    
    conserved_maps = [a.GetAtomMapNum() for a in products.GetAtoms() if a.HasProp('molAtomMapNumber')]
    hydrogen_changes = set() # keep track of Hydrogen changes

    # Look at changed bonds
    hydrogen_prev = {}
    for atom in reactants.GetAtoms():
        if atom.GetAtomMapNum() not in conserved_maps: continue
        hydrogen_prev['{}'.format(atom.GetAtomMapNum())] = atom.GetTotalNumHs()
        
    hydrogen_new = {}
    for atom in products.GetAtoms():      
        hydrogen_new['{}'.format(atom.GetAtomMapNum())] = atom.GetTotalNumHs()
        
    for atom in hydrogen_new:
        if hydrogen_prev[atom] != hydrogen_new[atom]:
            hydrogen_changes.add((atom, hydrogen_new[atom]-hydrogen_prev[atom])) # changed hydrogen
    
    return hydrogen_changes


