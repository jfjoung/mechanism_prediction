import rdkit
import rdkit.Chem as Chem
import numpy as np
import random
from graph_utils.edit_mol_useScores import get_product_smiles #graph_utils.
from collections import defaultdict

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
atom_fdim = len(elem_list) + 6 + 7 + 6 + 7 + 9 + 1+3
bond_fdim = 5
max_nb = 10

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list)
                    + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                    + onek_encoding_unk(atom.GetExplicitValence(), [0, 1, 2, 3, 4, 5, 6])  ## Add 0 valance for ion
                    + onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
                    + onek_encoding_unk(atom.GetFormalCharge(), [-4, -3, -2, -1, 0, 1, 2, 3, 4])  ## Add formal charge
                    + onek_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6])  ## Add the number of hydrogen
                    + onek_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1, 2])  ## Add the number of radical electrons
                    + [atom.GetIsAromatic()], dtype=np.float32)

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()], dtype=np.float32)

def packnb(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M, 2))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i,0:n,0:m,0] = i
        a[i,0:n,0:m,1] = arr
    return a

def smiles2graph(rsmiles, psmiles, core_bonds, core_h, core_f, gold_bonds, gold_h, gold_f, cutoff=500,
                      idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1, kmax=7, return_found=False,
                      testing=False):
    '''This is the function that takes reactants, a true product (when defined), and the candidate bonds
    to generate all of the candidate products according to some bounds on the enumeration'''


    mol = Chem.MolFromSmiles(rsmiles, sanitize=False)
    if not mol:
        raise ValueError("Could not parse smiles string:", rsmiles)
    mol.UpdatePropertyCache(strict=False)
    if not testing:
        pmol = Chem.MolFromSmiles(psmiles, sanitize=False)
        if not pmol:
            raise ValueError("Could not parse smiles string:", psmiles)
        pmol.UpdatePropertyCache(strict=False)

    n_atoms = mol.GetNumAtoms()
    n_bonds = max(mol.GetNumBonds(), 1)
    fatoms = np.zeros((n_atoms, atom_fdim))
    fbonds = np.zeros((n_bonds, bond_fdim))
    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)
    raw_atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    raw_bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    raw_num_nbs = np.zeros((n_atoms,), dtype=np.int32)
    free_vals = np.zeros((n_atoms,))
    pfree_vals = np.zeros((n_atoms,))

    # gbonds = {(x,y):0 for x,y in core_bonds}

    # Feature Extraction
    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        fatoms[idx] = atom_features(atom)

    if not testing:
        tatoms = set()
        # Calculate free slots for each atom in product
        for bond in pmol.GetBonds():
            a1 = idxfunc(bond.GetBeginAtom())
            a2 = idxfunc(bond.GetEndAtom())
            t = bond_types.index(bond.GetBondType()) + 1
            a1, a2 = min(a1, a2), max(a1, a2)
            tatoms.add(a1)
            tatoms.add(a2)
            if (a1, a2) in core_bonds:
                # gbonds[(a1,a2)] = t
                tval = t if t < 4 else 1.5
                pfree_vals[a1] += tval
                pfree_vals[a2] += tval

    rbonds = {}
    rbond_vals = {}  # bond orders
    ring_bonds = set()
    # Calculate free slots for each atom in reactant
    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        t = bond_types.index(bond.GetBondType()) + 1
        a1, a2 = min(a1, a2), max(a1, a2)
        tval = t if t < 4 else 1.5
        rbonds[(a1, a2)] = t
        rbond_vals[(a1, a2)] = tval
        if (a1, a2) in core_bonds:
            free_vals[a1] += tval
            free_vals[a2] += tval
        if bond.IsInRing():
            ring_bonds.add((a1, a2))

            # Get all possible core configurations - NEW IN DIRECT VERSION
    from itertools import combinations
    core_configs = []  # will be list of lists of (x, y, t, v, l) tuples, where t is the bond order and v is CoreFinder score, l is label

    # Filter out core bonds that exactly match reactants
    #     prev_len = len(core_bonds)
    core_bonds = [(x, y, t, v, 'b') for (x, y, t, v) in core_bonds if ((x, y) not in rbond_vals) or (rbond_vals[(x, y)] != t)]

    index_list = [{i} for i in range(len(core_bonds))]
    indices = len(core_bonds)

    new_core_h = []
    for x, t, v in core_h:
        h_index = set()
        for (a, b), c in rbonds.items():
            if x == a:
                new_core_h.append((x, b, t, v, 'h'))
                h_index.add(indices)
                indices += 1
            elif x == b:
                new_core_h.append((x, a, t, v, 'h'))
                h_index.add(indices)
                indices += 1
        if len(h_index) == 0:
            new_core_h.append((x, x, t, v, 'h'))
            h_index.add(indices)
            indices += 1
        index_list.append(h_index)

    new_core_f = []
    for x, t, v in core_f:
        f_index = set()
        for (a, b), c in rbonds.items():
            if x == a:
                new_core_f.append((x, b, t, v, 'f'))
                f_index.add(indices)
                indices += 1
            elif x == b:
                new_core_f.append((x, a, t, v, 'f'))
                f_index.add(indices)
                indices += 1
        if len(f_index) == 0:
            new_core_f.append((x, x, t, v, 'f'))
            f_index.add(indices)
            indices += 1
        index_list.append(f_index)

    #     core_changes=core_bonds+core_h+core_f
    core_changes = core_bonds + new_core_h + new_core_f

    # Helper function to check if a combination is connected - this helps the number of valid combinations
    core_bonds_adj = np.eye(len(core_changes), dtype=bool)
    for i in range(len(core_changes)):
        a1, b1, t1, v1, l1 = core_changes[i]
        for j in range(i, len(core_changes)):
            a2, b2, t2, v2, l2 = core_changes[j]
            if a1 == a2 or a1 == b2 or b1 == a2 or b1 == b2:
                core_bonds_adj[i, j] = core_bonds_adj[j, i] = True

    #     print('Calculated core bonds adj matrix: \n {}'.format(core_bonds_adj * 1.0))

    def check_if_connected(combo_i):
        '''Checks if a set of candidate edits (by indeces) are all connected'''
        if len(combo_i) == 1 or len(combo_i) == 0:
            return True  # only one change, always connected
        temp_adj_pow = np.linalg.matrix_power(core_bonds_adj[combo_i, :][:, combo_i], len(combo_i) - 1)
        return np.all(temp_adj_pow)

    def two_change_in_same_atom(bond_change_combo):
        if len(bond_change_combo) == 1 or len(bond_change_combo) == 0:
            return True

        # Connor's suggestion
        b_atoms = [tup[0] for tup in bond_change_combo if tup[4] == 'b']
        f_atoms = [tup[0] for tup in bond_change_combo if tup[4] == 'f']

        if len(f_atoms) != len(set(f_atoms)) or len(f_atoms) > 2:
            return False

        h_atoms = [tup[0] for tup in bond_change_combo if tup[4] == 'h']
        if len(h_atoms) != len(set(h_atoms)) or len(h_atoms) > 2:
            return False
        len_b = len(b_atoms)
        len_h = len(h_atoms)
        len_f = len(f_atoms)

        if len_b > 5:
            return False
        edit_comb = (len_b, len_h, len_f)
        # In the order of b, h, f
        avoid_list = [(0, 1, 0), (0, 2, 0), (0, 2, 1), (0, 0, 1), (0, 0, 2), (0, 1, 2),
                      (1, 1, 0), (1, 2, 0), (1, 0, 1), (1, 2, 1), (1, 1, 2),
                      (2, 0, 1), (2, 2, 1),
                      (3, 2, 0), (3, 2, 1), (3, 1, 2), (3, 2, 2),
                      (4, 1, 0), (4, 2, 0), (4, 0, 1), (4, 2, 1), (4, 1, 2), (4, 2, 2),
                      (5, 1, 1), (5, 2, 1), (5, 1, 2), (5, 2, 2)]
        if edit_comb in avoid_list:
            return False
        return True

    # N choose k combinatorics
    # up to 5 bond changes at once - only 0.19% of train examples have 5 bonds changed, we can take the hit...
    core_bonds_i = range(len(index_list))
    for k in range(0, kmax+1):
        for bond_change_combo_i in combinations(core_bonds_i, k):
            list_of_index=[index_list[i] for i in bond_change_combo_i]
            new_combination=[x for y in list_of_index for x in y]
            bond_change_combo = [core_changes[list(i)[0]] for i in list_of_index]
            if len(list_of_index) <2:
                core_configs.append(bond_change_combo)
            else:
                # Check if connected
                if two_change_in_same_atom(bond_change_combo):
                    if not check_if_connected(new_combination):
                        continue
                    core_configs.append(bond_change_combo)

    # print('Total possible combiations: {}'.format(len(core_configs)))

    if not testing:
        random.shuffle(core_configs)
        idx = -1
        for i, cand_edits in enumerate(core_configs):
            if set([(x, y, t) for (x, y, t, v, l) in cand_edits if l == 'b']) == gold_bonds:
                if set([(x, t) for (x, y, t, v, l) in cand_edits if l == 'h']) == gold_h:
                    if set([(x, t) for (x, y, t, v, l) in cand_edits if l == 'f']) == gold_f:
                        idx = i
                        break
        #         print(idx)
        # If we are training and did not find the true outcome, make sure it is the first entry
        if idx == -1:
            # print('Did not find true outcome')
            found_true = False
            core_configs = [
                               [(x, y, t, 0.0, 'b') for (x, y, t) in gold_bonds]
                               + [(x, x, t, 0.0, 'h') for (x, t) in gold_h]
                               + [(x, x, t, 0.0, 'f') for (x, t) in gold_f]
                           ] + core_configs
        else:
            # print('Found true outcome')
            found_true = True
            core_configs[0], core_configs[idx] = core_configs[idx], core_configs[0]  # swap order so true is first
    else:
        found_true = False

    if not testing:
        # If it is possible to recover the true smiles from the set of bonds using the edit_mol method,
        # remove duplicates from the list by converting each candidate into a smiles string
        # note: get_product_smiles is HIGHLY imperfect, but that's not a huge deal. training tries to pick the
        #       right bonds. The evaluation script has a more robust function to get product_smiles
        smiles0 = get_product_smiles(mol, core_configs[0], tatoms)

        if len(smiles0) > 0:  #
            cand_smiles = set([smiles0])
            new_core_configs = [core_configs[0]]

            for core_conf in core_configs[1:]:
                try:
                    smiles = get_product_smiles(mol, core_conf, tatoms)
                except:
                    continue
                # print('candidate smiles: {}'.format(smiles))
                if smiles in cand_smiles or len(smiles) == 0:
                    continue
                cand_smiles.add(smiles)
                new_core_configs.append(core_conf)
                if len(new_core_configs) > cutoff:
                    break
            core_configs = new_core_configs

        else:
            # TODO log instead of actually printing to out
            print('\nwarning! could not recover true smiles from gbonds: {}'.format(psmiles))
            print('{}    {}'.format(rsmiles, gold_bonds))

    # print('After removing duplicates, {} core configs'.format(len(core_configs)))
    # print(core_configs)
    core_configs = core_configs[:cutoff]

    n_batch = len(core_configs) + 1
    if not testing:
        labels = np.zeros((n_batch - 1,))
        labels[0] = 1

    # Calculate information that is the same for all candidates; do small updates based on specific changes later
    pending_reactant_neighbors = []  # reactant neighbors that *might* be over-ridden

    core_bonds_noScore = [(x, y, t) for (x, y, t, z, l) in core_bonds]

    #     print(core_bonds_noScore)

    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        a1, a2 = min(a1, a2), max(a1, a2)

        if (a1, a2, 0.0) not in core_bonds_noScore:  # are a1 and a2 guaranteed to be neighbors?
            raw_atom_nb[a1, raw_num_nbs[a1]] = a2
            raw_atom_nb[a2, raw_num_nbs[a2]] = a1
            raw_bond_nb[a1, raw_num_nbs[a1]] = idx
            raw_bond_nb[a2, raw_num_nbs[a2]] = idx
            raw_num_nbs[a1] += 1
            raw_num_nbs[a2] += 1
        else:
            pending_reactant_neighbors.append((a1, a2, bond.GetBondTypeAsDouble()))

        # Reactants have this bond...
        atom_nb[a1, num_nbs[a1]] = a2
        atom_nb[a2, num_nbs[a2]] = a1
        bond_nb[a1, num_nbs[a1]] = idx
        bond_nb[a2, num_nbs[a2]] = idx
        num_nbs[a1] += 1
        num_nbs[a2] += 1
        fbonds[idx] = bond_features(bond)

    #     print('What is core_bonds here?: {}'.format(core_bonds))
    if not testing:
        num_newbonds = max(len(gold_bonds), len(core_bonds)) * 2 + 1  # CC fixed in case where core_bonds isn't large enough
    else:
        num_newbonds = len(core_bonds) * 2 + 1

    new_fbonds = np.zeros(
        (n_bonds + num_newbonds + len(pending_reactant_neighbors), bond_fdim))  # CC added + len(pending_reactant_neighbors)

    new_fbonds[:n_bonds, :] = fbonds
    fbonds = new_fbonds
    batch_fatoms, batch_fbonds, batch_anb, batch_bnb, batch_nbs = [fatoms], [fbonds], [atom_nb], [bond_nb], [
        num_nbs]  # first entry is reactants
    batch_corebias = []


    error_count=0
    for index, edits in enumerate(core_configs):
        # Make new graph sets for every possible candidates
        try:
            fatoms2 = np.copy(fatoms)  # Joonyoung added for chaning atom feature
            atom_nb2 = np.copy(raw_atom_nb)
            bond_nb2 = np.copy(raw_bond_nb)
            num_nbs2 = np.copy(raw_num_nbs)
            fbonds2 = np.copy(fbonds)
            n_bonds2 = n_bonds + 1
            # Add back reactant bonds?
            core_bonds_nobo = [(x, y) for (x, y, t, v, l) in edits if l == 'b']
            for (x, y, t) in pending_reactant_neighbors:
                if (x, y) not in core_bonds_nobo:
                    edits.append((x, y, t, 0.0, 'b'))

            for x, y, t, v, l in edits:  # add new bond features to the "default" reactant ones
                # print(x, y, t, v, l)
                if l == 'b':
                    if t == 0.0: continue
                    atom_nb2[x, num_nbs2[x]] = y
                    atom_nb2[y, num_nbs2[y]] = x
                    bond_nb2[x, num_nbs2[x]] = n_bonds2
                    bond_nb2[y, num_nbs2[y]] = n_bonds2
                    num_nbs2[x] += 1
                    num_nbs2[y] += 1
                    fbonds2[n_bonds2] = onek_encoding_unk(t, [1.0, 2.0, 3.0, 1.5, -1])
                    if (x, y) in ring_bonds:
                        fbonds2[n_bonds2][4] = 1
                    n_bonds2 += 1
                elif l == 'f':
                    f_index = np.where(fatoms2[x] == 1.)[0][4] # The fifth non zero elements corresponds to formal charges
                    fatoms2[x][f_index] = 0.
                    fatoms2[x][f_index + int(t)] = 1
                #                 if f_index+int(t) < 82 or f_index+int(t) > 91:
                #                     pass  #TODO:
                elif l == 'h':
                    #                 print(fatoms2[x])
                    f_index = np.where(fatoms2[x] == 1.)[0][5] #
                    fatoms2[x][f_index] = 0.
                    fatoms2[x][f_index + int(t)] = 1
            batch_fatoms.append(fatoms2)
            batch_fbonds.append(fbonds2)
            batch_anb.append(atom_nb2)
            batch_bnb.append(bond_nb2)
            batch_nbs.append(num_nbs2)
            batch_corebias.append(sum([v for (x, y, t, v, l) in edits]))
        except:
            # print('!'*20, ' Error found')
            # print('rsmiles, ',rsmiles)
            # print('psmiles, ', psmiles)
            # print('core_bonds, ', core_bonds)
            # print('core_h, ', core_h)
            # print('core_f', core_f)
            # print('gold_bonds, ', gold_bonds)
            # print('gold_h, ', gold_h)
            # print('gold_f', gold_f)
            # print('Error causing edits, ', edits)
            # print('\n')
            # print(len(labels))
            if not testing:
                labels = np.delete(labels, index-error_count)
            core_configs = np.delete(core_configs, index-error_count)
            error_count+=1

        #                 print(fatoms2[x])

    # TODO: change atom features for each candidate? Maybe update degree at least

    # TODO: I need to change np.array([fatoms] * n_batch. some atoms can have different features.

    if return_found:
        return (np.array(batch_fatoms), np.array(batch_fbonds), packnb(batch_anb), packnb(batch_bnb),
                np.array(batch_nbs), np.array(batch_corebias), labels), core_configs, found_true
    if not testing:
        return (np.array(batch_fatoms), np.array(batch_fbonds), packnb(batch_anb), packnb(batch_bnb), np.array(batch_nbs),
                np.array(batch_corebias), labels), core_configs
    return (np.array(batch_fatoms), np.array(batch_fbonds), packnb(batch_anb), packnb(batch_bnb), np.array(batch_nbs),
            np.array(batch_corebias)), core_configs


if __name__ == "__main__":
    import gzip
    from WLN.data_loading import convert_detail_data

    val_smiles, val_cand, val_edits = convert_detail_data('../data/test.txt.proc', '../data/test.cbond_detailed.tar')
    ctr = 0
    tot = 0
    tot_found_prefilter = 0
    tot_found = 0
    tot_candidates = 0

    assert len(val_smiles) == len(val_cand) == len(val_edits)

    for i in range(len(val_smiles)):
        ctr += 1
        rsmiles = val_smiles[i].split('>')[0]
        psmiles = val_smiles[i].split('>')[-1]
        gold_bonds = set([tuple(sorted([int(x.split('-')[0])-1, int(x.split('-')[1])-1]) + [float(x.split('-')[2])]) for x in val_edits[i].split(';')])

        if cands:
            # core_bonds = [(int(x.split('-')[0])-1, int(x.split('-')[1])-1, float(x.split('-')[2]), 0.0) for x in cands if x]
            core_bonds = []
            for i in range(0, len(cands), 2):
                x,y,t = cands[i].split('-')
                x,y = tuple(sorted([int(float(x)) - 1, int(float(y)) - 1]))
                core_bonds.append((x,y,float(t),float(cands[i+1])))
        else:
            core_bonds = []

        core_bonds = []

        found_prefilter = True
        for gold_bond in gold_bonds:
            if gold_bond not in core_bonds[:20]:
                found_prefilter = False
                break
        tot_found_prefilter += found_prefilter

        _, core_configs, found_true = smiles2graph(rsmiles, psmiles, core_bonds, gold_bonds, cutoff=1000, core_size=18, kmax=5, return_found=True)

        tot += 1
        tot_found += found_true
        tot_candidates += len(core_configs)

        # Debugging
        if not found_true and found_prefilter:
            print('\nNot found')
            print(rxnsmiles)
            print(gold_bonds)
            print(core_bonds)

        if tot % 10 == 0:
            print('\nAfter {} processed'.format(tot))
            print('Total processed: {}'.format(tot))
            print('Coverage of true product: {}'.format(float(tot_found) / tot))
            print('Average number of cands: {}'.format(float(tot_candidates) / tot))
            print('Coverage from initial cand list, before filters: {}'.format(float(tot_found_prefilter) / tot))


    if tot:
        print('Total processed: {}'.format(tot))
        print('Coverage of true product: {}'.format(float(tot_found)/tot))
        print('Average number of cands: {}'.format(float(tot_candidates)/tot))
