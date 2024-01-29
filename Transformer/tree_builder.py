import numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

gamma=0.5

class Node:
    def __init__(self, val):
        self.prob = val[0]
        self.smiles=val[1]
        self.plain_smiles=None
        self.childrens = []
        self.parent = None
        self.termination = False
        self.depth = 0
        self.weighted_rank= 1*(gamma**self.depth)
        self.rank=1
        self.get_plain_smiles()

    def get_plain_smiles(self):
        try:
            plain=''.join(self.smiles.decode('utf-8').strip().split())
        except:
            plain=''.join(str(self.smiles).strip().split())
        self.plain_smiles=plain

    def add(self, lst):
        ## If the smiles is not valid, then it should not be added at first.
        plain_smiles= [''.join(str(item[1]).strip().split()) for item in lst]

        new_lst = []

        idx=1 # this is for the rank
        for item, smiles in zip(lst, plain_smiles):
            try:
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                if not mol:
                    continue
                node = Node(item)
                node.parent = self
                self.binarize(node)
                rxn_path = self.get_rxnpath(rxn_path=[])
                if node.check_duple(rxn_path) and not self.termination:
                    self.childrens.append(node)
                node.rank = idx
                node.depth = self.depth + 1
                node.weighted_rank = self.weighted_rank +node.rank*(gamma**node.depth)
                idx += 1

            except Exception as e:
                pass


    def get_rxnpath(self,rxn_path=[]):
        rxn_path.append(self.smiles)
        if self.parent:
            self.parent.get_rxnpath(rxn_path)
        return rxn_path

    def check_duple(self,rxn_path):
        if len(rxn_path) == 1 and rxn_path[0] == self.smiles:  #It means there is no reaction
            return False
        elif self.smiles == rxn_path[0]:  # This means the reaction is over
            self.termination = True
            return True
        elif self.smiles in rxn_path: # This means the reaction occurs backward
            return False
        else: return True


    def __str__(self):
        return "Node smiles of {} with rank {}, {}, {}".format( self.plain_smiles,self.rank,self.weighted_rank, self.termination)

    def binarize(self, node):
        node.smiles+='\n'
        node.smiles=node.smiles.encode('utf-8')

    def get_leaf_nodes(self):
        leafs=[]
        self._collect_leaf_nodes(leafs)
        return leafs

    def _collect_leaf_nodes(self, leafs):
        if len(self.childrens) == 0:
            leafs.append(self)
        else:
            for n in self.childrens:
                n._collect_leaf_nodes(leafs)

    def get_topK_leafs(self, k=3):
        terminated_leafs = []
        unterminated_leafs = []
        leafs = self.get_leaf_nodes()
        sorted_leafs= sorted(leafs, key=lambda node: node.weighted_rank, reverse=False)[:20]

        for node in sorted_leafs:
            if node.plain_smiles:
                if node.termination:
                    if node.get_path():
                        terminated_leafs.append(node)
                else:
                    unterminated_leafs.append(node)
        top_k_terminated_leafs=terminated_leafs[:k]
        top_k_unterminated_leafs=unterminated_leafs[:(k-len(top_k_terminated_leafs))]
        return top_k_terminated_leafs, top_k_unterminated_leafs




    def get_path(self):
        path=[]
        self._get_parents(path)
        # for nodes11 in path:
        #     print(nodes11)
        # print(self.check_bond_change())
        if not self.check_bond_change():
            return path

    def _get_parents(self, path):
        path.insert(0, self)
        if self.parent:
            self.parent._get_parents(path)


    def check_bond_change(self):
        if not self.parent:
            return False
        child_smiles=self.plain_smiles
        parent_smiles=self.parent.plain_smiles
        if compare_smiles_structure(child_smiles, parent_smiles):
            self.parent.check_bond_change()
        else: return True

def simplify_molecule(mol):
    # Create a copy of the molecule
    mol_copy = Chem.RWMol(mol)
    mol_copy.UpdatePropertyCache(strict=False)

    for atom in mol_copy.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)

    return mol_copy


def compare_smiles_structure(smiles1, smiles2):  #TODO: Here is the problem!
    # Parse the SMILES strings and generate molecular graphs
    try:
        mol1 = Chem.MolFromSmiles(smiles1, sanitize=False)
        mol2 = Chem.MolFromSmiles(smiles2, sanitize=False)
    except:
        return True
    # Ensure both SMILES strings were successfully parsed
    if mol1 is None or mol2 is None:
        return True

    # Simplify the molecular graphs (remove formal charges and hydrogens)
    try:
        mol1 = simplify_molecule(mol1)
        mol2 = simplify_molecule(mol2)
    except:
        return True
    # Compare the simplified molecular graphs
    return Chem.MolToSmiles(mol1) == Chem.MolToSmiles(mol2)

def min_max(lst):
    min_val = min(lst)
    max_val = max(lst)
    softmax_values = []
    normalized_values = []
    for item in lst:
        normalized_value = (item - min_val) / (max_val - min_val + 1e-10)
        normalized_values.append(normalized_value)

    total_value = sum(np.exp(item) for item in normalized_values)
    for item, norm_value in zip(lst, normalized_values):
        softmax_value = np.exp(norm_value) / total_value
        softmax_values.append(softmax_value)
    return np.log(softmax_values)

def print_tree(node, depth=0):
    print("  " * depth + str(node))
    for child in node.childrens[:9]:
        print_tree(child, depth + 1)
