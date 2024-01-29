import converting_graph
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
from rdkit.Chem import AllChem
import itertools
import time
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
import networkx as nx
from elementary_reaction import *

def graph_to_full_reaction(reactions, v=False):
    G=nx.node_link_graph(reactions)
    reaction_id=search_node(G, 'type', match='Reaction')
    spectator_id=search_node(G, 'type', match='spectator')
    reactant_id=search_node(G, 'type', match='reactant')
    product_id=search_node(G, 'type', match='product')
    impurity_id=search_node(G, 'type', match='impurity or byproduct')
    reaction_string=[]
    
    final_reactant=list(set(reactant_id+spectator_id))
    final_product=list(set(product_id))

    react_mol='.'.join([G.nodes[pid]['SMILES']for pid in final_reactant])
    prod_mol='.'.join([G.nodes[pid]['SMILES']for pid in final_product])
    
    try: 
        if Chem.MolFromSmiles(prod_mol).GetNumHeavyAtoms() < 5:
            return None
        rxn_str=make_reaction_string(react_mol, prod_mol)
        rsmi=rxn_str.split('>>')[0] 
        converting_graph.smiles2graph_list([rsmi,])
        reaction_string=rxn_str

        if v:
            print('Reactants are ', final_reactant)
            print('Products are ', final_product)
            print(rxn_str)
    except:
        return None

    return reaction_string

def graph_to_elem_reaction(reactions, v=False):
    G=nx.node_link_graph(reactions)
    reaction_id=search_node(G, 'type', match='Reaction')
    spectator_id=search_node(G, 'type', match='spectator')
    reactant_id=search_node(G, 'type', match='reactant')
    product_id=search_node(G, 'type', match='product')
    impurity_id=search_node(G, 'type', match='impurity or byproduct')
    reaction_string=[]

#     for idx in spectator_id+reactant_id+product_id+impurity_id:
#         atommap_list=[]
#         for a in Chem.MolFromSmiles(G.nodes[idx]['SMILES']).GetAtoms():
#             atommap_list.append(a.GetAtomMapNum())
#         nx.set_node_attributes(G, {idx: atommap_list}, name="atommap_list")
        
    for idx in reaction_id:
        precursor_id=[n for n in G.predecessors(idx)]
        successors_id=[n for n in G.successors(idx)]
        save_id=[]
        used_reactant_id=[]
        produced_impurity_id=[]
        
        if v: 
            print('Reaction {}'.format(idx))
            print('precursor_id: ', precursor_id)
            print('successors_id: ', successors_id)
        
        for pre_id in precursor_id:
            save_id.append(pre_id)
            
            for rid in reactant_id:
                if pre_id in reactant_id:
                    used_reactant_id.append(pre_id)
                    if v: print('For {}, Reactant {} is in precursor'.format(idx, pre_id))
                    break
                else:
                    try:
                        reaction_path=[path for path in nx.all_shortest_paths(G, source=rid, target=pre_id)][0]
                        visiting_reaction=reaction_path[1::2]
                        if not set(reaction_path[1:-1])&set(reactant_id):
                            if v: print('For {}, Found route from reactant {} to precursor {}'.format(idx, rid, pre_id))
                            if v: print('For {}, reaction path is {}'.format(idx, visiting_reaction))

                            if visiting_reaction:
                                for visit_rid in visiting_reaction:
                                    if visit_rid in reaction_id: #To make sure visit_rid is in reaction_id

                                        used_reactant_id.append(rid)

                                        candidate_impurity_id=[n for n in G.successors(visit_rid) if n in impurity_id]
                                        for piid in candidate_impurity_id:
                                            if piid not in produced_impurity_id:
                                                produced_impurity_id.append(piid)
                                        
                    
                    except nx.exception.NetworkXNoPath:
                            continue
        not_used_reactant_id = list(set(reactant_id)- set(used_reactant_id)- set(save_id))
        
        if v: print('reaction {} has precursor {} and not used reactant {}'.format(idx,save_id, not_used_reactant_id))

        final_reactant=list(set(save_id+spectator_id+not_used_reactant_id+produced_impurity_id))
        final_product=list(set(successors_id+spectator_id+not_used_reactant_id+produced_impurity_id))

        react_mol='.'.join([G.nodes[pid]['SMILES']for pid in final_reactant])
        prod_mol='.'.join([G.nodes[pid]['SMILES']for pid in final_product])
        
        try: 
            rxn_str=make_reaction_string(react_mol, prod_mol)
            rsmi=rxn_str.split('>>')[0] 
            converting_graph.smiles2graph_list([rsmi,])
            
#             bond_change=';'.join(['{}-{}-{}'.format(x[0], x[1], x[2]) for x in get_changed_bonds(rxn_str)])
#             hydro_change=';'.join(['{}:{}'.format(x[0], x[1]) for x in get_changed_hydrogens(rxn_str)])
#             formal_change=';'.join(['{}:{}'.format(x[0], x[1]) for x in get_changed_formal(rxn_str)])
#             changes='/'.join([bond_change, hydro_change,formal_change])

#             rxn_str=' '.join([rxn_str, changes])
            reaction_string.append(rxn_str)
            
            
            if v:
                print('Reactants are ', final_reactant)
                print('Products are ', final_product)
                print(rxn_str)
        except:
            print(make_reaction_string(react_mol, prod_mol))
            continue


        
    for idx in product_id:
        for pre_idx in G.predecessors(idx):
            successors_id=[n for n in G.successors(pre_idx)]
            react_mol='.'.join([G.nodes[pid]['SMILES']for pid in list(set(successors_id+spectator_id))])               
            try: 
                rxn_str=make_reaction_string(react_mol, react_mol)
                rsmi=rxn_str.split('>>')[0] 
                converting_graph.smiles2graph_list([rsmi,])
                
#                 bond_change=';'.join(['{}-{}-{}'.format(x[0], x[1], x[2]) for x in get_changed_bonds(rxn_str)])
#                 hydro_change=';'.join(['{}:{}'.format(x[0], x[1]) for x in get_changed_hydrogens(rxn_str)])
#                 formal_change=';'.join(['{}:{}'.format(x[0], x[1]) for x in get_changed_formal(rxn_str)])
#                 changes='/'.join([bond_change, hydro_change,formal_change])
                
#                 rxn_str=' '.join([rxn_str, changes])
                reaction_string.append(rxn_str)
            except:
                print(make_reaction_string(react_mol, prod_mol))
                continue
    return reaction_string




def make_reaction_string(rsmi, psmi):
    rmol = Chem.MolFromSmiles(rsmi,sanitize=False)
        
    atom_map_list=set()
    for atom in rmol.GetAtoms():
        if atom.GetAtomMapNum() not in atom_map_list:
            atom_map_list.add(atom.GetAtomMapNum())
        else:
            return None


    missing = set(range(1, max(atom_map_list) + 1)) - atom_map_list

    #check missing atom mapping number
    if not missing:
        reaction_string='>>'.join([rsmi,psmi])

    else:
        missing=list(sorted(atom_map_list))
        change_dict= {key: value+1 for value, key in enumerate(atom_map_list)}

        for atom in rmol.GetAtoms():
            atom.SetAtomMapNum(change_dict[atom.GetAtomMapNum()])         
        pmol = Chem.MolFromSmiles(psmi,sanitize=False)
        
        for atom in pmol.GetAtoms():
            
            atom.SetAtomMapNum(change_dict[atom.GetAtomMapNum()])  

        react_mol=Chem.MolToSmiles(rmol)
        prod_mol=Chem.MolToSmiles(pmol)
        reaction_string ='>>'.join([react_mol,prod_mol])
        
    return reaction_string



def get_changed_bonds(rxn_smi):
    reactants = Chem.MolFromSmiles(rxn_smi.split('>')[0],sanitize=False)
    products  = Chem.MolFromSmiles(rxn_smi.split('>')[2],sanitize=False)

    conserved_maps = [a.GetAtomMapNum() for a in products.GetAtoms() if a.HasProp('molAtomMapNumber')]
    bond_changes = set() # keep track of bond changes

    # Look at changed bonds
    bonds_prev = {}
    for bond in reactants.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetAtomMapNum(),
             bond.GetEndAtom().GetAtomMapNum()])
        if (nums[0] not in conserved_maps) and (nums[1] not in conserved_maps): continue
        bonds_prev['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
    bonds_new = {}
    for bond in products.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetAtomMapNum(),
             bond.GetEndAtom().GetAtomMapNum()])
        bonds_new['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
    
#     print(bonds_prev)
#     print(bonds_new)
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
