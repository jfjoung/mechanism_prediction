from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
from rdkit.Chem import AllChem
import itertools
import time
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 
import networkx as nx
from matplotlib import pyplot as plt


def run_elementary_reaction(example_reactants,elem_reactions,v=False):
    RDLogger.DisableLog('rdApp.*') 
       
    ## Set the atom mapping with a trick of isotope
    idx=1
    reactant_pool=dict()
    _smiles_list=[]
    for mol in example_reactants:
        start_idx=idx
        for atom in mol.GetAtoms():
            atom.SetIsotope(idx) 
            idx+=1
            
        _smiles_list.append(Chem.MolToSmiles(mol))
        
        #Save molecule in a dictionary
        reactant_pool[mol]={'smiles_w_isotope':Chem.MolToSmiles(mol),   # Using isotope as atom mapping for reaction
                            'atom_mapping':[i for i in range(start_idx,idx)],  #Checking for atom-mapping collision
                            'smiles':Chem.MolToSmiles(remove_atom_map(mol), isomericSmiles=False),  #Plain SMILES string
                           'smiles_w_mapping': Chem.MolToSmiles(isotope_to_atommap(mol), isomericSmiles=False) # Output for generating elementary reaction graph
                           }

    #Reaction occurs
    reaction_pair=[]
    
    for reaction_idx, rxn_smarts_set in enumerate(elem_reactions):
        if v: 
            print('='*50)
            print('This is step number {} '.format(reaction_idx+1))
        if v: print('Total molecules in the reactant pool are {}'.format(len(reactant_pool.keys())))
            
        for rxn_idx, rxn_smarts in enumerate(rxn_smarts_set):
            if v: 
                print('This reaction is {}th of {}th.'.format(rxn_idx+1,reaction_idx+1))
                print('Reaction template is {}'.format(rxn_smarts))
            rxn = AllChem.ReactionFromSmarts(rxn_smarts)
            
            combinations = itertools.permutations(list(itertools.chain([i for i in reactant_pool.keys()])), rxn.GetNumReactantTemplates())

            stime=time.time()
            for combination in combinations:
                
                # Check atom mapping collision
                _reactant_atommap=[i for i in [reactant_pool[mol]['atom_mapping'] for mol in combination]]
                reactant_atommap=[set(i) for i in [reactant_pool[mol]['atom_mapping'] for mol in combination]]
                
                _combination=combination #Saved because mol object is changed when it passes rxn.RunReactants.
                
                if len(reactant_atommap)==2 and reactant_atommap[0]&reactant_atommap[1]:  
                    # There is no three body reactions in the elementary reaction
                    continue
                    
                # Run reaction!
                try:
                    # Use isotope labeling for reaction. Isotope persists over the reaction.
                    combination=[Chem.MolFromSmiles(reactant_pool[mol]['smiles_w_isotope'],sanitize=False)
                                 for mol in combination]                    
                    outcomes = rxn.RunReactants(list(combination))
                except:
                    continue
                    
                if not outcomes:
                    continue
                    
                reactant_smi_atommap=[i for i in [reactant_pool[mol]['smiles_w_mapping'] for mol in _combination]]
                reactant_smi=[i for i in [reactant_pool[mol]['smiles'] for mol in _combination]]
                
                if v:
                    print('\nFor reactants {}'.format(reactant_smi))

                _transient_pairs=list()  # This will contain all reaction pairs that are generated in this reaction
                # There could be several products for one template
                for j, outcome in enumerate(outcomes):
                    if v: print('The {} outcome of {} outcomes'.format(j+1, len(outcomes)))
                    _transient_products=list()
                    product_atommap_list=list()
                    _transient_pair=[reactant_smi]  # This will contain one reaction pair for a single outcome
                    
                    for prod_mol in outcome:
                        try:# Not realistic molecule should be rejected, except aromaticity error.
                            prod_mol.UpdatePropertyCache(strict=False)
                            Chem.SanitizeMol(prod_mol,
                                             Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
                            
                        except:
                            continue
                            
                        new_mol=Chem.MolFromSmiles(Chem.MolToSmiles(prod_mol),sanitize=False)
                        _transient_products.append(Chem.MolToSmiles(remove_atom_map(new_mol,isotope=True), isomericSmiles=False))
                    _transient_pair.append(_transient_products)
                    
                    if _transient_pair not in _transient_pairs:
                        _transient_pairs.append(_transient_pair)  # To prevent saving the reaction of symmetric molecules
                        if v:
                            print('\nFor prodcuts {}'.format(_transient_products))

                        try:
                            _prod_smiles_mol=[]
#                             found=False
                            for m in outcome:                            
                                # Assign new atommap for product
                                if v: print(Chem.MolToSmiles(m))
                                atommap_list=[]
                                for idx, a in enumerate(m.GetAtoms()):
                                    a.SetAtomMapNum(a.GetIsotope())
                                    atommap_list.append(a.GetIsotope())
                                product_atommap_list.append(atommap_list)
                                
                                _prod_smiles_mol.append(Chem.MolToSmiles(m, isomericSmiles=False))
                                if Chem.MolToSmiles(m) not in _smiles_list:  # Collect in the case of not duplicated molecule
                                    found=True
                                    _smiles_list.append(Chem.MolToSmiles(m))
                                    new_mol=Chem.MolFromSmiles(Chem.MolToSmiles(m),sanitize=False)
                                    no_atommap_smi=Chem.MolToSmiles(remove_atom_map(new_mol), isomericSmiles=False)
                                    reactant_pool[m]={'smiles_w_isotope':Chem.MolToSmiles(m),
                                                      'atom_mapping':atommap_list,
                                                      'smiles':no_atommap_smi,
                                                     'smiles_w_mapping': Chem.MolToSmiles(isotope_to_atommap(m), isomericSmiles=False)}
                                    
                                    if v:
                                        print('\nAdded to intermediate for product {}'.format(no_atommap_smi))
                                    
                                
                                else:
                                    continue  
                            if not found:
                                _prod_smiles_mol=[]
                            if reactant_smi_atommap and _prod_smiles_mol:
                                _reaction_pair=(reactant_smi_atommap,
                                                _prod_smiles_mol)

                                reaction_pair.append(_reaction_pair)
                        except Exception as e:
                            if v: print('The error: ',e)
                            continue
                                
                if time.time()-stime > 25:
                    if v:
                        print('Time out!')
                    break
    if v: return reaction_idx+1, [reactant_pool[key]['smiles'] for key in reactant_pool.keys()], reaction_pair, reactant_pool
    else: return reaction_idx+1, [reactant_pool[key]['smiles'] for key in reactant_pool.keys()], reaction_pair
    
def reaction_graph(example_rxn_smiles,elem_reactions, v=False, full_data=False): 
    RDLogger.DisableLog('rdApp.*') 
#     try:
    try:
        rxn = AllChem.ReactionFromSmarts(example_rxn_smiles)

        reactants, agents, products = [mols_from_smiles_list(x) for x in 
                                       [mols.split('.') for mols in example_rxn_smiles.split('>')]]
        [Chem.SanitizeMol(mol) for mol in reactants + agents + products] #Check sanitization

        #Generate the mol object again to aviod sanitization.
        #Some molecules should not be sanitized for comparing molecule in the reaction mixture
        reactants, agents, products = [mols_from_smiles_list(x) for x in 
                                       [mols.split('.') for mols in example_rxn_smiles.split('>')]]


    except Exception as e:
        if v: print('Failed to sanitize ', e)
        return None

    try: 
        # Make every elementary reactions
        num_reacted,reaction_mixture, reaction_pair=run_elementary_reaction(reactants+agents, elem_reactions,v=False)
        if v: print('reaction mixture has {} molecules'.format(len(reaction_mixture)))
        if v: print('There are {} reaction pairs'.format(len(reaction_pair)))
    except Exception as e:
        if v: print('failed reaction', e)
        return None
    try:
        # Removing all atom mapping in Pistachio dataset
        product_smiles = [Chem.MolToSmiles(remove_atom_map(mol), isomericSmiles=False) for mol in products]
        reactant_smiles = [Chem.MolToSmiles(remove_atom_map(mol), isomericSmiles=False) for mol in reactants]+[Chem.MolToSmiles(remove_atom_map(mol), isomericSmiles=False) for mol in agents]
        for reactant in reactant_smiles:
            while reactant in product_smiles:
                product_smiles.remove(reactant)
    except Exception as e:
        if v: print('Failed to remove atom mapping  ', e)
        return None

    #Find real product is formed    
    real_products=set(reaction_mixture).intersection(set(product_smiles))
    if v: print('The real products are {}'.format(set(product_smiles)))
    if v: print('Found real product {}'.format(real_products))
    if real_products:
        G=reaction_pair_to_graph(reaction_pair)

        reaction_id=[]
        removing_id=[]
        nx.set_node_attributes(G, 'intermediate', "type")   
        for nid, attrs in G.nodes.data():
            if 'Reaction' in attrs.get('SMILES'):
                nx.set_node_attributes(G, {nid: 'Reaction'}, name="type")
                del G.nodes[nid]['SMILES']
            else: 
                try:
                    nx.set_node_attributes(G, 
                                           {nid: Chem.MolToSmiles(remove_atom_map(Chem.MolFromSmiles(attrs.get('SMILES'),sanitize=False), isotope=True), isomericSmiles=False)}
                                           , name="unmapped_SMILES")
                except:
                    removing_id.append(nid)  # Not SMILES-convertable mlecule should be deleted       
                    continue
            if not [n for n in G.neighbors(nid)]:
                # Molecules that no longer reacts is assgined as impurity or byproduct
                try: 
                    if G.nodes[nid]['SMILES']:
                        nx.set_node_attributes(G, {nid: 'impurity or byproduct'}, name="type") 
                except:
                    removing_id.append(nid)


        if removing_id:
            for nid in removing_id:
                G.remove_nodes_from(removing_id)
        spectators=[]

        for reactant in reactant_smiles:          
            for real_product in real_products:
                try:
                    reactant_id=search_node(G, 'unmapped_SMILES', match=reactant,check=True)
                    if reactant_id: 
                        product_id=search_node(G, 'unmapped_SMILES', match=real_product)
                        for rid in range(len(reactant_id)):
                            if v: print('reactant id is {} and product id is {}'.format(reactant_id[rid],product_id[0]))
                            if nx.all_simple_paths(G, source=reactant_id[rid], target=product_id[0]):
                                for path in nx.all_simple_paths(G, source=reactant_id[rid], target=product_id[0]):
                                    reaction_id.append(path[1::2])    
                                nx.set_node_attributes(G, {reactant_id[rid]: "reactant"}, name="type")
                                nx.set_node_attributes(G, {product_id[0]: "product"}, name="type")
                    else: 
                        spectators.append(reactant)
                        if v: print('{} is a spectator.'.format(reactant))
                except Exception as e:
                    spectators.append(reactant)
                    if v: print('{} is a spectator.'.format(reactant))
                    pass


        reaction_id=list(set([y for x in reaction_id for y in x]))

        if v: print('reaction id is ', reaction_id)

        path_node=[]
        for reaction_node in reaction_id:
            path_node.append(reaction_node)
            neighbor_node=[n for n in nx.all_neighbors(G, reaction_node)]
            for nn in neighbor_node:
                path_node.append(nn)

        G_new=G
        for nid in [x for x,y in G_new.nodes(data=True) if y['type']=='impurity or byproduct']:
            try:
                check_smiles=G_new.nodes[nid]['unmapped_SMILES']
            except Exception as e:
                pass
            mol_node=[x for x,y in G_new.nodes(data=True) if y['type']!='Reaction' and y['type']!='impurity or byproduct']
            for mol_id in mol_node:
                if mol_id != nid:
                    if G_new.nodes[mol_id]['unmapped_SMILES'] == check_smiles:
                        if G_new.nodes[mol_id]['SMILES'] == G_new.nodes[nid]['SMILES']:
                            try:
                                predecessors_id=[n for n in G_new.predecessors(nid)]
                                G_new.remove_node(nid)
                                for predecessor_id in predecessors_id:
                                    G_new.add_edge(predecessor_id, mol_id)
                            except: pass 


        H = nx.DiGraph(G_new.subgraph(path_node))
        L = nx.DiGraph(G.subgraph(path_node)) 
        
        for i in range(num_reacted):
            for nid in [x for x,y in H.nodes(data=True) if y['type']=='intermediate']:
#                 if not [n for n in H.successors(nid)]: 
#                     for nnid in [n for n in H.predecessors(nid)]:
#                         H.remove_nodes_from([n for n in H.successors(nnid)]) 
#                         H.remove_node(nnid)               
#                 elif not [n for n in H.predecessors(nid)]:
#                     H.remove_nodes_from([n for n in H.successors(nid)])
#                     H.remove_node(nid)        
                if not [n for n in H.successors(nid)]: 
                    H.remove_nodes_from([n for n in H.predecessors(nid)])
                    H.remove_node(nid)               
                elif not [n for n in H.predecessors(nid)]:
                    H.remove_nodes_from([n for n in H.successors(nid)])
                    H.remove_node(nid)             
                        
            for nid in [x for x,y in H.nodes(data=True) if y['type']=='impurity or byproduct']:
                if not [n for n in H.predecessors(nid)]:
                    H.remove_nodes_from([n for n in H.successors(nid)])
                    H.remove_node(nid)    

        for nid in [x for x,y in H.nodes(data=True) if y['type']=='reactant']:
            if not [n for n in H.successors(nid)]: 
                nx.set_node_attributes(H, {nid: 'spectator'}, name="type")


        reactant_product_pairs=list()
        for nid in [x for x,y in H.nodes(data=True) if y['type']=='Reaction']:
            reactant_product_pair=[[n for n in H.predecessors(nid)],[n for n in H.successors(nid)]]
            if reactant_product_pair not in reactant_product_pairs:
                reactant_product_pairs.append(reactant_product_pair)
            else:
                H.remove_node(nid)      

        atom_map=0
        for nid in [x for x,y in H.nodes(data=True) if y['type']=='reactant' or y['type']=='spectator' ]:
            mol = Chem.MolFromSmiles(H.nodes[nid]['SMILES'],sanitize=False)
            for atom in mol.GetAtoms():
                if atom_map < atom.GetAtomMapNum():
                    atom_map=atom.GetAtomMapNum()

        if v: print('Max atom mapping for reactants is ', atom_map)

        atom_map+=1
        mapped_spectator=[]
        for smi in spectators:
            spectator_mol=Chem.MolFromSmiles(smi,sanitize=False)
            for atom in spectator_mol.GetAtoms():
                atom.SetAtomMapNum(atom_map)
                atom_map+=1
            if v: print('{} is a mapped spectator.'.format(Chem.MolToSmiles(spectator_mol)))
            mapped_spectator.append(Chem.MolToSmiles(spectator_mol))

        H_new=nx.convert_node_labels_to_integers(H)
        G_len=len(G.nodes.data())
        H_new_len=len(H_new.nodes.data())
        for idx, spectator in enumerate(mapped_spectator):
            G.add_node(G_len+idx, SMILES=spectator, type='spectator')
            nx.set_node_attributes(G, {G_len+idx: Chem.MolToSmiles(remove_atom_map(Chem.MolFromSmiles(spectator,sanitize=False)))}, name="unmapped_SMILES")
            H_new.add_node(H_new_len+idx, SMILES=spectator, type='spectator')
            nx.set_node_attributes(H_new, {H_new_len+idx: Chem.MolToSmiles(remove_atom_map(Chem.MolFromSmiles(spectator,sanitize=False)))}, name="unmapped_SMILES")

        if full_data: return reaction_pair, nx.node_link_data(G), nx.node_link_data(H), nx.node_link_data(H_new)
        else: return nx.node_link_data(H_new)
    else:
        return None
#     except:
#         return None



def reaction_graph_old(example_rxn_smiles,elem_reactions, v=False, full_data=False): 
    RDLogger.DisableLog('rdApp.*') 
    try:
        rxn = AllChem.ReactionFromSmarts(example_rxn_smiles)

        reactants, agents, products = [mols_from_smiles_list(x) for x in 
                                       [mols.split('.') for mols in example_rxn_smiles.split('>')]]
        [Chem.SanitizeMol(mol) for mol in reactants + agents + products] #Check sanitization
        
        #Generate the mol object again to aviod sanitization.
        #Some molecules should not be sanitized for comparing molecule in the reaction mixture
        reactants, agents, products = [mols_from_smiles_list(x) for x in 
                                       [mols.split('.') for mols in example_rxn_smiles.split('>')]]
    

    except Exception as e:
        if v: print('Failed to sanitize ', e)
        return None

    try: 
        # Make every elementary reactions
        num_reacted,reaction_mixture, reaction_pair=run_elementary_reaction(reactants+agents, elem_reactions,v=False)
        if v: print('reaction mixture has {} molecules'.format(len(reaction_mixture)))
        if v: print('There are {} reaction pairs'.format(len(reaction_pair)))
    except Exception as e:
        if v: print('failed reaction', e)
        return None
    try:
        # Removing all atom mapping in Pistachio dataset
        product_smiles = [Chem.MolToSmiles(remove_atom_map(mol), isomericSmiles=False) for mol in products]
        reactant_smiles = [Chem.MolToSmiles(remove_atom_map(mol), isomericSmiles=False) for mol in reactants]+[Chem.MolToSmiles(remove_atom_map(mol), isomericSmiles=False) for mol in agents]
        for reactant in reactant_smiles:
            while reactant in product_smiles:
                product_smiles.remove(reactant)
    except Exception as e:
        if v: print('Failed to remove atom mapping  ', e)
        return None

    #Find real product is formed    
    real_products=set(reaction_mixture).intersection(set(product_smiles))
    if v: print('The real products are {}'.format(set(product_smiles)))
    if v: print('Found real product {}'.format(real_products))
    if real_products:
        G=reaction_pair_to_graph(reaction_pair)

        reaction_id=[]
        removing_id=[]
        nx.set_node_attributes(G, 'intermediate', "type")   
        for nid, attrs in G.nodes.data():
            if 'Reaction' in attrs.get('SMILES'):
                nx.set_node_attributes(G, {nid: 'Reaction'}, name="type")
                del G.nodes[nid]['SMILES']
            else: 
                try:
                    nx.set_node_attributes(G, 
                                           {nid: Chem.MolToSmiles(remove_atom_map(Chem.MolFromSmiles(attrs.get('SMILES'),sanitize=False), isotope=True), isomericSmiles=False)}
                                           , name="unmapped_SMILES")
                except:
                    removing_id.append(nid)  # Not SMILES-convertable mlecule should be deleted       
                    continue
            if not [n for n in G.neighbors(nid)]:
                # Molecules that no longer reacts is assgined as impurity or byproduct
                try: 
                    if G.nodes[nid]['SMILES']:
                        nx.set_node_attributes(G, {nid: 'impurity or byproduct'}, name="type") 
                except:
                    removing_id.append(nid)
                    

        if removing_id:
            for nid in removing_id:
                G.remove_nodes_from(removing_id)
        spectators=[]

        for reactant in reactant_smiles:                
            for real_product in real_products:
                try:
                    reactant_id=search_node(G, 'unmapped_SMILES', match=reactant,check=True)
                    product_id=search_node(G, 'unmapped_SMILES', match=real_product)
                    if v: print('reactant id is {} and product id is {}'.format(reactant_id[0],product_id[0]))

                    if nx.all_simple_paths(G, source=reactant_id[0], target=product_id[0]):
                        for path in nx.all_simple_paths(G, source=reactant_id[0], target=product_id[0]):
                            reaction_id.append(path[1::2])    
                        nx.set_node_attributes(G, {reactant_id[0]: "reactant"}, name="type")
                        nx.set_node_attributes(G, {product_id[0]: "product"}, name="type")

                except Exception as e:
                    spectators.append(reactant)
                    if v: print('{} is a spectator.'.format(reactant))
                    pass
                

        reaction_id=list(set([y for x in reaction_id for y in x]))

        if v: print('reaction id is ', reaction_id)

        path_node=[]
        for reaction_node in reaction_id:
            path_node.append(reaction_node)
            neighbor_node=[n for n in nx.all_neighbors(G, reaction_node)]
            for nn in neighbor_node:
                path_node.append(nn)

        G_new=G
        for nid in [x for x,y in G_new.nodes(data=True) if y['type']=='impurity or byproduct']:
            try:
                check_smiles=G_new.nodes[nid]['unmapped_SMILES']
            except Exception as e:
                pass
            mol_node=[x for x,y in G_new.nodes(data=True) if y['type']!='Reaction' and y['type']!='impurity or byproduct']
            for mol_id in mol_node:
                if mol_id != nid:
                    if G_new.nodes[mol_id]['unmapped_SMILES'] == check_smiles:
                        if G_new.nodes[mol_id]['SMILES'] == G_new.nodes[nid]['SMILES']:
                            try:
                                predecessors_id=[n for n in G_new.predecessors(nid)]
                                G_new.remove_node(nid)
                                for predecessor_id in predecessors_id:
                                    G_new.add_edge(predecessor_id, mol_id)
                            except: pass 
        
        
        H = nx.DiGraph(G_new.subgraph(path_node))
        L = nx.DiGraph(G.subgraph(path_node))  
        
        for i in range(num_reacted):
            for nid in [x for x,y in H.nodes(data=True) if y['type']=='intermediate']:
                if not [n for n in H.successors(nid)]: 
                    H.remove_nodes_from([n for n in H.predecessors(nid)])
                    H.remove_node(nid)               
                elif not [n for n in H.predecessors(nid)]:
                    H.remove_nodes_from([n for n in H.successors(nid)])
                    H.remove_node(nid)             
            for nid in [x for x,y in H.nodes(data=True) if y['type']=='impurity or byproduct']:
                if not [n for n in H.predecessors(nid)]:
                    H.remove_nodes_from([n for n in H.successors(nid)])
                    H.remove_node(nid)    

        for nid in [x for x,y in H.nodes(data=True) if y['type']=='reactant']:
            if not [n for n in H.successors(nid)]: 
                nx.set_node_attributes(H, {nid: 'spectator'}, name="type")


        reactant_product_pairs=list()
        for nid in [x for x,y in H.nodes(data=True) if y['type']=='Reaction']:
            reactant_product_pair=[[n for n in H.predecessors(nid)],[n for n in H.successors(nid)]]
            if reactant_product_pair not in reactant_product_pairs:
                reactant_product_pairs.append(reactant_product_pair)
            else:
                H.remove_node(nid)      

        atom_map=0
        for nid in [x for x,y in H.nodes(data=True) if y['type']=='reactant' or y['type']=='spectator' ]:
            mol = Chem.MolFromSmiles(H.nodes[nid]['SMILES'],sanitize=False)
            for atom in mol.GetAtoms():
                if atom_map < atom.GetAtomMapNum():
                    atom_map=atom.GetAtomMapNum()

        if v: print('Max atom mapping for reactants is ', atom_map)
        atom_map+=1
        mapped_spectator=[]
        for smi in spectators:
            spectator_mol=Chem.MolFromSmiles(smi,sanitize=False)
            for atom in spectator_mol.GetAtoms():
                atom.SetAtomMapNum(atom_map)
                atom_map+=1
            if v: print('{} is a mapped spectator.'.format(Chem.MolToSmiles(spectator_mol)))
            mapped_spectator.append(Chem.MolToSmiles(spectator_mol))

        H_new=nx.convert_node_labels_to_integers(H)
        G_len=len(G.nodes.data())
        H_new_len=len(H_new.nodes.data())
        for idx, spectator in enumerate(mapped_spectator):
            G.add_node(G_len+idx, SMILES=spectator, type='spectator')
            nx.set_node_attributes(G, {G_len+idx: Chem.MolToSmiles(remove_atom_map(Chem.MolFromSmiles(spectator,sanitize=False)))}, name="unmapped_SMILES")
            H_new.add_node(H_new_len+idx, SMILES=spectator, type='spectator')
            nx.set_node_attributes(H_new, {H_new_len+idx: Chem.MolToSmiles(remove_atom_map(Chem.MolFromSmiles(spectator,sanitize=False)))}, name="unmapped_SMILES")

        if full_data: return reaction_pair, nx.node_link_data(G), nx.node_link_data(H), nx.node_link_data(H_new)
        else: return nx.node_link_data(H_new)
    else:
        return None





def remove_atom_map(mol, isotope=False):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if isotope: atom.SetIsotope(0)
    return mol
    

def isotope_to_atommap(mol):
    for idx, a in enumerate(mol.GetAtoms()):
        a.SetAtomMapNum(a.GetIsotope())
        a.SetIsotope(0)
    return mol


def search_node(G, key_to_find, match=None, include=None, check=None):
    nids=[]
    if match:
        for nid, attrs in G.nodes.data():
            if check:
                if attrs.get('type')!='impurity or byproduct':
                    if attrs.get(key_to_find) == match:
                        nids.append(nid)
            else:
                if attrs.get(key_to_find) == match:
                    nids.append(nid)
    if include:
        for nid, attrs in G.nodes.data():
            if attrs.get('type')!='impurity or byproduct':
                if include in attrs.get(key_to_find):
                    nids.append(nid)
                
    if not match and not include:
        for nid, attrs in G.nodes.data():
            if attrs.get(key_to_find):
                nids.append(nid)
    return nids


def reaction_pair_to_graph(reaction_pair):
    graphs_basis=[]
    for react_idx, reaction in enumerate(reaction_pair):

        for smiles in enumerate(reaction[0]):
            _reactant=(smiles[1], 'Reaction '+str(react_idx))
            graphs_basis.append(_reactant)
        for idx, smiles in enumerate(reaction[1]):
            _reactant=('Reaction '+str(react_idx), smiles)
            graphs_basis.append(_reactant)
    G = nx.DiGraph(graphs_basis)
    G=nx.convert_node_labels_to_integers(G, label_attribute='SMILES')
    nx.node_link_data(G)
                      
    return G


def topo_pos(G):
    """Display in topological order, with simple offsetting for legibility"""
    pos_dict = {}
    for i, node_list in enumerate(nx.topological_generations(G)):
        x_offset = len(node_list) / 2
        y_offset = 0.0
        for j, name in enumerate(node_list):
            pos_dict[name] = (j - x_offset, -i + j * y_offset)

    return pos_dict


def draw_reaction_graph(G, size=[5,5]):

    G=nx.node_link_graph(G)
    try:
        posit =  topo_pos(G)
    except:
        posit =  nx.kamada_kawai_layout(G, scale=2)
    plt.figure(3,figsize=(size[0],size[1]))

    mol_list=[]
    reaction_list=[]
    reactant_list=[]
    product_list=[]
    by_product_list=[]
    spectator_list=[]
    for nid, attrs in G.nodes.data():
        if attrs.get('SMILES'):
            if attrs.get('type') == 'reactant':
                reactant_list.append(nid)
            elif attrs.get('type') == 'product':
                product_list.append(nid)
            elif  attrs.get('type') == 'impurity or byproduct':
                by_product_list.append(nid)
            elif  attrs.get('type') == 'spectator':
                spectator_list.append(nid)
            else:
                mol_list.append(nid)
        else: reaction_list.append(nid)

    nx.draw(G , posit, with_labels = True)
    nx.draw_networkx_nodes(G, posit, nodelist=mol_list, node_color="tab:red")
    nx.draw_networkx_nodes(G, posit, nodelist=reaction_list, node_shape="s",node_color="tab:green")
    nx.draw_networkx_nodes(G, posit, nodelist=reactant_list,  node_color="tab:blue")
    nx.draw_networkx_nodes(G, posit, nodelist=by_product_list, node_color="tab:cyan")
    nx.draw_networkx_nodes(G, posit, nodelist=product_list, node_color="magenta")
    nx.draw_networkx_nodes(G, posit, nodelist=spectator_list, node_color="tab:purple")

    
def mols_from_smiles_list(all_smiles):
	'''Given a list of smiles strings, this function creates rdkit
	molecules'''
	mols = []
	for smiles in all_smiles:
		if not smiles: continue
		mols.append(Chem.MolFromSmiles(smiles,sanitize=False))
	return mols

def mol_list_to_inchi(mols):
	'''List of RDKit molecules to InChI string separated by ++'''
	inchis = [Chem.MolToInchi(mol) for mol in mols]
	return ' ++ '.join(sorted(inchis))


def split_molecules(reaction_smiles):
    reactants, agents, products = [mols_from_smiles_list(x) for x in 
                                            [mols.split('.') for mols in reaction_smiles.split('>')]]
    return reactants, agents, products


def pickle_to_example_doc(cleaned_entry, idx):
    example_doc={}
    
    # to remove some words after ' |'
    example_rxn_smiles=cleaned_entry['smiles']
    if ' |' in example_rxn_smiles:
        example_rxn_smiles=example_rxn_smiles.rsplit(' |')[0]
    example_doc['reaction_smiles']=example_rxn_smiles
    example_doc['_id']=idx
    example_doc['title']=cleaned_entry['title']
    
    return example_doc


def convert_to_preferred_format(sec):
    year=sec //(24 * 3600*365)
    sec %= (24 * 3600*365)
    day=sec //(24 * 3600)
    sec %= (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    minute = sec // 60
    sec %= 60
    
    if year!=0:
        return "%d years %02d days %02d hours %02d mins %02d sec" % (year,day,hour, minute, sec) 
    if day!=0:
        return "%02d days %02d hours %02d mins %02d sec" % (day,hour, minute, sec) 
    if hour != 0:
        return "%02d hours %02d mins %02d sec" % (hour, minute, sec) 
    if minute !=0:
        return "%02d mins %02d sec" % (minute, sec) 
    return "{:3f} sec".format(sec) 