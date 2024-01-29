from elementary_reaction_template import *

class_reaction_templates={
    'Carboxylic acid + amine condensation':
    {
    'Reagent': [['[N]=[C]=[N]']],
    'Reaction type':['Condensation using DCC', 'Condensation without catalyst'],
    'Templates': [COOH_amine_condense_cat,COOH_amine_condense, ]
    },
    'N-Boc deprotection':
    {
    'Reagent': [['[O;H1;-1]'],['[Li,Na,K][O;H1]'],['[H2O]']],
    'Reaction type':[ 'Deprotection with OH-', 'Deptrotection with Alkali hydroxide','Deprotection with water',  'Non aqueous solution'],
    'Templates': [NBoc_deprotection_OH,NBoc_deprotection_OH, NBoc_deprotection_aq, NBoc_deprotection_nonaq]
    },
    
    'CO2H-Me deprotection':
    {
    'Reagent': [['[O;H1;-1]'],['[Li,Na,K][O;H1]'],['[H2O]'],],
    'Reaction type':[ 'Deprotection with OH-', 'Deptrotection with Alkali hydroxide', 'Deprotection with water', 'Missing agents'],
    'Templates': [Alkyl_ester_deprotection_OH,Alkyl_ester_deprotection_OH,Alkyl_ester_deprotection_aq, None]
    },
    'CO2H-Et deprotection':
    {
    'Reagent': [['[O;H1;-1]'],['[Li,Na,K][O;H1]'],['[H2O]'],],
    'Reaction type':[ 'Deprotection with OH-', 'Deptrotection with Alkali hydroxide', 'Deprotection with water', 'Missing agents'],
    'Templates': [Alkyl_ester_deprotection_OH,Alkyl_ester_deprotection_OH,Alkyl_ester_deprotection_aq, None]
    },
    'CO2H-tBu deprotection':
    {
    'Reagent': [['[O;H1;-1]'],['[Li,Na,K][O;H1]'],['[H2O]'],],
    'Reaction type':[ 'Deprotection with OH-', 'Deptrotection with Alkali hydroxide', 'Deprotection with water', 'Missing agents'],
    'Templates': [Alkyl_ester_deprotection_OH,Alkyl_ester_deprotection_OH,Alkyl_ester_deprotection_aq, None]
    },
    'Amide Schotten-Baumann':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Amide_SchottenBaumann]
    },
    
    'Sulfonamide Schotten-Baumann':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Amide_SchottenBaumann]
    },
    
    'Sulfonic ester Schotten-Baumann':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Amide_SchottenBaumann]
    },
    
    'Carbamate Schotten-Baumann':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Amide_SchottenBaumann]
    },
    
    'Ester Schotten-Baumann':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Amide_SchottenBaumann]
    },
    
    'Williamson ether synthesis':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Willamson_ether]
    },
    
    'Bromo N-alkylation':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Halo_N_alkyl]
    },
    'Chloro N-alkylation':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Halo_N_alkyl]
    },
    'Iodo N-alkylation':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Halo_N_alkyl]
    },
    'Bromo Suzuki coupling':
    {
    'Reagent': [['[Pd;D0,D1,D2]'], ['[Pd;D3,D4]'], ],
    'Reaction type':['Reaction with element Pd', 'Reaction with Pd-ligand', 'Unknown'],
    'Templates': [Suzuki_coupling, [Pd_leaving_ligands]+Suzuki_coupling, None]
    },
    'Chloro Suzuki coupling':
    {
    'Reagent': [['[Pd;D0,D1,D2]'], ['[Pd;D3,D4]'], ],
    'Reaction type':['Reaction with element Pd', 'Reaction with Pd-ligand', 'Unknown'],
    'Templates': [Suzuki_coupling, [Pd_leaving_ligands]+Suzuki_coupling, None]
    },
    
    'Iodo Suzuki coupling':
    {
    'Reagent': [['[Pd;D0,D1,D2]'], ['[Pd;D3,D4]'], ],
    'Reaction type':['Reaction with element Pd', 'Reaction with Pd-ligand', 'Unknown'],
    'Templates': [Suzuki_coupling, [Pd_leaving_ligands]+Suzuki_coupling, None]
    },
    
    'SNAr ether synthesis':
    {
    'Reagent': [],
    'Reaction type':['reaction with alcohol group'],
    'Templates': [SNAr_ether]
    },
    
    'Aldehyde reductive amination':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [CHO_amination]
    },
    
    'Bromination':
    {
    'Reagent': [['O=C-[O;D1]','BrBr'],['O=C-[C;!H0]','BrBr'],['Br[Fe](Br)Br'],['BrN']],
    'Reaction type':['Bromination on benzene', 'Bromination on alpha ketone', 'Using FeBr3', 'Using NBS', 'Unidentified'],
    'Templates': [Br2_AcOH_benzene, Br2_AcOH_ketone, FeBr3, NBS, None]
    },
    
    'N-Boc protection':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [NBoc_Protection]
    },  
    
    'Carboxylic ester + amine reaction':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [COOH_amine_condense]
    },  
    'Carboxy to carbamoyl':
    {
    'Reagent': [['[N]=[C]=[N]'],['[NH1;D0]'],],
    'Reaction type':['Condensation using DCC','Error in reactants' ,'Condensation without catalyst'],
    'Templates': [[COOH_amine_condense_a]+COOH_amine_condense_cat,None,[COOH_amine_condense_a]+COOH_amine_condense]
    },  
    
    'Esterification':
    {
    'Reagent': [['[N]=[C]=[N]'],['[C](=[O])-[O;H1]']],
    'Reaction type':['Condensation using DCC','Esterification', 'Willamson ether synthesis type', ],
    'Templates': [COOH_amine_condense_cat,Esterification,Willamson_ether, ]
    },  
    
    'Methoxy to hydroxy':
    {
    'Reagent': [['[Cl,Br][B]([Cl,Br])[Cl,Br]'],['[Cl,Br,I;H1]']],
    'Reaction type':['Demethylation uisng BBr3','Ether acidic cleavage', 'Unidentified', ],
    'Templates': [Demethylation_BBr3,Ether_acidic_cleavage,None ]
    },  
    'Ester to alcohol reduction':
    {
    'Reagent': [],
    'Reaction type':['Ester reduction using hydride'],
    'Templates': [Ester_reduction_hydride ]
    },  
    
    'Bromo Buchwald-Hartwig amination':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Buchwald_Hartwig ]
    },  
    'Chloro Buchwald-Hartwig amination':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Buchwald_Hartwig ]
    },  
    'Iodo Buchwald-Hartwig amination':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Buchwald_Hartwig ]
    },  
    'Mitsunobu aryl ether synthesis':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Mitsunobu]
    },  
    'Mitsunobu imide reaction':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Mitsunobu]
    },  
    'Mitsunobu sulfonamide reaction':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Mitsunobu]
    },  
    'Mitsunobu ester synthesis':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Mitsunobu]
    },  
    
    'Ketone to alcohol reduction':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Ketone_Reduction_hydride]
    },  
    'O-Bn deprotection':
    {
    'Reagent': [['[Pd]'], ['[Cl,Br][B]([Cl,Br])[Cl,Br]']],
    'Reaction type':['Pd with H2', 'Deprotection uisng BBr3', 'Unidentified'],
    'Templates': [Pd_H2, Demethylation_BBr3]
    },  
    'Isocyanate + amine urea coupling':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Isocyanate_transformation]
    },  
    'Isocyanate + alcohol reaction':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Isocyanate_transformation]
    },  
    'Ketone reductive amination':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Ketone_amination_hydride]
    },  
    'Sulfanyl to sulfonyl':
    {
    'Reagent': [['[O]-[O]']],
    'Reaction type':['Reaction using peroxide', 'Unidentified'],
    'Templates': [Sulfanyl_oxidation_perodxide, None]
    },  
    
    'N-Cbz deprotection':
    {
    'Reagent': [['[Cl,Br,I;D0]'],['[Pd]', '[H][H]']],
    'Reaction type':['Strong acid', 'Pd with H2', 'Unidentified'],
    'Templates': [NCBz_deprotect_HBr, None, None]
    },  
    
    'Iodo N-methylation':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [CH3I_N]
    },  
    'O-TBS deprotection':
    {
    'Reagent': [['[F;-1]'], ['[F;+0;H1]'], ['[OH2]']],
    'Reaction type':['Fluoride condition', 'Fluoride condition', 'Aqueous acidic condition', 'Unidentified'],
    'Templates': [Silyl_ether_deprotection_F, Silyl_ether_deprotection_F, Silyl_ether_deprotection_acidic, None]
    },  
    'O-TBDPS deprotection':
    {
    'Reagent': [['[F;-1]'], ['[F;+0;H1]'], ['[OH2]']],
    'Reaction type':['Fluoride condition', 'Fluoride condition', 'Aqueous acidic condition', 'Unidentified'],
    'Templates': [Silyl_ether_deprotection_F, Silyl_ether_deprotection_F, Silyl_ether_deprotection_acidic, None]
    },  
    'O-TIPS deprotection':
    {
    'Reagent': [['[F;-1]'], ['[F;+0;H1]'], ['[OH2]']],
    'Reaction type':['Fluoride condition', 'Fluoride condition', 'Aqueous acidic condition', 'Unidentified'],
    'Templates': [Silyl_ether_deprotection_F, Silyl_ether_deprotection_F, Silyl_ether_deprotection_acidic, None]
    },  
    'Methyl esterification':
    {
    'Reagent': [['[CH3]-[I]'], ['[CH3]-[OH]'],],
    'Reaction type':['Methyl iodide', 'Methanol', 'Unidentified'],
    'Templates': [CH3I_O, Methyl_esterification_MeOH, None]
    },  
    
    'Bromo Miyaura boration':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Miyaura_borylation]
    },  
    'Chloro Miyaura boration':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Miyaura_borylation]
    },  
    'Triflyloxy Miyaura boration':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Miyaura_borylation]
    },  
    'Iodo Miyaura boration':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Miyaura_borylation]
    },  
    'Chlorination':
    {
    'Reagent': [['ClN(C=O)C=O'], ['Cl[Fe,Al](Cl)Cl']],
    'Reaction type':['NCS', 'FeCl3 or AlCl3', 'Unidentified'],
    'Templates': [NCS,FeCl3 ,None]
    },  
    'Hydroxy to chloro':
    {
    'Reagent': [['O=[S,P]-[Cl]'], ['ClP(Cl)(Cl)(Cl)Cl'], ['O=C(C(Cl)=O)Cl']],
    'Reaction type':['SOCl2 or POCl3', 'PCl5', 'Oxalyl chloride', 'Unidentified'],
    'Templates': [SOCl2_POCl3, PCl5, oxalyl_chloride, None]
    },  
    'Iodination':
    {
    'Reagent': [['IN(C=O)C=O'], ['[O]-[O]', 'I-I']],
    'Reaction type':['NIS', 'Peroxide', 'Unidentified'],
    'Templates': [NIS, O2_I2]
    },  
    
    'Bechamp reduction':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Bechamp_reduction]
    },  
    
    'Amide to amine reduction':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Amide_reduction_hydride]
    },  
    'Alcohol + amine condensation':
    {
'Reagent': [['C-N=N-C'],['O=[S,P]-[Cl]']],
'Reaction type':['Mitsunobu', 'React with SOCl2 or POCl3','Unidentified'],
'Templates': [Mitsunobu,ROH_NH2_SOCl2_POCl3, None ]
    },    
    'Friedel-Crafts acylation':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Friedel_Crafts_acylation ]
    },    
    'Bromo Sonogashira coupling':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Sonogashira ]
    },    
    'Iodo Sonogashira coupling':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Sonogashira ]
    },    
    'Chloro Sonogashira coupling':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Sonogashira ]
    },    
    'Triflyloxy Sonogashira coupling':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Sonogashira ]
    },    
    'Knoevenagel condensation':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Knoevenagel]
    },    
    'Wohl-Ziegler bromination':
    {
'Reagent': [['O-O'],['N=N'],],
'Reaction type':['Reaction with radical','Reaction with radical', 'Reaction with NBS'],
'Templates': [Wohl_Ziegler_radical,Wohl_Ziegler_radical, Wohl_Ziegler]
    },    
    
    'Bromo Grignard reaction':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Grignard]
    },    
    
    'Bromo Grignard + ester reaction':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Grignard]
    },    
    
    'Chloro Grignard reaction':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Grignard]
    },    
    'Chloro Grignard + ester reaction':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Grignard]
    },    
    'Grignard ester substitution':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Grignard]
    },    
    'Iodo Grignard reaction':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Grignard]
    },    
    'Grignard + acid chloride ketone synthesis':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Grignard]
    },    
    'Bromo Grignard preparation':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Grignard_prep]
    },    
    'Iodo Grignard preparation':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Grignard_prep]
    },    
    'Chloro Grignard preparation':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Grignard_prep]
    },
    'Wittig olefination':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Wittig]
    },
    'Bromo Stille reaction':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Stille_reaction]
    },
    'Chloro Stille reaction':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Stille_reaction]
    },
    'Iodo Stille reaction':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Stille_reaction]
    },
    'Darzens chlorination':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Darzens_halogenation]
    },
    'Darzens bromination':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Darzens_halogenation]
    },
    'Weinreb amide synthesis':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Weinreb_amide]
    },
    'Weinreb ketone synthesis':
    {
'Reagent': [],
'Reaction type':['Reaction'],
'Templates': [Weinreb_ketone]
    },
    'Fischer-Speier esterification':
    {
    'Reagent': [ ],
    'Reaction type':['Reaction', ],
    'Templates': [Esterification ]
    },
    'Steglich esterification':
    {
    'Reagent': [ ],
    'Reaction type':['Reaction', ],
    'Templates': [COOH_amine_condense_cat ]
    },
    
    'Azide-alkyne Huisgen cycloaddition':
    {
    'Reagent': [ ],
    'Reaction type':['Reaction', ],
    'Templates': [Huisgen_cycloaddition ]
    },
    
    'Claisen-Schmidt condensation':
    {
    'Reagent': [ ],
    'Reaction type':['Reaction', ],
    'Templates': [claisen_schmidt_condense ]
    },
    
    'Aldehyde Dess-Martin oxidation':
    {
    'Reagent': [ ],
    'Reaction type':['Reaction', ],
    'Templates': [Dess_Martin_ox ]
    },
    
    'Ketone Dess-Martin oxidation':
    {
    'Reagent': [ ],
    'Reaction type':['Reaction', ],
    'Templates': [Dess_Martin_ox ]
    },
    
    'Pinner reaction':
    {
    'Reagent': [ ],
    'Reaction type':['Reaction', ],
    'Templates': [Pinner_reaction ]
    },
    
    'N-Cbz protection':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Amide_SchottenBaumann]
    },
    'Mesyloxy N-alkylation':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Mesyloxy_N_alkylation]
    },
    'Keto alpha-alkylation':
    {
    'Reagent': [],
    'Reaction type':['Reaction'],
    'Templates': [Keto_alpha_alkylation]
    },    
    'Ester hydrolysis':
    {
    'Reagent': [['[O;H1;-1]'],['[Li,Na,K][O;H1]'],['[H2O]'],],
    'Reaction type':[ 'Deprotection with OH-', 'Deptrotection with Alkali hydroxide', 'Deprotection with water', 'Missing agents'],
    'Templates': [Alkyl_ester_deprotection_OH,Alkyl_ester_deprotection_OH,Alkyl_ester_deprotection_aq, None]
    },
}
    

