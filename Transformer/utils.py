import re
from rdkit import Chem


def smi_tokenizer(smi: str):
    """Tokenize a SMILES molecule or reaction, adapted from https://github.com/pschwllr/MolecularTransformer"""
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == "".join(tokens)

    return " ".join(tokens)


def canonicalize_smiles(smiles: str, remove_atom_number: bool = True):
    """Adapted from Molecular Transformer"""
    smiles = "".join(smiles.split())
    cano_smiles = ""

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        if remove_atom_number:
            [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]

        cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        # Sometimes stereochem takes another canonicalization... (just in case)
        mol = Chem.MolFromSmiles(cano_smiles)
        if mol is not None:
            cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    return cano_smiles
