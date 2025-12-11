from rdkit import Chem
from rdkit.Chem import AllChem

# Alanine dipeptide SMILES (Ace-Ala-Nme)
smiles = "CC(C)[C@H](N)C(=O)N"

mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)

AllChem.EmbedMolecule(mol, useBasicKnowledge=True, maxAttempts=1000)
AllChem.UFFOptimizeMolecule(mol)

Chem.MolToPDBFile(mol, "data/pdb/alanine_dipeptide.pdb")

print("✓ Saved: data/pdb/alanine_dipeptide.pdb")
