# Script to select the best k molecules form a list of inorganic molecules obtained from sdf files
# as ccdc molecules.
# This script is meant to test Molscore and its newly added capability to mangae other molecular 
# formats besides SMILES and use molecular identidiers to track molecules.

import argparse
import os
import numpy as np
from typing import List
from molscore.manager import MolScore, MolScoreBenchmark
from ccdc import io
from openbabel import pybel as pb
from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromMol2File, MolFromPDBFile, MolFromMolFile, MolFromXYZFile, SDMolSupplier

ccdc_input_formats = ['sdf', 'mol2', 'pdb', 'mol', 'cif']
pybel_input_formats = ['sdf', 'mol2', 'pdb', 'mol', 'cif', 'xyz']
rdkit_input_formats = ['sdf', 'mol2', 'pdb', 'mol', 'xyz']

def read_mol_from_file(path, removeHs=False, sanitize=False):
    """
    General RDKit reader 
    """

    extension = path.split('.')[-1]
    if extension == 'mol':
        return MolFromMolFile(path, removeHs=removeHs, sanitize=sanitize)
    elif extension == 'mol2':
        return MolFromMol2File(path, removeHs=removeHs, sanitize=sanitize)
    elif extension == 'pdb':
        return MolFromPDBFile(path, removeHs=removeHs, sanitize=sanitize)
    elif extension == 'xyz':
        return MolFromXYZFile(path)
    elif extension == 'sdf':
        suppl = Chem.SDMolSupplier(path, removeHs=removeHs, sanitize=sanitize)
        return next(suppl, None)
    

def main(args):
    # TODO: remove hard coding of package to use
    
    packages = ['ccdc', 'pybel', 'rdkit']
    package = packages[0]
    
    # Read molecules from sdf files
    dir_path = args.init_dir
    molecules = []
    if package == 'ccdc':
        for file in os.listdir(dir_path):
            # Check that file ends with one of the supported formats
            if file.endswith(tuple(ccdc_input_formats)):
                molecules.extend(io.MoleculeReader(os.path.join(dir_path, file)))
    elif package == 'pybel':
        for file in os.listdir(dir_path):
            # Check that file ends with one of the supported formats
            format = file.split('.')[-1]
            if file.endswith(tuple(pybel_input_formats)):
                molecules.extend(pb.readfile(format, os.path.join(dir_path, file)))
    elif package == 'rdkit':
        for file in os.listdir(dir_path):
            # Check that file ends with one of the supported formats
            format = file.split('.')[-1]
            if file.endswith(tuple(rdkit_input_formats)):
                molecules.append(read_mol_from_file(os.path.join(dir_path, file)))

    # create a unique 2 digits identifier for each molecule
    mol_ids = []
    for i, mol in enumerate(molecules):
        mol.identifier = f"{i:02}"
        mol_ids.append(mol.identifier)

    # Divide the molecules into 3 batches
    n_batches = 3
    batch_size = len(molecules) // n_batches
    batch1 = molecules[:batch_size]
    batch2 = molecules[batch_size:2*batch_size]
    batch3 = molecules[2*batch_size:]
    
    # Divide the identifiers into the same 3 batches
    batch1_ids = mol_ids[:batch_size]
    batch2_ids = mol_ids[batch_size:2*batch_size]
    batch3_ids = mol_ids[2*batch_size:]
    
    # HOW TO INITILAIZE MOLSCORE
    ms = MolScore(model_name='topKSampler', task_config=args.molscore)

    # Extract the output directory from the MolScore object
    output_dir = ms.save_dir

    # HOW TO COMPUTE SCORES OF MOLECULE OBJECTS (NOT SMILES)
    # Score all the molecules with caching enabled (Generation 0)
    scores = ms.score(mol_ids=batch1_ids, molecular_inputs=batch1, flt=True, score_only=False)
    
    # SIMULATION of a GA
    # Select the best k molecules
    selected_indices = np.argsort(scores)[::-1][:args.population_size]
    best_molecules = [molecules[i] for i in selected_indices]
    # get the identifiers of the selected molecules
    best_mol_ids = [batch1_ids[i] for i in selected_indices]
    
    # simulate Genetic operation and addition of new molecules
    # Add new molecules to the population (Genration 1)
    new_molecules = best_molecules + batch2
    gen1_ids = best_mol_ids + batch2_ids
    scores = ms.score(mol_ids=gen1_ids, molecular_inputs=new_molecules, flt=True, score_only=False)
    
    # Select the best k molecules
    selected_indices = np.argsort(scores)[::-1][:args.population_size]
    best_molecules = [new_molecules[i] for i in selected_indices]
    best_mol_ids = [gen1_ids[i] for i in selected_indices]
    
    # simulate Genetic operation and addition of new molecules
    # Add new molecules to the population (Genration 2)
    new_molecules = best_molecules + batch3
    gen2_ids = best_mol_ids + batch3_ids
    scores = ms.score(mol_ids=gen2_ids, molecular_inputs=new_molecules, flt=True, score_only=False)

    # Select the best k molecules
    selected_indices = np.argsort(scores)[::-1][:args.population_size]
    best_molecules = [molecules[i] for i in selected_indices]
    best_scores = [scores[i] for i in selected_indices]
    sdf_mols = [m.to_string('sdf') for m in best_molecules]
    
    # Save the selected molecules to a file
    output_file = os.path.join(output_dir, 'selected_molecules.sdf')
    with open(output_file, 'w') as f:
        for mol, score in zip(sdf_mols, best_scores):
            f.write(mol)
            f.write(f"> {score}\n")
            
    print(f"Selected molecules saved to {output_file}")


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--molscore', type=str, help='Path to MolScore config, or directory of configs, or name of Benchmark')
    parser.add_argument('--init_dir', type=str, required=True, help='Path to directory containing sdf files of the molecules to sample from')
    parser.add_argument('--population_size', type=int, default=5, help='Number of molecules to select sample')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
