import argparse
import os
import numpy as np
from typing import List
from molscore.manager import MolScore, MolScoreBenchmark
from openbabel import pybel as pb
from rdkit import Chem
import time 

class RandomSampler:
    """
    Generator that samples molecules from a predefined list.
    """

    def __init__(self, molecules: List[str]) -> None:
        """
        Args:
            molecules: list of molecules from which the samples will be drawn
        """
        self.molecules = molecules

    def generate(self, number_samples: int, molscore) -> List[str]:
        """
        Randomly sample molecules from the predefined list.
        
        Args:
            number_samples: number of molecules to sample.
            
        Returns:
            A list of sampled molecules.
        """
        selected_molecules = list(np.random.choice(self.molecules, size=number_samples, replace=False))
        scores = molscore.score(molecular_inputs=selected_molecules, flt=True, score_only=False)
        selected_molecules, scores = zip(*sorted(zip(selected_molecules, scores), key=lambda x: x[1], reverse=True))
        # make a dictionary with the molecule and its score
        selected_molecules = [f'{molecule}\t{score}' for molecule, score in zip(selected_molecules, scores)]
        return selected_molecules
    
def get_population(population_file):
        pkg_reader = 'rdkit'
        if pkg_reader == 'rdkit':
            suppl = Chem.SDMolSupplier(population_file, removeHs=False, sanitize=False)
            molecules = [mol for mol in suppl]
        elif pkg_reader == 'pybel':
            suppl = pb.readfile('sdf', population_file)
            molecules = [mol for mol in suppl]
        return molecules

def main(args):
    # Get populations files from the directory in order (The will have names population_1, population_2, etc)
    # Get only the ones that exist
    populations = [os.path.join(args.populations_dir, f'population_{i}.sdf') for i in range(1, 100) if os.path.exists(os.path.join(args.populations_dir, f'population_{i}.sdf'))]

    # Use MolScore to compute the scores for the selected SMILES
    if args.molscore in MolScoreBenchmark.presets:
        benchmark = MolScoreBenchmark(model_name='randomSampler', output_dir="./", budget=1000, benchmark=args.molscore)
        for molscore, population in zip(benchmark, populations):
            molecules = get_population(population)
            # Initialize the random sampler population
            sampler = RandomSampler(molecules=molecules)
            selected_molecules = sampler.generate(number_samples=args.population_size, molscore=molscore)
        benchmark.summarize(chemistry_filter_basic=False, diversity_check=False, mols_in_3d=True)
    elif os.path.isdir(args.molscore):
        benchmark = MolScoreBenchmark(benchmark= '3D_Benchmark', model_name='randomSampler', output_dir="./", budget=1000, custom_benchmark=args.molscore)
        for molscore, population in zip(benchmark, populations):
            molecules = get_population(population)
            # Initialize the random sampler population
            sampler = RandomSampler(molecules=molecules)
            selected_molecules = sampler.generate(number_samples=args.population_size, molscore=molscore)
        benchmark.summarize(chemistry_filter_basic=False, diversity_check=False, mols_in_3d=True)
    else:
        ms = MolScore(model_name='randomSampler', task_config=args.molscore)

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--molscore', type=str, help='Path to MolScore config, or directory of configs, or name of Benchmark')
    parser.add_argument('--populations_dir', type=str, required=True, help='Path to directory containing sdf files of the molecules to sample from')
    parser.add_argument('--population_size', type=int, default=100, help='Number of molecules to randomly sample')

    return parser.parse_args()

if __name__ == "__main__":
    start = time.time()
    args = get_args()
    main(args)
    print(f'Time elapsed: {time.time() - start}')