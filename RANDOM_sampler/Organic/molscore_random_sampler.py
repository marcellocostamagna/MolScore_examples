import argparse
import os
import numpy as np
from typing import List
from molscore.manager import MolScore, MolScoreBenchmark

class RandomSmilesSampler:
    """
    Generator that samples SMILES strings from a predefined list.
    """

    def __init__(self, molecules: List[str]) -> None:
        """
        Args:
            molecules: list of molecules from which the samples will be drawn
        """
        self.molecules = molecules

    def generate(self, number_samples: int) -> List[str]:
        """
        Randomly sample SMILES strings from the predefined list.
        
        Args:
            number_samples: number of SMILES strings to sample.
            
        Returns:
            A list of sampled SMILES strings.
        """
        return list(np.random.choice(self.molecules, size=number_samples, replace=False))


def main(args):
    # Read SMILES file
    with open(args.smiles_file, 'r') as smiles_file:
        smiles_list = [s.strip() for s in smiles_file.readlines()]

    # Initialize the random sampler with the SMILES list
    sampler = RandomSmilesSampler(molecules=smiles_list)

    # Use MolScore to compute the scores for the selected SMILES
    if args.molscore in MolScoreBenchmark.presets:
        scoring_function = MolScoreBenchmark(model_name='randomSampler', output_dir="./", budget=1000, benchmark=args.molscore)
    elif os.path.isdir(args.molscore):
        scoring_function = MolScoreBenchmark(model_name='randomSampler', output_dir="./", budget=1000, custom_benchmark=args.molscore)
    else:
        scoring_function = MolScore(model_name='randomSampler', task_config=args.molscore)

    # Extract the output directory from the MolScore object
    output_dir = scoring_function.save_dir

    # Randomly select the molecules
    selected_smiles = sampler.generate(number_samples=args.population_size)

    # Calculate the scores using MolScore
    scores = scoring_function(selected_smiles, flt=True, score_only=True)
    
    # Sort the selected molecules by their scores in descending order
    selected_smiles, scores = zip(*sorted(zip(selected_smiles, scores), key=lambda x: x[1], reverse=True))

    # Save the randomly selected molecules with their scores to a file
    output_file = os.path.join(output_dir, 'randomly_selected_molecules.txt')
    with open(output_file, 'w') as f:
        for smi, score in zip(selected_smiles, scores):
            f.write(f'{smi}\t{score}\n')

    print(f"Randomly selected SMILES and scores saved to {output_file}")


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--molscore', type=str, help='Path to MolScore config, or directory of configs, or name of Benchmark')
    parser.add_argument('--smiles_file', type=str, required=True, help='Path to SMILES file containing molecules to sample from')
    parser.add_argument('--population_size', type=int, default=100, help='Number of molecules to randomly sample')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
