import argparse
import os
import numpy as np
from typing import List
from molscore.manager import MolScore, MolScoreBenchmark
from openbabel import pybel as pb
from rdkit import Chem
import importlib.resources as pkg_resources

import molscore.configs
pb.ob.obErrorLog.StopLogging()
import time
import json
from pathlib import Path
import re
import glob
import pandas as pd


MOLSCORE_PATH = str(pkg_resources.files(molscore.configs))

def import_ccdc():
    """
    Safely imports required modules from the CCDC library, handles exceptions, and returns None if import fails.
    
    :return: io, conformer, ccdcMolecule (or None for each if the import fails)
    """
    try:
        # Attempt to import the necessary modules from the CCDC library
        from ccdc import io
        return io
    except ImportError as e:
        # Handle ImportError if the module is not installed
        print(f"ImportError: CCDC module not found. Please ensure the CCDC package is installed and licensed. {e}")
    except Exception as ccdc_exception:
        # Handle any other unexpected exceptions during import
        print(f"Unexpected error with CCDC module: {ccdc_exception}. "
               "If you want to use the CCDC Python API, ensure the package is correctly installed and licensed.")
    # Return None for all if import fails
    return None,

def update_json_generator(generator: str):
    """Modify all JSON files in 3D_Benchmark to update the generator field."""
    
    # Get the path to the 3D_Benchmark config folder
    benchmark_path = Path(MOLSCORE_PATH)/"3D_Benchmark"

    # Iterate through all JSON files in the benchmark directory
    for json_file in benchmark_path.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)  # Load JSON content
        
        # Check if the scoring function exists and update the generator field
        if "scoring_functions" in data and isinstance(data["scoring_functions"], list):
            # Update the generator field with the user-defined value
            data["scoring_functions"][0]["parameters"]["generator"] = generator
            
            # Save the modified JSON file
            with open(json_file, "w") as f:
                json.dump(data, f, indent=4)

            print(f"Updated '{json_file.name}': Generator set to '{generator}'")


class RandomSampler:
    """
    Generator that samples molecules from a predefined list.
    """

    def __init__(self, molecules: List[str], format: str) -> None:
        """
        Args:
            molecules: list of molecules from which the samples will be drawn
            representations: Type of representation to use: 'SMILES', '3D', 'CSD_ENTRY'
        """
        self.molecules = molecules
        self.format = format

    def generate(self, number_samples: int, molscore) -> List[str]:
        """
        Randomly sample molecules from the predefined list.
        
        Args:
            number_samples: number of molecules to sample.
            
        Returns:
            A list of sampled molecules.
        """
        selected_molecules = list(np.random.choice(self.molecules, size=number_samples, replace=False))

        if self.format == 'ccdc':
            self.io, = import_ccdc()
            molecules = []
            for csd_entry in selected_molecules:
                mol = self.io.MoleculeReader('CSD')[csd_entry]
                molecules.append(mol)
            scores = molscore.score(molecular_inputs=molecules, flt=True, score_only=False)
            selected_molecules, scores = zip(*sorted(zip(selected_molecules, scores), key=lambda x: x[1], reverse=True))
        else:
            scores = molscore.score(molecular_inputs=selected_molecules, flt=True, score_only=False)
            selected_molecules, scores = zip(*sorted(zip(selected_molecules, scores), key=lambda x: x[1], reverse=True))
        # make a dictionary with the molecule and its score
        selected_molecules = [f'{molecule}  {score}' for molecule, score in zip(selected_molecules, scores)]
        return selected_molecules
    
def get_population(population_file, generator):
    """
    Load the population file and extract molecules based on the generator type.

    :param population_file: Path to the population file (expected in .txt format).
    :param generator: The molecular representation generator ('ccdc', 'rdkit', or 'obabel').
    :return: List of extracted molecules (CSD entries for 'ccdc', SMILES for others).
    """
    
    molecules = []
    
    # Ensure the file exists
    if not os.path.exists(population_file):
        raise FileNotFoundError(f"Population file '{population_file}' does not exist.")

    # Open the .txt population file and parse content
    with open(population_file, "r") as f:
        lines = f.readlines()

    # Extract molecules based on the generator type
    if generator == 'ccdc':
        # Extract CSD entries (first part before ":")
        molecules = [line.split(maxsplit=1)[0].strip() for line in lines if line.strip()]
    elif generator in ['rdkit', 'obabel']:
        # Extract SMILES (second part after ":")
        molecules = [line.split(maxsplit=1)[1].strip() for line in lines if len(line.split()) > 1]
    else:
        raise ValueError(f"Invalid generator '{generator}'. Supported options: 'ccdc', 'rdkit', 'pybel'.")

    return molecules
    

def main():
    # Choose the generator to use for molecular representation
    generator = 'obabel'
    # Update the generator field in all JSON files in the 3D_Benchmark directory
    update_json_generator(generator)
    
    # Number of repetitions
    num_runs = 10
    benchmark_name = "3D_Benchmark"

    # Define the folder where all benchmark results will be stored
    output_base_folder = "./benchmark_results_rnd_obabel"

    # Ensure base folder exists
    os.makedirs(output_base_folder, exist_ok=True)

    # Store all CSV file paths
    csv_files = []

    # Run the benchmark multiple times
    for run in range(1, num_runs + 1):  
        print(f"\n### Running Benchmark Iteration {run}/{num_runs} ###\n")
        
        # Define the output directory for this run
        output_dir = os.path.join(output_base_folder, f"run_{run}")
        os.makedirs(output_dir, exist_ok=True)
        
        benchmark = MolScoreBenchmark(benchmark='3D_Benchmark', model_name='randomSampler', output_dir=output_dir, budget=1000, exclude = ['15_ACNCOB10', '17_ACAZFE', '20_DAJLAC', '25_ABEVAG', '26_AKOQOH',]) # '1_ABAHIW', '10_ABEHAU', '11_TITTUO', '12_EGEYOG', '13_ABOBUP', '14_XIDTOW', '16_TACXUQ', '18_NIVHEJ', '19_ADUPAS', '21_OFOWIS', '22_CATSUL', '23_HESMUQ01', '24_GUDQOL', '28_AFECIA', '29_ACOVUL', ''  ])
        
        for molscore in sorted(benchmark, key=lambda x: int(re.search(r'\d+', x.cfg["task"]).group())):
            print(f"Running MolScore for task: {molscore.cfg['task']}")
            generator = molscore.cfg["scoring_functions"][0]["parameters"].get("generator", "rdkit")
            population_file = pkg_resources.files("molscore.configs.3D_Benchmark").joinpath(f"initial_populations/{molscore.cfg['task']}_init_pop.txt")
            molecules = get_population(population_file, generator)
            
            sampler = RandomSampler(molecules=molecules, format=generator)
            selected_molecules = sampler.generate(number_samples=100, molscore=molscore)
            # Print the first 5 selected molecules
            # for molecule in selected_molecules[:5]:
            #     print(f'{molecule}')
                            
        benchmark.summarize(chemistry_filter_basic=False,)
    
        # Locate the results.csv file in the benchmark output directory
        results_csv = glob.glob(os.path.join(output_dir, "**", "results.csv"), recursive=True)
        
        if results_csv:
            csv_files.append(results_csv[0])  # Store path to results.csv

    # If no CSVs were found, exit
    if not csv_files:
        print("No results.csv files found. Exiting.")
        exit()

    # Load all results and compute the average score per test
    df_list = [pd.read_csv(csv) for csv in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    # Compute average 3D_Benchmark_Score for each test
    average_scores = combined_df.groupby("task")["3D_Benchmark_Score"].mean().reset_index()

    # Print final results
    print("\n### Average 3D_Benchmark Scores Across 10 Runs ###\n")
    print(average_scores)

    # Save to CSV for reference
    average_scores.to_csv(os.path.join(output_base_folder, "average_result_rnd_obabel.csv"), index=False)

    print(f"\nResults saved to {output_base_folder}/average_results.csv")

if __name__ == "__main__":
    start = time.time()
    main()
    # Print the total time in minutes
    print(f'Time elapsed: {(time.time() - start) / 60:.2f} minutes')