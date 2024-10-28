
#TODO: modify the origin of the code got from smiles GA implementation in Guacamol
'''
Adapted from:
  Written by Jan H. Jensen 2018.
  Many subsequent changes inspired by https://github.com/BenevolentAI/guacamol_baselines/tree/master/graph_ga
And:
  https://github.com/BenevolentAI/guacamol_baselines/blob/jtvae/graph_ga/goal_directed_generation.py
'''

from rdkit import rdBase
rdBase.DisableLog('rdApp.*')
from typing import List, Optional
import numpy as np
import argparse
from time import time
import os
import joblib
from joblib import delayed
from molscore.manager import MolScore, MolScoreBenchmark
import nltk
import copy
from molscore.utils.smiles_grammar import GCFG
from collections import namedtuple
from molscore.utils.cfg_util import encode, decode
from ccdc.molecule import Molecule as CCDCMolecule

Molecule = namedtuple('Molecule', ['score', 'smiles', 'gene'])

def cfg_to_gene(prod_rules, max_len=-1):
    gene = []
    for r in prod_rules:
        lhs = GCFG.productions()[r].lhs()
        possible_rules = [idx for idx, rule in enumerate(GCFG.productions())
                          if rule.lhs() == lhs]
        gene.append(possible_rules.index(r))
    if max_len > 0:
        if len(gene) > max_len:
            gene = gene[:max_len]
        else:
            gene = gene + [np.random.randint(0, 256)
                           for _ in range(max_len - len(gene))]
    return gene


def gene_to_cfg(gene):
    prod_rules = []
    stack = [GCFG.productions()[0].lhs()]
    for g in gene:
        try:
            lhs = stack.pop()
        except Exception:
            break
        possible_rules = [idx for idx, rule in enumerate(GCFG.productions())
                          if rule.lhs() == lhs]
        rule = possible_rules[g % len(possible_rules)]
        prod_rules.append(rule)
        rhs = filter(lambda a: (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None'),
                     GCFG.productions()[rule].rhs())
        stack.extend(list(rhs)[::-1])
    return prod_rules

def mutation(gene):
    idx = np.random.choice(len(gene))
    gene_mutant = copy.deepcopy(gene)
    gene_mutant[idx] = np.random.randint(0, 256)
    return gene_mutant

def remove_duplicates(population):
    unique_smiles = set()
    unique_population = []
    for item in population:
        score, smiles, gene = item
        if smiles not in unique_smiles:
            unique_population.append(item)
        unique_smiles.add(smiles)
    return unique_population
    
def robust_mutation(original_smiles, original_gene, max_attempts=10):
    while max_attempts > 0:
        # Step 1: Perform the mutation
        c_gene = mutation(original_gene)
        
        try:
            # Step 2: Check if mutation produces a syntactically valid SMILES
            c_smiles = decode(gene_to_cfg(c_gene))

            # Step 3: Check if the SMILES is not an empty string
            if not c_smiles:
                max_attempts -= 1
                continue  

            # Step 4: Check if the new SMILES is different from the original
            if c_smiles == original_smiles:
                max_attempts -= 1
                continue  
            
            # Step 5: Check if a valid Molecule object can be generated
            # We consider a molecule valid if it can be instantiated as a CCDCMolecule
            molecule = CCDCMolecule.from_string(c_smiles)
            if molecule is None:
                max_attempts -= 1
                continue 

            # If all checks pass, return the new SMILES and gene
            return c_smiles, c_gene

        except Exception as e:
            # If decoding fails, decrement attempts and retry
            max_attempts -= 1
            continue

    # If all attempts are exhausted, return None to indicate failure
    return None, None


class SMILES_GA:

    def __init__(self, smi_file, population_size, n_mutations, gene_size, generations, n_jobs=-1, random_start=False, patience=5):
        """
        Initialize SMILES-based GA for molecule optimization.
        """
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.smi_file = smi_file
        self.population_size = population_size
        self.n_mutations = n_mutations
        self.gene_size = gene_size
        self.generations = generations
        self.random_start = random_start
        self.patience = patience
        self.all_smiles = self.load_smiles_from_file(smi_file)

    def load_smiles_from_file(self, smi_file):
        with open(smi_file) as f:
            return [s.strip() for s in f if s.strip()]

    def generate_optimized_molecules(self, scoring_function, number_molecules: int,
                                     starting_population: Optional[List[str]] = None,) -> List[str]:

        # Initialize a cache for already scored molecules
        score_cache = {} 
        
        if number_molecules > self.population_size:
            self.population_size = number_molecules
            print(f'Benchmark requested more molecules than expected: new population is {number_molecules}')
            
        # fetch initial population
        # Get the first 'population_size' SMILES from the list of all SMILES
        # The population size will ultimately be 100, but we will start with 20
        print(f'Selecting initial population of {self.population_size} molecules...')
        starting_population = self.all_smiles[:self.population_size]
        
        # Calculate initial genes with multiprocessing
        start_time = time()  # <-- Time tracking 
        print(f'Calculating initial genes...')
        joblist = (delayed(cfg_to_gene)(encode(s), max_len=self.gene_size) for s in starting_population)
        initial_genes = self.pool(joblist)
        print(f'Initial gene calculation took {time() - start_time:.2f} seconds')  # <-- Time tracking 
        
       
        # Insert SMILES and respective genes in the cache
        # The score field will be empty (None) for now
        for smiles, gene in zip(starting_population, initial_genes):
            if smiles not in score_cache:
                score_cache[smiles] = {"score": None, "gene": gene}  
        
        start_time = time()  # <-- Time tracking 
        # Separate SMILES that need scoring from those already scored, if cache is used
        print(f'Scoring and caching initial population...') #(TODO: Remove or change position)
        # Find SMILES in the cache that need scoring (i.e., their score is None)
        smiles_to_score = [smiles for smiles in score_cache if score_cache[smiles]["score"] is None]
        # Score the SMILES that need scoring
        new_scores = scoring_function(smiles_to_score, flt=True, score_only=True)
        
        # Update cache with the newly scored values
        for smiles, score in zip(smiles_to_score, new_scores):
            score_cache[smiles]["score"] = score

        print(f'Scoring initial population took {time() - start_time:.2f} seconds')  # <-- Time tracking
       
        # Initialize the population with scores and genes
        population = [ Molecule(entry["score"], smiles, entry["gene"]) 
                       for smiles,entry in score_cache.items() if entry["score"] is not None ]
        population = sorted(population, key=lambda x: x.score, reverse=True)[:self.population_size]
        
        print(f'Starting evolution process...')
        # Evolution process
        t0 = time()
        patience = 0

        for generation in range(self.generations):

            # print the population (This is for DEBUGGING purposes)
            print(f'Generation {generation}:')
            for molecule in population:
                print(f'{molecule.smiles} --> {molecule.score}')


            # Track scores to check for early stopping
            old_scores = [molecule.score for molecule in population]
            all_genes = [molecule.gene for molecule in population]
            all_smiles = [molecule.smiles for molecule in population]
            choice_indices = np.random.choice(len(all_genes), self.n_mutations, replace=False)
            genes_to_mutate = [all_genes[i] for i in choice_indices]
            smiles_to_mutate = [all_smiles[i] for i in choice_indices]

            start_time = time()  # <-- Time tracking 
            
            # EVOLVE/MUTATE GENES
            # Mutation using multiprocessing
            joblist = (delayed(robust_mutation)(smiles, gene) for smiles, gene in zip(smiles_to_mutate, genes_to_mutate))
            mutated_results = self.pool(joblist)
            
            print(f'Mutation phase took {time() - start_time:.2f} seconds')  # <-- Time tracking 
            
            # Insert mutated molecules into cache (This steo does also remove duplicates from the cache:
            # If a mutated smiles is equal to another one already present it is not added)
            for c_smiles, c_gene in mutated_results:
                if c_smiles is not None and c_smiles not in score_cache:
                    score_cache[c_smiles] = {"score": None, "gene": c_gene}
                    
                         
            start_time = time()  # <-- Time tracking 
            
            # Score the MSILES that have not been scored yet
            smiles_to_score = [smiles for smiles in score_cache if score_cache[smiles]["score"] is None]
            new_scores = scoring_function(smiles_to_score, flt=True, score_only=True)
            
            # Update cache with new scores
            for smiles, score in zip(smiles_to_score, new_scores):
                score_cache[smiles]["score"] = score

            print(f'Scoring phase took {time() - start_time:.2f} seconds')  # <-- Time tracking
            
            # SELECTION: Survival of the fittest
            # Create population from cache
            population = [
                Molecule(entry["score"], smiles, entry["gene"]) 
                for smiles, entry in score_cache.items() if entry["score"] is not None
            ]
            # Select the top `population_size` based on the scores
            population = sorted(population, key=lambda x: x.score, reverse=True)[:self.population_size]

            # Stats
            gen_time = time() - t0
            mol_sec = (self.population_size + self.n_mutations) / gen_time
            t0 = time()
            
            # Extract the scores for use in the early stopping condition and statistics
            population_scores = [molecule.score for molecule in population]

            # Early stopping
            if population_scores == old_scores:
                patience += 1
                print(f'Failed to progress: {patience}')
                if patience >= self.patience:
                    print(f'No more patience, bailing...')
                    break
            else:
                patience = 0

            print(f'{generation} | '
                  f'max: {np.max(population_scores):.3f} | '
                  f'avg: {np.mean(population_scores):.3f} | '
                  f'min: {np.min(population_scores):.3f} | '
                  f'std: {np.std(population_scores):.3f} | '
                  f'{gen_time:.2f} sec/gen | '
                  f'{mol_sec:.2f} mol/sec')

        # Return final population
        return [(molecule.smiles, molecule.score) for molecule in population]


def main(args):
    generator = SMILES_GA(smi_file=args.smiles_file,
                          population_size=args.population_size,
                          n_mutations=args.n_mutations,
                          gene_size=args.gene_size,
                          generations=args.generations,
                          n_jobs=args.n_jobs,
                          random_start=args.random_start,
                          patience=args.patience)
                          

    if args.molscore in MolScoreBenchmark.presets:
        scoring_function = MolScoreBenchmark(model_name='smilesGA', output_dir="./", budget=1000, benchmark=args.molscore)
        for task in scoring_function:
            final_population_smiles = generator.generate_optimized_molecules(scoring_function=task, number_molecules=args.population_size)
    elif os.path.isdir(args.molscore):
        scoring_function = MolScoreBenchmark(model_name='smilesGA', output_dir="./", budget=1000, custom_benchmark=args.molscore)
        for task in scoring_function:
            final_population_smiles = generator.generate_optimized_molecules(scoring_function=task, number_molecules=args.population_size)
    else:
        scoring_function = MolScore(model_name='smilesGA', task_config=args.molscore)
        final_population_smiles = generator.generate_optimized_molecules(scoring_function=scoring_function,
                                                                         number_molecules=args.population_size)

    with open(os.path.join(scoring_function.save_dir, 'final_population.smi'), 'w') as f:
        for smi, score in final_population_smiles:
            f.write(f'{smi}\t{score}\n')


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--molscore', type=str, help='Path to MolScore config, or directory of configs, or name of Benchmark')
    parser.add_argument('--smiles_file', type=str, help='Path to initial SMILES population file')

    # Optional arguments for GA setup
    optional = parser.add_argument_group('Optional')
    optional.add_argument('--seed', type=int, default=42, help='Random seed')
    optional.add_argument('--population_size', type=int, default=10, help='Population size')
    optional.add_argument('--n_mutations', type=int, default=5, help='Number of mutations per generation')
    optional.add_argument('--gene_size', type=int, default=-1, help='Gene size for the CFG-based encoding')
    optional.add_argument('--generations', type=int, default=3, help='Number of generations')
    optional.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs')
    optional.add_argument('--random_start', action='store_true', help='Start with a random population')
    optional.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    return args


if __name__ == "__main__":
    start = time()
    args = get_args()
    main(args)
    print(f'Total Time Taken: {time() - start:.2f} seconds')