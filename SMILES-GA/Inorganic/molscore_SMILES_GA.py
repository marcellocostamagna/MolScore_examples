# Script to run the SMILES-GA optimization using MolScore with HSR scoring function

'''
Original Algorithm from:

  doi:10.1246/cl.180665, https://github.com/tsudalab/ChemGE
  
  Many subsequent changes inspired by https://github.com/BenevolentAI/guacamol_baselines/tree/master/smiles_ga
'''

from rdkit import rdBase
rdBase.DisableLog('rdApp.*')
from typing import List, Optional
import numpy as np
import argparse
from time import time
import os
# import joblib
import multiprocessing
from functools import partial
from joblib import delayed
from molscore.manager import MolScore, MolScoreBenchmark
from molscore.scoring_functions.utils import timedFunc2
import nltk
import copy
from collections import namedtuple
from cfg_util import encode, decode
from smiles_grammars import GCFG_INORG, GCFG_ORG


from cfg_util import encode, decode

try:
    from ccdc.molecule import Molecule as CCDCMolecule
    from ccdc import search
except Exception as ccdc_exception:
    print(f"Error importing CCDC: {ccdc_exception}")
    
import signal

Molecule = namedtuple('Molecule', ['score', 'smiles', 'gene'])

class TimeoutException(Exception):
    pass

# Set up a timeout handler
def timeout_handler(signum, frame):
    raise TimeoutException

# Set the signal handler for alarm
signal.signal(signal.SIGALRM, timeout_handler)

def cfg_to_gene(prod_rules, grammar, max_len=-1):
    gene = []
    for r in prod_rules:
        lhs = grammar.productions()[r].lhs()
        possible_rules = [idx for idx, rule in enumerate(grammar.productions())
                          if rule.lhs() == lhs]
        gene.append(possible_rules.index(r))
    if max_len > 0:
        if len(gene) > max_len:
            gene = gene[:max_len]
        else:
            gene = gene + [np.random.randint(0, 256)
                           for _ in range(max_len - len(gene))]
    return gene


def gene_to_cfg(gene, grammar):
    prod_rules = []
    stack = [grammar.productions()[0].lhs()]
    for g in gene:
        try:
            lhs = stack.pop()
        except Exception:
            break
        possible_rules = [idx for idx, rule in enumerate(grammar.productions())
                          if rule.lhs() == lhs]
        rule = possible_rules[g % len(possible_rules)]
        prod_rules.append(rule)
        rhs = filter(lambda a: (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None'),
                     grammar.productions()[rule].rhs())
        stack.extend(list(rhs)[::-1])
    return prod_rules

# Define the smiles_to_gene function with a timeout
def smiles_to_gene(smile, max_len=-1, timeout=20):
    """
    Converts a SMILES string to a gene representation using the encode and cfg_to_gene functions.
    Uses a timeout to prevent long execution times.

    :param smile: The SMILES string to be processed.
    :param max_len: Maximum length of the gene.
    :param timeout: Timeout value in seconds.
    :return: The gene representation or None if the process times out.
    """
    try:
        # Set the alarm for the timeout
        signal.alarm(timeout)
        # Step 1: Encode the SMILES string
        prod_rules = encode(smile)
        # Step 2: Convert to gene representation
        gene = cfg_to_gene(prod_rules, max_len=max_len)
        # Disable the alarm if it finishes successfully
        signal.alarm(0)
        return gene
    except TimeoutException:
        print(f"Timeout occurred while processing SMILES: {smile}")
        return None
    except Exception as e:
        print(f"Error processing SMILES {smile}: {e}")
        return None
    finally:
        # Always disable the alarm in the end to prevent it from affecting other parts of the code
        signal.alarm(0)

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

def get_csd_fingerprint(molecule):
    sim_search = search.SimilaritySearch(molecule)
    fp_builder = sim_search._fp
    fp = fp_builder.similarity_fingerprint(molecule._molecule)
    return fp
    
def topological_similarity(smiles1, smiles2):
    query = CCDCMolecule.from_string(smiles1)
    target = CCDCMolecule.from_string(smiles2)
    query_fp = get_csd_fingerprint(query)
    target_fp = get_csd_fingerprint(target)
    similarity = query_fp.tanimoto(target_fp)
    return similarity
    
def diversity_check(candidate_smiles, candidate_score, population, score_threshold=0.001, similarity_threshold=0.99):
    """
    Checks whether a candidate molecule satisfies diversity constraints with respect to the current population.
    A candidate molecule is considered acceptable if:
    1. Its score is sufficiently different from the scores of the molecules in the population.
    2. Its topological similarity to the molecules in the population is below a certain threshold.

    """
    for molecule in population:
        # Check if the score difference is below the threshold
        if abs(candidate_score - molecule.score) <= score_threshold:
            # Compute topological similarity
            similarity = topological_similarity(candidate_smiles, molecule.smiles)
            if similarity >= similarity_threshold:
                # Too similar in both score and topology
                return False
    # Passed all checks
    return True
    
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
            #TODO: remove comment if CCDC is available
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
        # self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
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

    def generate_optimized_molecules(self, scoring_function, number_molecules: int, output_dir=None,
                                     starting_population: Optional[List[str]] = None,) -> List[str]:

        start_time = time() 
        # Initialize a cache for already scored molecules
        score_cache = {} 
        
        if number_molecules > self.population_size:
            self.population_size = number_molecules
            print(f'Benchmark requested more molecules than expected: new population is {number_molecules}')
            
        # Fetch initial population
        # Get the first 'population_size' SMILES from the list of all SMILES
        print(f'Selecting initial population of {self.population_size} molecules...')
        starting_population = self.all_smiles[:self.population_size]
        
        # Calculate initial genes with multiprocessing 
        print(f'Calculating initial genes...')
        n_processes = min(self.n_jobs, len(starting_population), os.cpu_count())
        timeout = 5
        smiles_to_gene_partial = partial(smiles_to_gene, max_len=self.gene_size, timeout=timeout)
        with multiprocessing.Pool(n_processes) as pool:
            initial_genes = pool.map(smiles_to_gene_partial, starting_population)
            
        # Insert SMILES and respective genes in the cache
        for smiles, gene in zip(starting_population, initial_genes):
            if smiles not in score_cache and gene is not None:
                score_cache[smiles] = {"score": None, "gene": gene}  

        print(f'Scoring initial population...\n')
        # Separate SMILES that need scoring from those already scored, if cache is used
        # Find SMILES in the cache that need scoring (i.e., their score is None)
        smiles_to_score = [smiles for smiles in score_cache if score_cache[smiles]["score"] is None]
        # Score the SMILES that need scoring
        new_scores = scoring_function(smiles_to_score, flt=True, score_only=True)
        # Update cache with the newly scored values
        for smiles, score in zip(smiles_to_score, new_scores):
            score_cache[smiles]["score"] = score
       
        # Initialize the population with scores and genes
        population = [ Molecule(entry["score"], smiles, entry["gene"]) 
                       for smiles,entry in score_cache.items() if entry["score"] is not None ]
        population = sorted(population, key=lambda x: x.score, reverse=True)[:self.population_size]
        
        population_scores = [molecule.score for molecule in population]
        
        # Print details of initial population
        print(f'Initial population | '
                  f'max: {np.max(population_scores):.3f} | '
                  f'avg: {np.mean(population_scores):.3f} | '
                  f'min: {np.min(population_scores):.3f} | '
                  f'std: {np.std(population_scores):.3f} | '
                  f'{(time() - start_time):.2f} sec/gen \n')
        
        print(f'Starting evolution process...\n')
        # Evolution process
        t0 = time()
        patience = 0

        for generation in range(self.generations):

            # Track scores to check for early stopping
            old_scores = [molecule.score for molecule in population]
            all_genes = [molecule.gene for molecule in population]
            all_smiles = [molecule.smiles for molecule in population]
            choice_indices = np.random.choice(len(all_genes), self.n_mutations, replace=False)
            genes_to_mutate = [all_genes[i] for i in choice_indices]
            smiles_to_mutate = [all_smiles[i] for i in choice_indices]

            # EVOLVE/MUTATE GENES
            print(f'Mutating genes...')
            # 1- Generation of children: mutation of genes
            start_mutations = time()
            n_processes = min(self.n_jobs, len(smiles_to_mutate), os.cpu_count())
            with multiprocessing.Pool(self.n_jobs) as pool:
                mutated_results = pool.starmap(robust_mutation, zip(smiles_to_mutate, genes_to_mutate))
            end_mutations = time()
                    
            # 2- Collect candidates to score (remove SMILES that are already in the cache)
            start_collect = time()
            candidates_to_score = []
            candidates_genes = []
            for c_smiles, c_gene in mutated_results:
                if c_smiles is not None and c_smiles not in score_cache:
                    candidates_to_score.append(c_smiles)
                    candidates_genes.append(c_gene)
            end_collect = time()
            
            # 3- Score the candidates      
            start_scoring = time()   
            print(f'Scoring mutated genes...\n')
            # Score the SMILES that have not been scored yet         
            new_scores = scoring_function(candidates_to_score, flt=True, save_files=True) #, score_only=True)
            end_scoring = time()
            
            # 4- Update cache with new scores
            start_update = time()
            # for candidate_smiles, candidate_gene, candidate_score in zip(candidates_to_score, candidates_genes, new_scores):
            #     if diversity_check(candidate_smiles, candidate_score, population):
            #         score_cache[candidate_smiles] = {"score": candidate_score, "gene": candidate_gene}
            end_update = time()
                
            # 5- Selection: survival of the fittest
            start_selection = time()
            # Create population from cache
            population = [
                Molecule(entry["score"], smiles, entry["gene"]) 
                for smiles, entry in score_cache.items() if entry["score"] is not None
            ]
            # Select the top `population_size` based on the scores
            population = sorted(population, key=lambda x: x.score, reverse=True)[:self.population_size]
            end_selection = time()

            # Stats
            gen_time = time() - t0
            mol_sec = (self.population_size + self.n_mutations) / gen_time
            t0 = time()
            
            # Extract the scores for use in the early stopping condition and statistics
            population_smiles = [(molecule.smiles, molecule.score) for molecule in population]
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
                  f'{mol_sec:.2f} mol/sec\n')
            
            # for debugging purposes print the time taken for each step
            print(f'Time taken for generation: {gen_time:.2f} seconds')
            print(f'1- Mutations: {end_mutations - start_mutations:.2f} seconds')
            print(f'2- Collection of candidates: {end_collect - start_collect:.2f} seconds')
            print(f'3- Scoring of candidates: {end_scoring - start_scoring:.2f} seconds')
            print(f'4- Update of cache: {end_update - start_update:.2f} seconds')
            print(f'5- Selection of population: {end_selection - start_selection:.2f} seconds')
            
            # with open(os.path.join(output_dir, f'population_{generation}.smi'), 'w') as f:
            #     for smi, score in population_smiles:
            #         f.write(f'{smi}\t{score}\n')

        # Return final population
        return [(molecule.smiles, molecule.score) for molecule in population]

def get_population(population_file):
    """
    Return a list of SMILES from a population file. 
    """
    

def main(args):
    
    # Get populations files from the directory in order (The will have names population_1, population_2, etc)
    populations = [os.path.join(args.populations_dir, f'population_{i}.smi') for i in range(1, 100) if os.path.exists(os.path.join(args.populations_dir, f'population_{i}.smi'))]    
    
    if args.molscore in MolScoreBenchmark.presets:
        benchmark = MolScoreBenchmark(model_name='smilesGA', output_dir="./", budget=1000, benchmark=args.molscore)
        for task, population in zip(benchmark, populations):
            
            # Initialize the GA generator with the correct population
            generator = SMILES_GA(smi_file=population,
                          population_size=args.population_size,
                          n_mutations=args.n_mutations,
                          gene_size=args.gene_size,
                          generations=args.generations,
                          n_jobs=args.n_jobs,
                          random_start=args.random_start,
                          patience=args.patience)
            final_population_smiles = generator.generate_optimized_molecules(scoring_function=task, number_molecules=args.population_size)
        benchmark.summarize(chemistry_filter_basic=False, diversity_check=False, mols_in_3d=True)
        
    elif os.path.isdir(args.molscore):
        benchmark = MolScoreBenchmark(model_name='smilesGA', output_dir="./", budget=1000, custom_benchmark=args.molscore)
        for task in benchmark:
            final_population_smiles = generator.generate_optimized_molecules(scoring_function=task, number_molecules=args.population_size)
        benchmark.summarize()
        
    else:
        ms = MolScore(model_name='smilesGA', task_config=args.molscore)
        final_population_smiles = generator.generate_optimized_molecules(scoring_function=ms.score,
                                                                         number_molecules=args.population_size,
                                                                         output_dir=ms.save_dir)
            
    return True


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--molscore', type=str, help='Path to MolScore config, or directory of configs, or name of Benchmark')
    # parser.add_argument('--smiles_file', type=str, help='Path to initial SMILES population file')
    parser.add_argument('--populations_dir', type=str, required=True, help='Path to directory containing sdf files of the molecules to sample from')


    # Optional arguments for GA setup
    optional = parser.add_argument_group('Optional')
    optional.add_argument('--seed', type=int, default=42, help='Random seed')
    optional.add_argument('--population_size', type=int, default=100, help='Population size')
    optional.add_argument('--n_mutations', type=int, default=50, help='Number of mutations per generation')
    optional.add_argument('--gene_size', type=int, default=-1, help='Gene size for the CFG-based encoding')
    optional.add_argument('--generations', type=int, default=10, help='Number of generations')
    optional.add_argument('--n_jobs', type=int, default=8, help='Number of parallel jobs')
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