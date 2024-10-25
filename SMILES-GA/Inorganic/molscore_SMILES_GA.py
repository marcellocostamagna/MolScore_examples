
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


def select_parent(population, tournament_size=3):
    idx = np.random.randint(len(population), size=tournament_size)
    best = population[idx[0]]
    for i in idx[1:]:
        if population[i][0] > best[0]:
            best = population[i]
    return best


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


def mutate(p_gene, scoring_function):
    c_gene = mutation(p_gene)
    try:
        # Decode the mutated gene into SMILES directly without RDKit canonicalization
        # (Devation from original Guacamol code)
        c_smiles = decode(gene_to_cfg(c_gene))
        
    except Exception as e:
        # Handle any decoding errors gracefully
        print(f'The mutation resulted in an invalid molecule: {e}')
        c_smiles = ''
        c_score = 0.0
        
    try:
        c_score = scoring_function(c_smiles)
    except Exception as e:
        # Handle any scoring errors gracefully
        print(f'The scoring of the mutated molecule failed: {e}')
        print(f'SMILES: {c_smiles}')
        c_score = 0.0
         
    return Molecule(c_score, c_smiles, c_gene)

def mutate_no_score(p_gene):
    #TODO: In case the mutation produces an invalid molecules we could 
    # attempt to mutate again until a valid molecule is produced or a 
    # maximum number of attempts is reached
    c_gene = mutation(p_gene)
    try:
        # Decode the mutated gene into SMILES directly without RDKit canonicalization
        # (Devation from original Guacamol code)
        c_smiles = decode(gene_to_cfg(c_gene))
        
    except Exception as e:
        # Handle any decoding errors gracefully
        print(f'The mutation resulted in an invalid molecule: {e}')
        c_smiles = ''

    return Molecule(0.0, c_smiles, c_gene)
    
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
            # return self.pool(delayed(self.canonicalize)(s.strip()) for s in f)
            # Deviation from original Guacamol code: directly return SMILES strings
            return [s.strip() for s in f if s.strip()]


    def top_k(self, smiles, scoring_function, k):
        # scores = scoring_function(smiles, flt=True, score_only=True)
        # joblist = (delayed(scoring_function.score)(s) for s in smiles)
        # scores = self.pool(joblist)
        # scored_smiles = list(zip(scores, smiles))
        # scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        # Just get the first k SMILES
        return [smile for smile in smiles][:k]
        # return [smile for score, smile in scored_smiles][:k]

    def generate_optimized_molecules(self, scoring_function, number_molecules: int,
                                     starting_population: Optional[List[str]] = None,
                                     use_cache: bool = False) -> List[str]:

        # NEW CODE
        # Initialize a cache for already scored molecules if caching is enabled
        score_cache = {} if use_cache else None
        
        if number_molecules > self.population_size:
            self.population_size = number_molecules
            print(f'Benchmark requested more molecules than expected: new population is {number_molecules}')

        # NEW CODE
        # fetch initial population
        # Get the first 'population_size' SMILES from the list of all SMILES
        # The population size will ultimately be 100, but we will start with 20
        starting_population = self.all_smiles[:self.population_size]
        
         # ORIGINAL CODE
        # # fetch initial population
        # if starting_population is None:
        #     print('Selecting initial population...')
        #     init_size = self.population_size + self.n_mutations
        #     all_smiles = copy.deepcopy(self.all_smiles)
        #     if self.random_start:
        #         starting_population = np.random.choice(all_smiles, init_size)
        #     else:
        #         starting_population = self.top_k(all_smiles, scoring_function, init_size)
        
        
        # # ORIGINAL CODE
        # # Calculate initial genes
        # initial_genes = [cfg_to_gene(encode(s), max_len=self.gene_size) for s in starting_population]
        
        # NEW CODE
        # Calculate initial genes with multiprocessing
        joblist = (delayed(cfg_to_gene)(encode(s), max_len=self.gene_size) for s in starting_population)
        initial_genes = self.pool(joblist)
        # Score initial population, using cache if enabled
        population_smiles = [s for s in starting_population]
        updated_genes = [g for g in initial_genes]        

        # NEW CODE
        # Separate SMILES that need scoring from those already scored, if cache is used
        if use_cache:
            new_smiles = [smiles for smiles in population_smiles if smiles not in score_cache]
            new_scores = scoring_function(new_smiles, flt=True, score_only=True)
            # Update the score cache with newly scored SMILES
            score_cache.update({smiles: (score, gene) for smiles, score, gene in zip(new_smiles, new_scores, updated_genes)})
            # Retrieve all scores from the cache
            population_scores = [score_cache[smiles][0] for smiles in population_smiles]
        else:
            # Directly score all SMILES if no cache is used
            population_scores = scoring_function(population_smiles, flt=True, score_only=True)

        # ORIGINAL CODE
        # Score initial population
        # initial_scores = scoring_function(starting_population, flt=True, score_only=True)
        # population = [Molecule(*m) for m in zip(initial_scores, starting_population, initial_genes)]
        # population = sorted(population, key=lambda x: x.score, reverse=True)[:self.population_size]
        # population_scores = [p.score for p in population]
        
        # NEW CODE
        # Initialize the population with scores and genes
        population = [Molecule(score, smiles, gene) for score, smiles, gene in zip(population_scores, population_smiles, updated_genes)]
        population = sorted(population, key=lambda x: x.score, reverse=True)[:self.population_size]
        

        # Evolution process
        t0 = time()
        patience = 0

        for generation in range(self.generations):
            
            # NEW CODE
            # print the population (This is for DEBUGGING purposes)
            print(f'Generation {generation}:')
            for molecule in population:
                print(f'{molecule.smiles} --> {molecule.score}')

            # Track scores to check for early stopping
            old_scores = [molecule.score for molecule in population]
            all_genes = [molecule.genes for molecule in population]
            all_smiles = [molecule.smiles for molecule in population]
            choice_indices = np.random.choice(len(all_genes), self.n_mutations, replace=False)
            genes_to_mutate = [all_genes[i] for i in choice_indices]
            smiles_to_mutate = [all_smiles[i] for i in choice_indices]

            # EVOLVE/MUTATE GENES
            # Mutation using multiprocessing
            joblist = (delayed(robust_mutation)(smiles, gene) for smiles, gene in zip(smiles_to_mutate, genes_to_mutate))
            mutated_results = self.pool(joblist)
            
            # Filter out failed mutations
            mutated_genes = [Molecule(0.0, c_smiles, c_gene) for c_smiles, c_gene in mutated_results if c_smiles is not None]
                        
            print('Number of genes to mutate:', len(genes_to_mutate))
            print('Number of mutated genes:', len(mutated_genes))
            # if len(genes_to_mutate) > 0:
            #     print(f'Effective mutation rate: {len(mutated_molecules) / len(genes_to_mutate) * 100:.2f} %')
            # else:
            #     print('No genes available for mutation.')
            
            # # Deviation from original Guacamol code: directly mutate genes and score SMILES
            # new_population = [mutate(g, scoring_function) for g in genes_to_mutate]
            # TODO: here we cannot use mutate(g, scoring_function) since the object MOlScore 
            # cannot be pickled
            # print(f'Generated {len(new_population)} new molecules')
    
            # Deduplicate
            population += mutated_genes
            population = remove_duplicates(population)
            
            # print the size of the population
            print(f'Population size after deduplication: {len(population)}')            
            
            # Deviation from original Guacamol code: use MolScore to score the population
            population_smiles = [molecule.smiles for molecule in population]
            updated_genes = [molecule.genes for molecule in population]
            
            # ORIGINAL CODE
            # population_scores = scoring_function(population_smiles, flt=True, score_only=True)
            # Extract genes from deduplicated population
            # updated_genes = [molecule.genes for molecule in population]
            
            if use_cache:
                new_smiles = [smiles for smiles in population_smiles if smiles not in score_cache]
                new_scores = scoring_function(new_smiles, flt=True, score_only=True)
                score_cache.update({smiles: (score, gene) for smiles, score, gene in zip(new_smiles, new_scores, updated_genes)})
                population_scores = [score_cache[smiles][0] for smiles in population_smiles]
            else:
                population_scores = scoring_function(population_smiles, flt=True, score_only=True)
            
            
            # Survival of the fittest: select top molecules
            population = [Molecule(score, smiles, gene) for score, smiles, gene in zip(population_scores, population_smiles, updated_genes)]
            population = sorted(population, key=lambda x: x.score, reverse=True)[:self.population_size]

            print(f'Generation {generation}: after mutation and scoring')
            for molecule in population:
                print(f'{molecule.smiles} --> {molecule.score}')
            
            # Stats
            gen_time = time() - t0
            mol_sec = (self.population_size + self.n_mutations) / gen_time
            t0 = time()
            # population_scores = [p.score for p in population]

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
                                                                         number_molecules=args.population_size,
                                                                         use_cache=args.use_cache)

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
    optional.add_argument('--population_size', type=int, default=20, help='Population size')
    optional.add_argument('--n_mutations', type=int, default=10, help='Number of mutations per generation')
    optional.add_argument('--gene_size', type=int, default=-1, help='Gene size for the CFG-based encoding')
    optional.add_argument('--generations', type=int, default=5, help='Number of generations')
    optional.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs')
    optional.add_argument('--random_start', action='store_true', help='Start with a random population')
    optional.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    #new addition
    optional.add_argument('--use_cache', action='store_true', help='Enable caching of scored molecules to avoid recomputation')
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)