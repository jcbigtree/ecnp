"""Differential Evolution

Storn, R. and Price, K. "Differential Evolution: A Simple and Efficient Adaptive Scheme
for Global Optimization over Continuous Spaces." J. Global Optimization 11, 341-359, 1997.
"""

import numpy as np
import copy
from base import BaseEA


__all__=[
    "DifferentialEvolution"     
]

class DifferentialEvolution(BaseEA):
    """Differential Evolution
    
    
    Attributes:
    -----------
    mutation_factor_: float, default 0.7
        A scale factor for mutation operator        
        
    crossover_prob_: float, default 0.2
        Crossover probability    
    """
    def __init__(self):
        super(DifferentialEvolution, self).__init__()        
        self.mutation_factor_ = 0.7
        self.crossover_prob_ = 0.7


    def _breed(self):
        """Overriden. Generate a whole population based on the best-so-far individual."""
        super(DifferentialEvolution, self)._breed()   # call base method
        # Differential Evolution breed operator
        self.offspring = np.empty_like(self.population)
        for i in range(self.population_size):
            xr1, xr2, xr3 = np.random.choice(self.population_size, 3, replace=False) 
            trial = self.population[xr1] + \
                        self.mutation_factor_*(self.population[xr2] - self.population[xr3])            
            mask = (np.random.rand(1, self.genome_size) < self.crossover_prob_).astype(float)
            self.offspring[i] = trial*mask + self.population[i]*(1-mask)
            

    def _select(self):
        """Overriden."""
        # Find the best individual in the offspring
        better_index = np.where(np.array(self.offspring_fitness) > np.array(self.pop_fitness))
        for i in better_index[0]:
            self.population[i,:] = self.offspring[i,:]
            self.pop_fitness[i] = self.offspring_fitness[i]


    def _check_stop_criteria(self):
        """Overriden."""
        # If it exceeds the max iteration, stop the evolution.
        stop_flag_0 = super(DifferentialEvolution, self)._check_stop_criteria()

        return stop_flag_0

        
    def _save_elite(self):
        """Save elite"""
        self.elite_index = np.argmax(self.pop_fitness)
        self.best_solution_ = self.population[self.elite_index,:]
        self.best_fitness_ = self.pop_fitness[self.elite_index]
        self.staged_best_solution_.append(self.best_solution_)
        self.staged_best_fitness_.append(self.best_fitness_)


       



