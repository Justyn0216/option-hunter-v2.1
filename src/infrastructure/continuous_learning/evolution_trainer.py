"""
Evolution Trainer Module

This module implements evolutionary algorithms for training and optimizing 
trading strategies, models and parameters in the Option Hunter system.
It uses genetic algorithms, evolutionary strategies, and genetic programming
to evolve effective trading systems over time.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import time
import copy
import multiprocessing
from functools import partial
import pickle
from collections import defaultdict, deque

class EvolutionTrainer:
    """
    Trainer that uses evolutionary algorithms to optimize trading strategies.
    
    Features:
    - Genetic algorithm for strategy evolution
    - Neuroevolution for neural network optimization
    - Genetic programming for rule discovery
    - Multi-objective optimization for balancing different performance metrics
    - Evolutionary strategies for direct parameter optimization
    """
    
    def __init__(self, config=None):
        """
        Initialize the EvolutionTrainer.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Extract configuration
        self.evolution_params = self.config.get("evolution_trainer", {})
        
        # Default parameters
        self.population_size = self.evolution_params.get("population_size", 50)
        self.generations = self.evolution_params.get("generations", 20)
        self.mutation_rate = self.evolution_params.get("mutation_rate", 0.2)
        self.crossover_rate = self.evolution_params.get("crossover_rate", 0.7)
        self.elitism_ratio = self.evolution_params.get("elitism_ratio", 0.1)
        self.tournament_size = self.evolution_params.get("tournament_size", 3)
        self.parallel_jobs = self.evolution_params.get("parallel_jobs", min(4, multiprocessing.cpu_count()))
        self.random_state = self.evolution_params.get("random_state", 42)
        
        # Initialize random seed for reproducibility
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Create directories
        self.logs_dir = "logs/evolution_trainer"
        self.models_dir = "models/evolution_trainer"
        self.checkpoints_dir = "checkpoints/evolution_trainer"
        
        for directory in [self.logs_dir, self.models_dir, self.checkpoints_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Store history of training
        self.training_history = []
        
        self.logger.info(f"EvolutionTrainer initialized with population size {self.population_size} "
                        f"and {self.generations} generations")
    
    def create_initial_population(self, creator_function, size=None):
        """
        Create an initial population of individuals.
        
        Args:
            creator_function (callable): Function to create a single individual
            size (int, optional): Population size (default: self.population_size)
            
        Returns:
            list: Initial population
        """
        size = size or self.population_size
        
        self.logger.info(f"Creating initial population of {size} individuals")
        
        population = []
        
        for _ in range(size):
            try:
                individual = creator_function()
                population.append(individual)
            except Exception as e:
                self.logger.error(f"Error creating individual: {str(e)}")
        
        self.logger.info(f"Created {len(population)} individuals")
        
        return population
    
    def evaluate_population(self, population, fitness_function, parallel=True):
        """
        Evaluate fitness for each individual in the population.
        
        Args:
            population (list): Population of individuals
            fitness_function (callable): Function to evaluate fitness
            parallel (bool): Whether to use parallel processing
            
        Returns:
            list: List of fitness values
        """
        self.logger.info(f"Evaluating population of {len(population)} individuals")
        
        start_time = time.time()
        
        # Evaluate fitness for each individual
        if parallel and self.parallel_jobs > 1 and len(population) > 1:
            # Use parallel processing
            with multiprocessing.Pool(self.parallel_jobs) as pool:
                fitness_values = pool.map(fitness_function, population)
        else:
            # Sequential processing
            fitness_values = [fitness_function(ind) for ind in population]
        
        execution_time = time.time() - start_time
        
        # Check for failed evaluations
        for i, fitness in enumerate(fitness_values):
            if fitness is None or (isinstance(fitness, (int, float)) and np.isnan(fitness)):
                self.logger.warning(f"Individual {i} has invalid fitness, assigning worst fitness")
                fitness_values[i] = float('-inf')  # Assume we're maximizing
        
        self.logger.info(f"Population evaluation completed in {execution_time:.2f} seconds")
        
        return fitness_values
    
    def select_parents(self, population, fitness_values, n_parents=None):
        """
        Select parents for reproduction using tournament selection.
        
        Args:
            population (list): Population of individuals
            fitness_values (list): Fitness values for each individual
            n_parents (int, optional): Number of parents to select
            
        Returns:
            list: Selected parents
        """
        n_parents = n_parents or self.population_size
        
        # Create list of indices
        indices = list(range(len(population)))
        
        selected_parents = []
        
        for _ in range(n_parents):
            # Tournament selection
            tournament_indices = random.sample(indices, min(self.tournament_size, len(indices)))
            tournament_fitness = [fitness_values[i] for i in tournament_indices]
            
            # Select winner (assuming maximization)
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_parents.append(population[winner_idx])
        
        return selected_parents
    
    def crossover(self, parent1, parent2, crossover_function):
        """
        Create offspring through crossover of two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            crossover_function (callable): Function to perform crossover
            
        Returns:
            tuple: (offspring1, offspring2)
        """
        # Skip crossover with some probability
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        try:
            offspring1, offspring2 = crossover_function(parent1, parent2)
            return offspring1, offspring2
        except Exception as e:
            self.logger.error(f"Error during crossover: {str(e)}")
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
    
    def mutate(self, individual, mutation_function):
        """
        Mutate an individual.
        
        Args:
            individual: Individual to mutate
            mutation_function (callable): Function to perform mutation
            
        Returns:
            object: Mutated individual
        """
        # Skip mutation with some probability
        if random.random() > self.mutation_rate:
            return individual
        
        try:
            mutated = mutation_function(individual)
            return mutated
        except Exception as e:
            self.logger.error(f"Error during mutation: {str(e)}")
            return individual
    
    def evolve(self, population, fitness_function, crossover_function, mutation_function, 
             generations=None, checkpoint_interval=5, callback=None):
        """
        Evolve a population over multiple generations.
        
        Args:
            population (list): Initial population
            fitness_function (callable): Function to evaluate fitness
            crossover_function (callable): Function to perform crossover
            mutation_function (callable): Function to perform mutation
            generations (int, optional): Number of generations
            checkpoint_interval (int): Interval for saving checkpoints
            callback (callable, optional): Function called after each generation
            
        Returns:
            dict: Evolution results
        """
        generations = generations or self.generations
        
        self.logger.info(f"Starting evolution with population size {len(population)} for {generations} generations")
        
        # Initialize tracking variables
        start_time = time.time()
        best_individual = None
        best_fitness = float('-inf')  # Assuming maximization
        gen_best_fitness = []
        gen_avg_fitness = []
        gen_best_individuals = []
        
        current_population = population.copy()
        
        # Main evolution loop
        for generation in range(generations):
            gen_start_time = time.time()
            
            self.logger.info(f"Generation {generation+1}/{generations}")
            
            # Evaluate current population
            fitness_values = self.evaluate_population(current_population, fitness_function)
            
            # Track statistics
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            current_best_individual = current_population[current_best_idx]
            current_avg_fitness = np.mean(fitness_values)
            
            gen_best_fitness.append(current_best_fitness)
            gen_avg_fitness.append(current_avg_fitness)
            gen_best_individuals.append(copy.deepcopy(current_best_individual))
            
            # Update overall best
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = copy.deepcopy(current_best_individual)
                
                self.logger.info(f"New best individual found with fitness: {best_fitness}")
            
            # Log generation stats
            gen_time = time.time() - gen_start_time
            self.logger.info(f"Generation {generation+1} - "
                           f"Best: {current_best_fitness:.4f}, "
                           f"Avg: {current_avg_fitness:.4f}, "
                           f"Time: {gen_time:.2f}s")
            
            # Call callback if provided
            if callback:
                callback({
                    'generation': generation + 1,
                    'best_fitness': current_best_fitness,
                    'avg_fitness': current_avg_fitness,
                    'best_individual': current_best_individual,
                    'population': current_population,
                    'fitness_values': fitness_values,
                    'overall_best': {
                        'fitness': best_fitness,
                        'individual': best_individual
                    }
                })
            
            # Create next generation
            if generation < generations - 1:  # Skip for last generation
                next_population = []
                
                # Apply elitism - keep best individuals
                elites_count = max(1, int(self.elitism_ratio * len(current_population)))
                elite_indices = np.argsort(fitness_values)[-elites_count:]
                elites = [copy.deepcopy(current_population[i]) for i in elite_indices]
                next_population.extend(elites)
                
                # Fill the rest with offspring
                while len(next_population) < len(current_population):
                    # Select parents
                    parent1 = self.select_parents(current_population, fitness_values, 1)[0]
                    parent2 = self.select_parents(current_population, fitness_values, 1)[0]
                    
                    # Crossover
                    offspring1, offspring2 = self.crossover(parent1, parent2, crossover_function)
                    
                    # Mutation
                    offspring1 = self.mutate(offspring1, mutation_function)
                    offspring2 = self.mutate(offspring2, mutation_function)
                    
                    # Add to next generation
                    next_population.append(offspring1)
                    if len(next_population) < len(current_population):
                        next_population.append(offspring2)
                
                # Update current population
                current_population = next_population
            
            # Save checkpoint if needed
            if checkpoint_interval > 0 and (generation + 1) % checkpoint_interval == 0:
                self._save_checkpoint(current_population, fitness_values, best_individual, 
                                    best_fitness, generation, generations)
        
        # Final evaluation of population
        final_fitness_values = self.evaluate_population(current_population, fitness_function)
        
        # Prepare results
        execution_time = time.time() - start_time
        
        results = {
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'final_population': current_population,
            'final_fitness_values': final_fitness_values,
            'generation_best_fitness': gen_best_fitness,
            'generation_avg_fitness': gen_avg_fitness,
            'execution_time': execution_time,
            'generations': generations,
            'population_size': len(population),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'generations': generations,
            'population_size': len(population),
            'best_fitness': best_fitness,
            'execution_time': execution_time
        })
        
        self.logger.info(f"Evolution completed in {execution_time:.2f} seconds")
        self.logger.info(f"Best fitness achieved: {best_fitness}")
        
        # Save final results
        self._save_results(results)
        
        return results
    
    def _save_checkpoint(self, population, fitness_values, best_individual, best_fitness, 
                      current_generation, total_generations):
        """
        Save a checkpoint of the current evolution state.
        
        Args:
            population (list): Current population
            fitness_values (list): Current fitness values
            best_individual: Best individual so far
            best_fitness: Best fitness so far
            current_generation (int): Current generation number
            total_generations (int): Total number of generations
            
        Returns:
            str: Path to checkpoint file
        """
        try:
            # Create checkpoint filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_gen{current_generation}_{timestamp}.pkl"
            filepath = os.path.join(self.checkpoints_dir, filename)
            
            # Prepare checkpoint data
            checkpoint = {
                'population': population,
                'fitness_values': fitness_values,
                'best_individual': best_individual,
                'best_fitness': best_fitness,
                'current_generation': current_generation,
                'total_generations': total_generations,
                'timestamp': timestamp
            }
            
            # Save checkpoint
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            self.logger.info(f"Saved checkpoint at generation {current_generation} to {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
            return None
    
    def load_checkpoint(self, filepath):
        """
        Load a saved checkpoint.
        
        Args:
            filepath (str): Path to checkpoint file
            
        Returns:
            dict: Checkpoint data
        """
        try:
            with open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.logger.info(f"Loaded checkpoint from {filepath}")
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            return None
    
    def continue_evolution(self, checkpoint_path, fitness_function, crossover_function, 
                         mutation_function, additional_generations=None, callback=None):
        """
        Continue evolution from a checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            fitness_function (callable): Function to evaluate fitness
            crossover_function (callable): Function to perform crossover
            mutation_function (callable): Function to perform mutation
            additional_generations (int, optional): Additional generations to run
            callback (callable,
