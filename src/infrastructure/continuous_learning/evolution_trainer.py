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
            callback (callable, optional): Function called after each generation
            
        Returns:
            dict: Evolution results
        """
        # Load checkpoint
        checkpoint = self.load_checkpoint(checkpoint_path)
        if not checkpoint:
            self.logger.error("Failed to load checkpoint")
            return None
        
        # Extract data from checkpoint
        population = checkpoint['population']
        best_individual = checkpoint['best_individual']
        best_fitness = checkpoint['best_fitness']
        current_generation = checkpoint['current_generation']
        total_generations = checkpoint['total_generations']
        
        # Determine how many more generations to run
        if additional_generations is None:
            # Complete the original evolution
            generations_left = total_generations - current_generation
        else:
            generations_left = additional_generations
        
        if generations_left <= 0:
            self.logger.warning("No additional generations to run")
            return {
                'best_individual': best_individual,
                'best_fitness': best_fitness,
                'final_population': population,
                'generations': current_generation,
                'population_size': len(population),
                'timestamp': datetime.now().isoformat()
            }
        
        self.logger.info(f"Continuing evolution for {generations_left} more generations")
        
        # Run evolution with the loaded population
        new_total_generations = current_generation + generations_left
        
        # Create a callback wrapper that adjusts generation numbering
        wrapped_callback = None
        if callback:
            def wrapped_callback(data):
                data['generation'] = current_generation + data['generation']
                return callback(data)
        
        # Continue evolution
        results = self.evolve(
            population=population,
            fitness_function=fitness_function,
            crossover_function=crossover_function,
            mutation_function=mutation_function,
            generations=generations_left,
            callback=wrapped_callback
        )
        
        # Update the results to reflect the total evolution
        if results:
            results['generations'] = new_total_generations
            results['continued_from_checkpoint'] = True
            results['original_checkpoint'] = checkpoint_path
        
        return results
    
    def _save_results(self, results):
        """
        Save evolution results to a file.
        
        Args:
            results (dict): Evolution results
            
        Returns:
            str: Path to saved file
        """
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evolution_results_{timestamp}.pkl"
            filepath = os.path.join(self.models_dir, filename)
            
            # Save to file
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
            
            self.logger.info(f"Saved evolution results to {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            return None
    
    def visualize_evolution(self, results, save_path=None):
        """
        Visualize evolution progress.
        
        Args:
            results (dict): Evolution results
            save_path (str, optional): Path to save visualization
            
        Returns:
            tuple: Figure and axes
        """
        try:
            if not results:
                self.logger.error("No results to visualize")
                return None, None
            
            # Extract data
            gen_best_fitness = results.get('generation_best_fitness', [])
            gen_avg_fitness = results.get('generation_avg_fitness', [])
            generations = list(range(1, len(gen_best_fitness) + 1))
            
            if not gen_best_fitness:
                self.logger.warning("No fitness data found in results")
                return None, None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot fitness evolution
            ax.plot(generations, gen_best_fitness, 'b-', label='Best Fitness', linewidth=2)
            ax.plot(generations, gen_avg_fitness, 'r-', label='Average Fitness', linewidth=2)
            
            # Add labels and legend
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.set_title('Evolution Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add annotation for final results
            best_fitness = results.get('best_fitness', gen_best_fitness[-1])
            execution_time = results.get('execution_time', 0)
            pop_size = results.get('population_size', 0)
            
            annotation = (f"Best Fitness: {best_fitness:.4f}\n"
                        f"Population Size: {pop_size}\n"
                        f"Generations: {len(generations)}\n"
                        f"Execution Time: {execution_time:.2f}s")
            
            ax.annotate(annotation, xy=(0.02, 0.02), xycoords='axes fraction',
                      bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                      fontsize=10)
            
            # Save figure if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved visualization to {save_path}")
            
            return fig, ax
            
        except Exception as e:
            self.logger.error(f"Error visualizing evolution: {str(e)}")
            return None, None
    
    def evolve_neural_network(self, nn_factory, fitness_function, param_ranges,
                          population_size=None, generations=None):
        """
        Evolve neural network architectures and parameters.
        
        Args:
            nn_factory (callable): Function to create NN from parameters
            fitness_function (callable): Function to evaluate NN
            param_ranges (dict): Ranges for each parameter
            population_size (int, optional): Size of population
            generations (int, optional): Number of generations
            
        Returns:
            dict: Evolution results
        """
        population_size = population_size or self.population_size
        generations = generations or self.generations
        
        self.logger.info(f"Starting neural network evolution with population size {population_size}")
        
        # Create initial population of neural network parameters
        def create_individual():
            params = {}
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    # Numerical parameter
                    if all(isinstance(x, int) for x in param_range):
                        # Integer parameter
                        params[param_name] = random.randint(param_range[0], param_range[1])
                    else:
                        # Float parameter
                        params[param_name] = random.uniform(param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    # Categorical parameter
                    params[param_name] = random.choice(param_range)
            return params
        
        # Create the initial population
        population = self.create_initial_population(create_individual, population_size)
        
        # Define crossover function for NN parameters
        def crossover_nn_params(parent1, parent2):
            child1 = {}
            child2 = {}
            
            for param_name in parent1.keys():
                # Randomly select crossover point for each parameter
                if random.random() < 0.5:
                    child1[param_name] = parent1[param_name]
                    child2[param_name] = parent2[param_name]
                else:
                    child1[param_name] = parent2[param_name]
                    child2[param_name] = parent1[param_name]
            
            return child1, child2
        
        # Define mutation function for NN parameters
        def mutate_nn_params(individual):
            mutated = individual.copy()
            
            # Select a random parameter to mutate
            param_name = random.choice(list(mutated.keys()))
            param_range = param_ranges[param_name]
            
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # Numerical parameter
                if all(isinstance(x, int) for x in param_range):
                    # Integer parameter
                    current = mutated[param_name]
                    # Apply random perturbation
                    perturbation = random.randint(-max(1, (param_range[1] - param_range[0]) // 10),
                                               max(1, (param_range[1] - param_range[0]) // 10))
                    new_value = max(param_range[0], min(param_range[1], current + perturbation))
                    mutated[param_name] = new_value
                else:
                    # Float parameter
                    current = mutated[param_name]
                    # Apply random perturbation
                    perturbation = random.gauss(0, (param_range[1] - param_range[0]) * 0.1)
                    new_value = max(param_range[0], min(param_range[1], current + perturbation))
                    mutated[param_name] = new_value
            elif isinstance(param_range, list):
                # Categorical parameter
                current = mutated[param_name]
                # Select a different value
                other_values = [v for v in param_range if v != current]
                if other_values:
                    mutated[param_name] = random.choice(other_values)
            
            return mutated
        
        # Create a wrapped fitness function that builds and evaluates NN
        def nn_fitness_function(params):
            try:
                # Create neural network from parameters
                nn = nn_factory(**params)
                
                # Evaluate fitness
                fitness = fitness_function(nn)
                
                return fitness
            except Exception as e:
                self.logger.error(f"Error evaluating neural network: {str(e)}")
                return float('-inf')  # Worst possible fitness
        
        # Run the evolution
        results = self.evolve(
            population=population,
            fitness_function=nn_fitness_function,
            crossover_function=crossover_nn_params,
            mutation_function=mutate_nn_params,
            generations=generations
        )
        
        return results
    
    def evolve_trading_strategy(self, strategy_factory, backtest_function, price_data,
                            param_ranges, strategy_templates=None,
                            population_size=None, generations=None):
        """
        Evolve trading strategies.
        
        Args:
            strategy_factory (callable): Function to create strategy from parameters
            backtest_function (callable): Function to backtest strategy
            price_data (pd.DataFrame): Historical price data
            param_ranges (dict): Parameter ranges
            strategy_templates (list, optional): Template strategies
            population_size (int, optional): Size of population
            generations (int, optional): Number of generations
            
        Returns:
            dict: Evolution results
        """
        population_size = population_size or self.population_size
        generations = generations or self.generations
        
        self.logger.info(f"Starting trading strategy evolution with population size {population_size}")
        
        # Create initial population
        def create_individual():
            # If templates are provided, randomly select one as a base
            if strategy_templates and random.random() < 0.5:
                base = random.choice(strategy_templates)
                params = copy.deepcopy(base)
            else:
                # Create new parameters
                params = {}
                
            # Fill or modify parameters
            for param_name, param_range in param_ranges.items():
                # Skip parameters that are already defined if using a template
                if param_name in params and random.random() < 0.7:
                    continue
                    
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    # Numerical parameter
                    if all(isinstance(x, int) for x in param_range):
                        # Integer parameter
                        params[param_name] = random.randint(param_range[0], param_range[1])
                    else:
                        # Float parameter
                        params[param_name] = random.uniform(param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    # Categorical parameter
                    params[param_name] = random.choice(param_range)
            
            return params
        
        # Create the initial population
        population = self.create_initial_population(create_individual, population_size)
        
        # Define crossover function
        def crossover_strategy(parent1, parent2):
            child1 = {}
            child2 = {}
            
            # Get all parameter names from both parents
            all_params = set(list(parent1.keys()) + list(parent2.keys()))
            
            for param_name in all_params:
                # If parameter exists in both parents, do crossover
                if param_name in parent1 and param_name in parent2:
                    if random.random() < 0.5:
                        child1[param_name] = parent1[param_name]
                        child2[param_name] = parent2[param_name]
                    else:
                        child1[param_name] = parent2[param_name]
                        child2[param_name] = parent1[param_name]
                # If parameter only exists in one parent, copy to child
                elif param_name in parent1:
                    child1[param_name] = parent1[param_name]
                    # 50% chance to inherit to second child
                    if random.random() < 0.5:
                        child2[param_name] = parent1[param_name]
                elif param_name in parent2:
                    child2[param_name] = parent2[param_name]
                    # 50% chance to inherit to first child
                    if random.random() < 0.5:
                        child1[param_name] = parent2[param_name]
            
            return child1, child2
        
        # Define mutation function
        def mutate_strategy(individual):
            mutated = individual.copy()
            
            # Either modify existing parameter or add new one
            if random.random() < 0.7 and mutated:
                # Modify existing parameter
                param_name = random.choice(list(mutated.keys()))
                
                if param_name in param_ranges:
                    param_range = param_ranges[param_name]
                    
                    if isinstance(param_range, tuple) and len(param_range) == 2:
                        # Numerical parameter
                        if all(isinstance(x, int) for x in param_range):
                            # Integer parameter
                            current = mutated[param_name]
                            # Apply random perturbation
                            perturbation = random.randint(-max(1, (param_range[1] - param_range[0]) // 10),
                                                       max(1, (param_range[1] - param_range[0]) // 10))
                            new_value = max(param_range[0], min(param_range[1], current + perturbation))
                            mutated[param_name] = new_value
                        else:
                            # Float parameter
                            current = mutated[param_name]
                            # Apply random perturbation
                            perturbation = random.gauss(0, (param_range[1] - param_range[0]) * 0.1)
                            new_value = max(param_range[0], min(param_range[1], current + perturbation))
                            mutated[param_name] = new_value
                    elif isinstance(param_range, list):
                        # Categorical parameter
                        current = mutated[param_name]
                        # Select a different value
                        other_values = [v for v in param_range if v != current]
                        if other_values:
                            mutated[param_name] = random.choice(other_values)
            else:
                # Add new parameter or modify existing with completely new value
                param_name = random.choice(list(param_ranges.keys()))
                param_range = param_ranges[param_name]
                
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    # Numerical parameter
                    if all(isinstance(x, int) for x in param_range):
                        # Integer parameter
                        mutated[param_name] = random.randint(param_range[0], param_range[1])
                    else:
                        # Float parameter
                        mutated[param_name] = random.uniform(param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    # Categorical parameter
                    mutated[param_name] = random.choice(param_range)
            
            return mutated
        
        # Create a wrapped fitness function
        def strategy_fitness_function(params):
            try:
                # Create strategy from parameters
                strategy = strategy_factory(**params)
                
                # Backtest strategy
                results = backtest_function(strategy, price_data)
                
                # Extract fitness (usually Sharpe ratio or similar)
                if isinstance(results, dict):
                    # Check for common performance metrics in order of preference
                    if 'sharpe_ratio' in results:
                        fitness = results['sharpe_ratio']
                    elif 'profit_factor' in results:
                        fitness = results['profit_factor']
                    elif 'profit' in results:
                        fitness = results['profit']
                    elif 'returns' in results:
                        fitness = results['returns']
                    elif 'total_return' in results:
                        fitness = results['total_return']
                    else:
                        # Default to first numeric value
                        for key, value in results.items():
                            if isinstance(value, (int, float)):
                                fitness = value
                                break
                        else:
                            fitness = 0
                else:
                    # Assume numeric result
                    fitness = float(results)
                
                return fitness
            except Exception as e:
                self.logger.error(f"Error evaluating trading strategy: {str(e)}")
                return float('-inf')  # Worst possible fitness
        
        # Run the evolution
        results = self.evolve(
            population=population,
            fitness_function=strategy_fitness_function,
            crossover_function=crossover_strategy,
            mutation_function=mutate_strategy,
            generations=generations
        )
        
        return results
    
    def evolve_multi_objective(self, population_factory, fitness_functions, dominance_function=None,
                          population_size=None, generations=None):
        """
        Evolve solutions with multiple conflicting objectives.
        
        Args:
            population_factory (callable): Function to create initial population
            fitness_functions (list): List of fitness functions
            dominance_function (callable, optional): Function to determine dominance
            population_size (int, optional): Size of population
            generations (int, optional): Number of generations
            
        Returns:
            dict: Evolution results with Pareto front
        """
        population_size = population_size or self.population_size
        generations = generations or self.generations
        
        self.logger.info(f"Starting multi-objective evolution with {len(fitness_functions)} objectives")
        
        # Create initial population
        population = population_factory(population_size)
        
        # If no dominance function is provided, use Pareto dominance
        if dominance_function is None:
            # A solution dominates another if it's at least as good in all objectives
            # and strictly better in at least one objective
            def pareto_dominates(fitness_a, fitness_b):
                # Check if a dominates b
                at_least_as_good = all(a >= b for a, b in zip(fitness_a, fitness_b))
                strictly_better = any(a > b for a, b in zip(fitness_a, fitness_b))
                return at_least_as_good and strictly_better
            
            dominance_function = pareto_dominates
        
        # Function to compute non-dominated rank
        def compute_non_dominated_rank(fitness_values_list):
            # fitness_values_list is a list of multi-objective fitness tuples
            ranks = [0] * len(fitness_values_list)
            
            # Compare each solution with every other solution
            for i in range(len(fitness_values_list)):
                for j in range(len(fitness_values_list)):
                    if i != j:
                        # Check if solution j dominates solution i
                        if dominance_function(fitness_values_list[j], fitness_values_list[i]):
                            ranks[i] += 1
            
            return ranks
        
        # Function to compute crowding distance
        def compute_crowding_distance(fitness_values_list, non_dominated_indices):
            n_objectives = len(fitness_values_list[0])
            n_solutions = len(non_dominated_indices)
            
            # Initialize crowding distances
            distances = [0.0] * n_solutions
            
            # For each objective
            for obj_idx in range(n_objectives):
                # Extract values for this objective
                values = [fitness_values_list[idx][obj_idx] for idx in non_dominated_indices]
                
                # Sort indices by this objective
                sorted_indices = sorted(range(n_solutions), key=lambda i: values[i])
                
                # Set boundary points to infinity
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                
                # Compute distances for intermediate points
                obj_range = max(values) - min(values)
                if obj_range > 0:
                    for i in range(1, n_solutions - 1):
                        distances[sorted_indices[i]] += (values[sorted_indices[i+1]] - values[sorted_indices[i-1]]) / obj_range
            
            return distances
        
        # Initialize tracking variables
        best_front = []
        gen_best_fitness = []
        
        # Main evolution loop
        for generation in range(generations):
            self.logger.info(f"Generation {generation+1}/{generations}")
            
            # Evaluate population on all objectives
            fitness_values_list = []
            
            for individual in population:
                # Evaluate on all objectives
                fitness_tuple = tuple(fitness_func(individual) for fitness_func in fitness_functions)
                fitness_values_list.append(fitness_tuple)
            
            # Compute non-dominated rank for each individual
            ranks = compute_non_dominated_rank(fitness_values_list)
            
            # Get Pareto front (rank 0)
            front_indices = [i for i, rank in enumerate(ranks) if rank == 0]
            front = [population[i] for i in front_indices]
            front_fitness = [fitness_values_list[i] for i in front_indices]
            
            # Update best front (archived best solutions)
            if not best_front:
                best_front = list(zip(front, front_fitness))
            else:
                # Combine current front with best front
                combined = best_front + list(zip(front, front_fitness))
                
                # Recompute non-dominated solutions
                combined_fitness = [f for _, f in combined]
                combined_ranks = compute_non_dominated_rank(combined_fitness)
                
                # Keep only non-dominated solutions
                best_front = [combined[i] for i, rank in enumerate(combined_ranks) if rank == 0]
            
            # Log progress
            self.logger.info(f"Generation {generation+1} - Front size: {len(front)}, Archive size: {len(best_front)}")
            
            # Track statistics
            gen_best_fitness.append(front_fitness)
            
            # Create next generation if not the last
            if generation < generations - 1:
                next_population = []
                
                # Apply elitism - keep some individuals from current Pareto front
                elites_count = max(1, int(self.elitism_ratio * population_size))
                if len(front) <= elites_count:
                    next_population.extend(front)
                else:
                    # Select diverse subset using crowding distance
                    distances = compute_crowding_distance(front_fitness, list(range(len(front))))
                    
                    # Sort by crowding distance (higher is better)
                    sorted_indices = sorted(range(len(front)), key=lambda i: distances[i], reverse=True)
                    
                    # Add top individuals by crowding distance
                    for i in range(elites_count):
                        next_population.append(front[sorted_indices[i]])
                
                # Fill rest of population using tournament selection and variation
                while len(next_population) < population_size:
                    # Tournament selection based on rank and crowding distance
                    def tournament_selection():
                        indices = random.sample(range(len(population)), self.tournament_size)
                        
                        # Get ranks and distances
                        candidate_ranks = [ranks[i] for i in indices]
                        
                        # First, select by rank (lower is better)
                        min_rank = min(candidate_ranks)
                        min_rank_indices = [i for i, rank in zip(indices, candidate_ranks) if rank == min_rank]
                        
                        # If multiple with same rank, select by crowding distance
                        if len(min_rank_indices) > 1 and min_rank == 0:
                            # Calculate crowding distance for these candidates
                            front_idx = [list(front_indices).index(i) if i in front_indices else -1 for i in min_rank_indices]
                            valid_idx = [i for i in range(len(front_idx)) if front_idx[i] != -1]
                            
                            if valid_idx:
                                # Get crowding distances for valid candidates
                                crowd_dist = compute_crowding_distance(front_fitness, 
                                                                    [front_idx[i] for i in valid_idx])
                                
                                # Select individual with highest crowding distance
                                best_idx = valid_idx[crowd_dist.index(max(crowd_dist))]
                                return population[min_rank_indices[best_idx]]
                        
                        # Default to random selection from min rank
                        return population[random.choice(min_rank_indices)]
                    
                    # Select parents and create offspring
                    parent1 = tournament_selection()
                    parent2 = tournament_selection()
                    
                    # Crossover and mutation would need to be provided
                    # For simplicity, just clone parents with some mutation
                    child = copy.deepcopy(parent1)
                    
                    # Simulate mutation (this would need to be specific to the problem)
                    # In a real implementation, a proper mutation operator would be needed
                    
                    next_population.append(child)
                
                # Update population
                population = next_population
        
        # Prepare results
        results = {
            'pareto_front': best_front,
            'front_size': len(best_front),
            'generation_fronts': gen_best_fitness,
            'generations': generations,
            'population_size': population_size,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def visualize_pareto_front(self, results, objective_names=None, save_path=None):
        """
        Visualize Pareto front for multi-objective optimization.
        
        Args:
            results (dict): Results from evolve_multi_objective
            objective_names (list, optional): Names of objectives
            save_path (str, optional): Path to save visualization
            
        Returns:
            tuple: Figure and axes
        """
        try:
            if not results or 'pareto_front' not in results:
                self.logger.error("No valid Pareto front results to visualize")
                return None, None
            
            # Extract Pareto front data
            pareto_front = results['pareto_front']
            
            if not pareto_front:
                self.logger.warning("Empty Pareto front")
                return None, None
            
            # Extract fitness values
            fitness_values = [f for _, f in pareto_front]
            
            # Determine number of objectives
            n_objectives = len(fitness_values[0])
            
            if not objective_names:
                objective_names = [f"Objective {i+1}" for i in range(n_objectives)]
            
            # Create visualization based on number of objectives
            if n_objectives == 2:
                # 2D scatter plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Extract data
                x = [f[0] for f in fitness_values]
                y = [f[1] for f in fitness_values]
                
                # Plot Pareto front
                ax.scatter(x, y, c='blue', s=50, alpha=0.7)
                
                # Connect points to visualize front
                sorted_points = sorted(zip(x, y), key=lambda p: p[0])
                sorted_x, sorted_y = zip(*sorted_points)
                ax.plot(sorted_x, sorted_y, 'k--', alpha=0.5)
                
                # Add labels
                ax.set_xlabel(objective_names[0])
                ax.set_ylabel(objective_names[1])
                ax.set_title('Pareto Front')
                ax.grid(True, alpha=0.3)
                
            elif n_objectives == 3:
                # 3D scatter plot
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # Extract data
                x = [f[0] for f in fitness_values]
                y = [f[1] for f in fitness_values]
                z = [f[2] for f in fitness_values]
                
                # Plot Pareto front
                ax.scatter(x, y, z, c='blue', s=50, alpha=0.7)
                
                # Add labels
                ax.set_xlabel(objective_names[0])
                ax.set_ylabel(objective_names[1])
                ax.set_zlabel(objective_names[2])
                ax.set_title('Pareto Front')
                
            else:
                # Parallel coordinates plot for 4+ objectives
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Prepare data
                data = pd.DataFrame(fitness_values, columns=objective_names)
                
                # Normalize data for visualization
                for col in data.columns:
                    min_val = data[col].min()
                    max_val = data[col].max()
                    if max_val > min_val:
                        data[col] = (data[col] - min_val) / (max_val - min_val)
                    else:
                        data[col] = 0.5  # If all values are the same
                
                # Plot parallel coordinates
                for i in range(len(data)):
                    xs = range(len(objective_names))
                    ys = data.iloc[i].values
                    ax.plot(xs, ys, alpha=0.5)
                
                # Set x-axis ticks and labels
                ax.set_xticks(range(len(objective_names)))
                ax.set_xticklabels(objective_names, rotation=45)
                ax.set_xlim([0, len(objective_names) - 1])
                ax.set_ylim([0, 1])
                ax.set_title('Pareto Front (Normalized)')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved Pareto front visualization to {save_path}")
            
            return fig, ax
