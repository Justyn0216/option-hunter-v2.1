"""
Parameter Space Explorer Module

This module implements systematic exploration of parameter spaces for trading models.
It uses various optimization techniques to discover optimal parameters for trading
strategies, indicators, and other components of the Option Hunter system.
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
from functools import partial
import multiprocessing
import itertools
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from skopt import gp_minimize, forest_minimize, dummy_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

class ParameterSpaceExplorer:
    """
    Explores parameter spaces to optimize trading system performance.
    
    Features:
    - Grid search for exhaustive parameter exploration
    - Random search for efficient parameter discovery
    - Bayesian optimization for intelligent parameter tuning
    - Evolutionary algorithms for complex parameter landscapes
    - Hyperparameter optimization for ML models
    """
    
    def __init__(self, config=None):
        """
        Initialize the ParameterSpaceExplorer.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Extract configuration
        self.explorer_params = self.config.get("parameter_space_explorer", {})
        
        # Default parameters
        self.max_evaluations = self.explorer_params.get("max_evaluations", 100)
        self.parallel_jobs = self.explorer_params.get("parallel_jobs", min(4, multiprocessing.cpu_count()))
        self.default_method = self.explorer_params.get("default_method", "bayesian")
        self.random_state = self.explorer_params.get("random_state", 42)
        
        # History of explorations
        self.exploration_history = []
        
        # Create directories
        self.logs_dir = "logs/parameter_explorer"
        self.results_dir = "results/parameter_explorer"
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger.info(f"ParameterSpaceExplorer initialized with default method: {self.default_method}")
    
    def define_space(self, parameter_definitions):
        """
        Define the parameter space for exploration.
        
        Args:
            parameter_definitions (list): List of parameter definitions
                Each definition should be a dict with:
                - 'name': Parameter name
                - 'type': Parameter type ('real', 'integer', 'categorical')
                - 'bounds': Parameter bounds [low, high] or list of categories
                - 'transform' (optional): Transform ('log', 'logit', etc.)
                
        Returns:
            list: scikit-optimize parameter space definition
            dict: Parameter space information
        """
        space = []
        param_info = {}
        
        for param in parameter_definitions:
            name = param['name']
            param_type = param['type'].lower()
            bounds = param['bounds']
            transform = param.get('transform', None)
            
            if param_type == 'real':
                if transform == 'log':
                    space.append(Real(bounds[0], bounds[1], prior='log-uniform', name=name))
                else:
                    space.append(Real(bounds[0], bounds[1], name=name))
                    
                param_info[name] = {
                    'type': 'real',
                    'bounds': bounds,
                    'transform': transform
                }
                
            elif param_type == 'integer':
                space.append(Integer(bounds[0], bounds[1], name=name))
                param_info[name] = {
                    'type': 'integer',
                    'bounds': bounds
                }
                
            elif param_type == 'categorical':
                space.append(Categorical(bounds, name=name))
                param_info[name] = {
                    'type': 'categorical',
                    'categories': bounds
                }
            
            else:
                self.logger.warning(f"Unknown parameter type: {param_type} for parameter {name}")
        
        return space, param_info
    
    def explore(self, parameter_definitions, objective_function, method=None, n_evaluations=None, 
               n_initial_points=10, verbose=True, callback=None):
        """
        Explore parameter space to find optimal parameters.
        
        Args:
            parameter_definitions (list): Parameter space definitions
            objective_function (callable): Function to minimize
            method (str, optional): Optimization method
                ('grid', 'random', 'bayesian', 'forest', 'evolutionary')
            n_evaluations (int, optional): Number of evaluations
            n_initial_points (int): Number of initial random points for Bayesian
            verbose (bool): Whether to log progress
            callback (callable, optional): Callback after each evaluation
                
        Returns:
            dict: Optimization results
        """
        start_time = time.time()
        method = method or self.default_method
        n_evaluations = n_evaluations or self.max_evaluations
        
        self.logger.info(f"Starting parameter exploration using {method} method with {n_evaluations} evaluations")
        
        # Define parameter space
        space, param_info = self.define_space(parameter_definitions)
        
        if not space:
            self.logger.error("No valid parameters defined")
            return None
        
        # Create parameter names list for easier access
        param_names = [param.name for param in space]
        
        # Select exploration method
        result = None
        
        try:
            if method == 'grid':
                result = self._grid_search(space, param_info, objective_function, n_evaluations, verbose, callback)
            elif method == 'random':
                result = self._random_search(space, param_names, objective_function, n_evaluations, verbose, callback)
            elif method == 'bayesian':
                result = self._bayesian_optimization(space, param_names, objective_function, n_evaluations, 
                                                  n_initial_points, verbose, callback)
            elif method == 'forest':
                result = self._forest_minimize(space, param_names, objective_function, n_evaluations, 
                                            n_initial_points, verbose, callback)
            elif method == 'evolutionary':
                result = self._evolutionary_optimization(space, param_info, objective_function, n_evaluations, 
                                                      verbose, callback)
            else:
                self.logger.error(f"Unknown optimization method: {method}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during parameter exploration: {str(e)}")
            return None
        
        # Process results
        end_time = time.time()
        execution_time = end_time - start_time
        
        if result:
            # Extract the best parameters
            if method in ['bayesian', 'forest']:
                best_params = dict(zip(param_names, result.x))
                best_value = result.fun
                all_evaluations = [{
                    'params': dict(zip(param_names, params)),
                    'value': value
                } for params, value in zip(result.x_iters, result.func_vals)]
            else:
                best_params = result['best_params']
                best_value = result['best_value']
                all_evaluations = result['evaluations']
            
            # Create results dictionary
            results = {
                'best_params': best_params,
                'best_value': best_value,
                'method': method,
                'n_evaluations': n_evaluations,
                'executed_evaluations': len(all_evaluations),
                'execution_time': execution_time,
                'evaluations': all_evaluations,
                'parameter_space': param_info,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to history
            self.exploration_history.append(results)
            
            # Log results
            self.logger.info(f"Parameter exploration completed in {execution_time:.2f} seconds")
            self.logger.info(f"Best parameters: {best_params}")
            self.logger.info(f"Best objective value: {best_value}")
            
            # Save results to file
            self._save_results(results)
            
            return results
        
        else:
            self.logger.error("Parameter exploration failed")
            return None
    
    def _grid_search(self, space, param_info, objective_function, n_evaluations, verbose, callback):
        """
        Exhaustively search the parameter space on a grid.
        
        Args:
            space (list): Parameter space
            param_info (dict): Parameter information
            objective_function (callable): Function to minimize
            n_evaluations (int): Maximum number of evaluations
            verbose (bool): Whether to log progress
            callback (callable, optional): Callback after each evaluation
            
        Returns:
            dict: Grid search results
        """
        self.logger.info("Performing grid search")
        
        # Create grid for each parameter
        param_grids = []
        param_names = []
        
        for param in space:
            name = param.name
            param_names.append(name)
            
            if isinstance(param, Real):
                # Determine number of points based on available evaluations
                # For continuous parameters, we need to discretize
                n_points = max(2, min(10, int(np.power(n_evaluations, 1/len(space)))))
                bounds = param_info[name]['bounds']
                if param_info[name].get('transform') == 'log':
                    # Logarithmic spacing
                    grid = np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), n_points)
                else:
                    # Linear spacing
                    grid = np.linspace(bounds[0], bounds[1], n_points)
                param_grids.append(grid)
                
            elif isinstance(param, Integer):
                # For integers, we can enumerate all values if the range is small
                bounds = param_info[name]['bounds']
                if bounds[1] - bounds[0] + 1 <= int(np.power(n_evaluations, 1/len(space))):
                    grid = np.arange(bounds[0], bounds[1] + 1)
                else:
                    # Otherwise, sample evenly
                    n_points = max(2, min(10, int(np.power(n_evaluations, 1/len(space)))))
                    grid = np.linspace(bounds[0], bounds[1], n_points, dtype=int)
                param_grids.append(grid)
                
            elif isinstance(param, Categorical):
                # For categorical, use all categories
                grid = param_info[name]['categories']
                param_grids.append(grid)
        
        # Create the Cartesian product of all parameter grids
        grid_combinations = list(itertools.product(*param_grids))
        
        # Limit the number of evaluations if needed
        if len(grid_combinations) > n_evaluations:
            self.logger.warning(f"Grid has {len(grid_combinations)} points, limiting to {n_evaluations}")
            grid_combinations = random.sample(grid_combinations, n_evaluations)
        
        # Evaluate objective function for each grid point
        evaluations = []
        best_params = None
        best_value = float('inf')
        
        for i, params_tuple in enumerate(grid_combinations):
            params = dict(zip(param_names, params_tuple))
            
            try:
                value = objective_function(params)
                
                evaluation = {
                    'params': params,
                    'value': value
                }
                
                evaluations.append(evaluation)
                
                # Track best result
                if value < best_value:
                    best_value = value
                    best_params = params
                    
                    if verbose:
                        self.logger.info(f"New best: {value} with params {params}")
                
                # Call callback if provided
                if callback is not None:
                    callback(evaluation)
                    
                if verbose and (i + 1) % 10 == 0:
                    self.logger.info(f"Completed {i + 1}/{len(grid_combinations)} evaluations")
                    
            except Exception as e:
                self.logger.error(f"Error evaluating parameters {params}: {str(e)}")
        
        # Return results
        return {
            'best_params': best_params,
            'best_value': best_value,
            'evaluations': evaluations
        }
    
    def _random_search(self, space, param_names, objective_function, n_evaluations, verbose, callback):
        """
        Randomly sample points from the parameter space.
        
        Args:
            space (list): Parameter space
            param_names (list): Parameter names
            objective_function (callable): Function to minimize
            n_evaluations (int): Number of evaluations
            verbose (bool): Whether to log progress
            callback (callable, optional): Callback after each evaluation
            
        Returns:
            dict: Random search results
        """
        self.logger.info(f"Performing random search with {n_evaluations} evaluations")
        
        # Initialize results
        evaluations = []
        best_params = None
        best_value = float('inf')
        
        # Sample and evaluate points
        for i in range(n_evaluations):
            # Generate random parameters
            params_list = []
            for param in space:
                if isinstance(param, Real):
                    if hasattr(param, 'prior') and param.prior == 'log-uniform':
                        # Log-uniform sampling
                        low, high = np.log10(param.low), np.log10(param.high)
                        value = 10 ** (random.uniform(low, high))
                    else:
                        # Uniform sampling
                        value = random.uniform(param.low, param.high)
                elif isinstance(param, Integer):
                    value = random.randint(param.low, param.high)
                elif isinstance(param, Categorical):
                    value = random.choice(param.categories)
                params_list.append(value)
            
            params = dict(zip(param_names, params_list))
            
            try:
                # Evaluate the objective function
                value = objective_function(params)
                
                evaluation = {
                    'params': params,
                    'value': value
                }
                
                evaluations.append(evaluation)
                
                # Track best result
                if value < best_value:
                    best_value = value
                    best_params = params
                    
                    if verbose:
                        self.logger.info(f"New best: {value} with params {params}")
                
                # Call callback if provided
                if callback is not None:
                    callback(evaluation)
                    
                if verbose and (i + 1) % 10 == 0:
                    self.logger.info(f"Completed {i + 1}/{n_evaluations} evaluations")
                    
            except Exception as e:
                self.logger.error(f"Error evaluating parameters {params}: {str(e)}")
        
        # Return results
        return {
            'best_params': best_params,
            'best_value': best_value,
            'evaluations': evaluations
        }
    
    def _bayesian_optimization(self, space, param_names, objective_function, n_evaluations, 
                             n_initial_points, verbose, callback):
        """
        Use Bayesian optimization to efficiently search the parameter space.
        
        Args:
            space (list): Parameter space
            param_names (list): Parameter names
            objective_function (callable): Function to minimize
            n_evaluations (int): Number of evaluations
            n_initial_points (int): Number of initial random points
            verbose (bool): Whether to log progress
            callback (callable, optional): Callback after each evaluation
            
        Returns:
            OptimizeResult: Optimization results
        """
        self.logger.info(f"Performing Bayesian optimization with {n_evaluations} evaluations")
        
        # Create a wrapper for the objective function that handles dictionaries
        @use_named_args(space)
        def objective_wrapper(**params):
            try:
                value = objective_function(params)
                
                # Call callback if provided
                if callback is not None:
                    callback({
                        'params': params,
                        'value': value
                    })
                
                return value
            except Exception as e:
                self.logger.error(f"Error evaluating parameters {params}: {str(e)}")
                return float('inf')  # Return a high value on error
        
        # Run Bayesian optimization
        return gp_minimize(
            objective_wrapper,
            space,
            n_calls=n_evaluations,
            n_initial_points=n_initial_points,
            random_state=self.random_state,
            verbose=verbose,
            n_jobs=self.parallel_jobs
        )
    
    def _forest_minimize(self, space, param_names, objective_function, n_evaluations, 
                       n_initial_points, verbose, callback):
        """
        Use random forest regression to optimize the parameter space.
        
        Args:
            space (list): Parameter space
            param_names (list): Parameter names
            objective_function (callable): Function to minimize
            n_evaluations (int): Number of evaluations
            n_initial_points (int): Number of initial random points
            verbose (bool): Whether to log progress
            callback (callable, optional): Callback after each evaluation
            
        Returns:
            OptimizeResult: Optimization results
        """
        self.logger.info(f"Performing random forest optimization with {n_evaluations} evaluations")
        
        # Create a wrapper for the objective function that handles dictionaries
        @use_named_args(space)
        def objective_wrapper(**params):
            try:
                value = objective_function(params)
                
                # Call callback if provided
                if callback is not None:
                    callback({
                        'params': params,
                        'value': value
                    })
                
                return value
            except Exception as e:
                self.logger.error(f"Error evaluating parameters {params}: {str(e)}")
                return float('inf')  # Return a high value on error
        
        # Run forest-based optimization
        return forest_minimize(
            objective_wrapper,
            space,
            n_calls=n_evaluations,
            n_initial_points=n_initial_points,
            random_state=self.random_state,
            verbose=verbose,
            n_jobs=self.parallel_jobs
        )
    
    def _evolutionary_optimization(self, space, param_info, objective_function, n_evaluations, 
                                verbose, callback):
        """
        Use evolutionary algorithms to optimize parameters.
        
        Args:
            space (list): Parameter space
            param_info (dict): Parameter information
            objective_function (callable): Function to minimize
            n_evaluations (int): Number of evaluations
            verbose (bool): Whether to log progress
            callback (callable, optional): Callback after each evaluation
            
        Returns:
            dict: Optimization results
        """
        self.logger.info(f"Performing evolutionary optimization with {n_evaluations} evaluations")
        
        # Extract parameter names
        param_names = list(param_info.keys())
        
        # Population size and number of generations
        population_size = min(50, n_evaluations // 2)
        n_generations = n_evaluations // population_size
        
        self.logger.info(f"Using population size {population_size} with {n_generations} generations")
        
        # Initialize population with random individuals
        population = []
        for _ in range(population_size):
            individual = {}
            for param_name, info in param_info.items():
                if info['type'] == 'real':
                    if info.get('transform') == 'log':
                        # Log-uniform sampling
                        low, high = np.log10(info['bounds'][0]), np.log10(info['bounds'][1])
                        individual[param_name] = 10 ** (random.uniform(low, high))
                    else:
                        # Uniform sampling
                        individual[param_name] = random.uniform(info['bounds'][0], info['bounds'][1])
                elif info['type'] == 'integer':
                    individual[param_name] = random.randint(info['bounds'][0], info['bounds'][1])
                elif info['type'] == 'categorical':
                    individual[param_name] = random.choice(info['categories'])
            population.append(individual)
        
        # Evaluate initial population
        fitness = []
        evaluations = []
        
        for individual in population:
            try:
                value = objective_function(individual)
                fitness.append(value)
                
                evaluation = {
                    'params': individual,
                    'value': value
                }
                
                evaluations.append(evaluation)
                
                # Call callback if provided
                if callback is not None:
                    callback(evaluation)
                    
            except Exception as e:
                self.logger.error(f"Error evaluating parameters {individual}: {str(e)}")
                fitness.append(float('inf'))
        
        # Track best solution
        best_idx = np.argmin(fitness)
        best_params = population[best_idx]
        best_value = fitness[best_idx]
        
        if verbose:
            self.logger.info(f"Initial best: {best_value} with params {best_params}")
        
        # Evolution loop
        for generation in range(n_generations - 1):  # -1 because we already evaluated the initial population
            # Select parents using tournament selection
            parents = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 3
                tournament_indices = random.sample(range(population_size), tournament_size)
                tournament_fitness = [fitness[idx] for idx in tournament_indices]
                winner_idx = tournament_indices[np.argmin(tournament_fitness)]
                parents.append(population[winner_idx])
            
            # Create offspring through crossover and mutation
            offspring = []
            while len(offspring) < population_size:
                # Select two parents
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                
                # Crossover
                if random.random() < 0.7:  # 70% chance of crossover
                    child = {}
                    for param_name in param_names:
                        # Randomly select from either parent
                        if random.random() < 0.5:
                            child[param_name] = parent1[param_name]
                        else:
                            child[param_name] = parent2[param_name]
                else:
                    # No crossover, just clone parent1
                    child = parent1.copy()
                
                # Mutation
                if random.random() < 0.3:  # 30% chance of mutation
                    # Select a random parameter to mutate
                    param_to_mutate = random.choice(param_names)
                    info = param_info[param_to_mutate]
                    
                    if info['type'] == 'real':
                        # Perturb the value by a random amount
                        if info.get('transform') == 'log':
                            # Log-scale perturbation
                            current = np.log10(child[param_to_mutate])
                            bounds = [np.log10(info['bounds'][0]), np.log10(info['bounds'][1])]
                            perturbation = random.gauss(0, 0.1 * (bounds[1] - bounds[0]))
                            new_value = 10 ** min(max(current + perturbation, bounds[0]), bounds[1])
                        else:
                            # Linear-scale perturbation
                            current = child[param_to_mutate]
                            bounds = info['bounds']
                            perturbation = random.gauss(0, 0.1 * (bounds[1] - bounds[0]))
                            new_value = min(max(current + perturbation, bounds[0]), bounds[1])
                        child[param_to_mutate] = new_value
                        
                    elif info['type'] == 'integer':
                        # For integers, add a small integer perturbation
                        current = child[param_to_mutate]
                        bounds = info['bounds']
                        perturbation = random.randint(-2, 2)
                        new_value = min(max(current + perturbation, bounds[0]), bounds[1])
                        child[param_to_mutate] = new_value
                        
                    elif info['type'] == 'categorical':
                        # For categorical, randomly select a different category
                        current = child[param_to_mutate]
                        categories = info['categories']
                        if len(categories) > 1:
                            other_categories = [c for c in categories if c != current]
                            child[param_to_mutate] = random.choice(other_categories)
                
                offspring.append(child)
            
            # Evaluate offspring
            offspring_fitness = []
            
            for individual in offspring:
                try:
                    value = objective_function(individual)
                    offspring_fitness.append(value)
                    
                    evaluation = {
                        'params': individual,
                        'value': value
                    }
                    
                    evaluations.append(evaluation)
                    
                    # Call callback if provided
                    if callback is not None:
                        callback(evaluation)
                        
                except Exception as e:
                    self.logger.error(f"Error evaluating parameters {individual}: {str(e)}")
                    offspring_fitness.append(float('inf'))
            
            # Replace population with offspring
            population = offspring
            fitness = offspring_fitness
            
            # Update best solution
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_value:
                best_params = population[best_idx]
                best_value = fitness[best_idx]
                
                if verbose:
                    self.logger.info(f"New best in generation {generation+1}: {best_value} with params {best_params}")
            
            if verbose:
                self.logger.info(f"Completed generation {generation+1}/{n_generations}")
        
        # Return results
        return {
            'best_params': best_params,
            'best_value': best_value,
            'evaluations': evaluations
        }
    
    def visualize_results(self, results, top_n=3, filename=None):
        """
        Visualize parameter exploration results.
        
        Args:
            results (dict): Results from explore method
            top_n (int): Number of top parameter values to highlight
            filename (str, optional): File to save visualization
            
        Returns:
            tuple: (fig, axes) for further customization
        """
        if not results:
            self.logger.error("No results to visualize")
            return None, None
        
        # Extract data
        evaluations = results['evaluations']
        best_params = results['best_params']
        param_space = results['parameter_space']
        
        # Convert evaluations to DataFrame
        eval_data = []
        
        for eval_dict in evaluations:
            row = {'objective_value': eval_dict['value']}
            row.update(eval_dict['params'])
            eval_data.append(row)
            
        df = pd.DataFrame(eval_data)
        
        # Determine number of parameters and create appropriate figure
        param_names = list(param_space.keys())
        n_params = len(param_names)
        
        # Setup visualization
        if n_params <= 1:
            # Single parameter case
            fig, ax = plt.subplots(figsize=(10, 6))
            param_name = param_names[0]
            
            # Sort by parameter value
            df_sorted = df.sort_values(param_name)
            
            # Plot objective values
            ax.plot(df_sorted[param_name], df_sorted['objective_value'], 'o-', alpha=0.5)
            
            # Highlight best points
            top_idx = df['objective_value'].argsort()[:top_n]
            ax.plot(df.iloc[top_idx][param_name], df.iloc[top_idx]['objective_value'], 'ro', markersize=10)
            
            # Annotate best point
            best_idx = df['objective_value'].argmin()
            ax.annotate(f"Best: {df.iloc[best_idx]['objective_value']:.4f}",
                      (df.iloc[best_idx][param_name], df.iloc[best_idx]['objective_value']),
                      xytext=(10, -20), textcoords='offset points',
                      arrowprops=dict(arrowstyle='->', lw=1.5))
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('Objective Value')
            ax.set_title('Parameter Exploration Results')
            ax.grid(True, alpha=0.3)
            
        elif n_params == 2:
            # Two parameters case - create a contour plot
            fig, ax = plt.subplots(figsize=(10, 8))
            param1, param2 = param_names
            
            # Check if we have enough data points for a contour plot
            if len(df) >= 10:
                # Create a grid for contour plot
                try:
                    from scipy.interpolate import griddata
                    
                    x = df[param1].values
                    y = df[param2].values
                    z = df['objective_value'].values
                    
                    # Create a grid
                    xi = np.linspace(min(x), max(x), 100)
                    yi = np.linspace(min(y), max(y), 100)
                    X, Y = np.meshgrid(xi, yi)
                    
                    # Interpolate
                    Z = griddata((x, y), z, (X, Y), method='cubic')
                    
                    # Create contour plot
                    contour = ax.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.7)
                    fig.colorbar(contour, ax=ax, label='Objective Value')
                except Exception as e:
                    self.logger.warning(f"Could not create contour plot: {str(e)}")
            
            # Plot all evaluation points
            scatter = ax.scatter(df[param1], df[param2], c=df['objective_value'], 
                              cmap='viridis', edgecolor='k', s=50)
            
            # Highlight best points
            top_idx = df['objective_value'].argsort()[:top_n]
            ax.scatter(df.iloc[top_idx][param1], df.iloc[top_idx][param2], c='r', s=100, edgecolor='k')
            
            # Annotate best point
            best_idx = df['objective_value'].argmin()
            ax.annotate(f"Best: {df.iloc[best_idx]['objective_value']:.4f}",
                      (df.iloc[best_idx][param1], df.iloc[best_idx][param2]),
                      xytext=(10, 10), textcoords='offset points',
                      arrowprops=dict(arrowstyle='->', lw=1.5))
            
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_title('Parameter Exploration Results')
            ax.grid(True, alpha=0.3)
            
        else:
            # More than two parameters - create a grid of pairwise plots
            n_cols = min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols
            figsize = (5 * n_cols, 4 * n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            
            # Flatten axes for easier iteration
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            
            # Create pairwise plots
            plot_idx = 0
            for i, param1 in enumerate(param_names):
                for j, param2 in enumerate(param_names[i+1:], i+1):
                    if plot_idx < len(axes):
                        ax = axes[plot_idx]
                        
                        # Plot evaluations
                        scatter = ax.scatter(df[param1], df[param2], c=df['objective_value'], 
                                          cmap='viridis', alpha=0.7, s=30)
                        
                        # Highlight best point
                        best_idx = df['objective_value'].argmin()
                        ax.scatter(df.iloc[best_idx][param1], df.iloc[best_idx][param2], 
                                 c='r', s=100, edgecolor='k')
                        
                        ax.set_xlabel(param1)
                        ax.set_ylabel(param2)
                        ax.grid(True, alpha=0.3)
                        
                        plot_idx += 1
            
            # Add colorbar
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
            fig.colorbar(scatter, cax=cbar_ax, label='Objective Value')
            
            # Set a title for the entire figure
            fig.suptitle('Parameter Exploration Results', fontsize=16)
            
            # Hide any unused subplots
            for ax_idx in range(plot_idx, len(axes)):
                axes[ax_idx].set_visible(False)
        
        # Add method and timing information
        if hasattr(fig, 'text'):
            fig.text(0.5, 0.01, 
                   f"Method: {results['method']}, Evaluations: {results['executed_evaluations']}, "
                   f"Time: {results['execution_time']:.2f}s",
                   ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure if filename provided
        if filename:
            save_path = os.path.join(self.results_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Visualization saved to {save_path}")
        
        return fig, axes
    
    def _save_results(self, results):
        """
        Save exploration results to a file.
        
        Args:
            results (dict): Exploration results
            
        Returns:
            str: Path to saved file
        """
        try:
            # Generate filename based on timestamp and method
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            method = results['method']
            filename = f"exploration_{method}_{timestamp}.json"
            
            # Save to JSON file
            filepath = os.path.join(self.results_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=lambda obj: str(obj) if isinstance(obj, (np.ndarray, np.number)) else obj)
            
            self.logger.info(f"Saved exploration results to {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            return None
    
    def load_results(self, filepath):
        """
        Load exploration results from a file.
        
        Args:
            filepath (str): Path to results file
            
        Returns:
            dict: Exploration results
        """
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            self.logger.info(f"Loaded exploration results from {filepath}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error loading results: {str(e)}")
            return None
    
    def optimize_indicator_parameters(self, indicator_function, price_data, target_metric, 
                                  param_definitions, method=None, n_evaluations=None):
        """
        Optimize parameters for a technical indicator.
        
        Args:
            indicator_function (callable): Function to compute the indicator
            price_data (pd.DataFrame): Historical price data
            target_metric (callable): Function to compute target metric from indicator values
            param_definitions (list): Parameter definitions
            method (str, optional): Optimization method
            n_evaluations (int, optional): Number of evaluations
            
        Returns:
            dict: Optimization results
        """
        self.logger.info(f"Optimizing indicator parameters")
        
        # Define the objective function
        def objective(params):
            try:
                # Compute indicator values with the given parameters
                indicator_values = indicator_function(price_data, **params)
                
                # Compute target metric
                metric_value = target_metric(price_data, indicator_values)
                
                return -metric_value  # Negative because we're minimizing
                
            except Exception as e:
                self.logger.error(f"Error in indicator objective function: {str(e)}")
                return float('inf')
        
        # Run parameter exploration
        results = self.explore(
            parameter_definitions=param_definitions,
            objective_function=objective,
            method=method,
            n_evaluations=n_evaluations,
            verbose=True
        )
        
        # Invert objective values for more intuitive display
        if results:
            results['best_value'] = -results['best_value']
            for eval_idx in range(len(results['evaluations'])):
                results['evaluations'][eval_idx]['value'] = -results['evaluations'][eval_idx]['value']
        
        return results
    
    def optimize_ml_hyperparameters(self, model_builder, train_func, eval_func, 
                               param_definitions, X_train, y_train, X_val, y_val,
                               method=None, n_evaluations=None):
        """
        Optimize hyperparameters for a machine learning model.
        
        Args:
            model_builder (callable): Function to build model from parameters
            train_func (callable): Function to train model
            eval_func (callable): Function to evaluate model
            param_definitions (list): Parameter definitions
            X_train, y_train: Training data
            X_val, y_val: Validation data
            method (str, optional): Optimization method
            n_evaluations (int, optional): Number of evaluations
            
        Returns:
            dict: Optimization results
        """
        self.logger.info(f"Optimizing ML hyperparameters")
        
        # Define the objective function
        def objective(params):
            try:
                # Build model with the given parameters
                model = model_builder(**params)
                
                # Train model
                trained_model = train_func(model, X_train, y_train)
                
                # Evaluate model
                metrics = eval_func(trained_model, X_val, y_val)
                
                # Return negative metric (assuming higher is better)
                return -metrics
                
            except Exception as e:
                self.logger.error(f"Error in ML hyperparameter objective function: {str(e)}")
                return float('inf')
        
        # Run parameter exploration
        results = self.explore(
            parameter_definitions=param_definitions,
            objective_function=objective,
            method=method,
            n_evaluations=n_evaluations,
            verbose=True
        )
        
        # Invert objective values for more intuitive display
        if results:
            results['best_value'] = -results['best_value']
            for eval_idx in range(len(results['evaluations'])):
                results['evaluations'][eval_idx]['value'] = -results['evaluations'][eval_idx]['value']
        
        return results
    
    def optimize_trading_parameters(self, strategy_function, backtest_function, price_data, 
                               param_definitions, method=None, n_evaluations=None):
        """
        Optimize parameters for a trading strategy.
        
        Args:
            strategy_function (callable): Function to build strategy from parameters
            backtest_function (callable): Function to backtest strategy
            price_data (pd.DataFrame): Historical price data
            param_definitions (list): Parameter definitions
            method (str, optional): Optimization method
            n_evaluations (int, optional): Number of evaluations
            
        Returns:
            dict: Optimization results
        """
        self.logger.info(f"Optimizing trading strategy parameters")
        
        # Define the objective function
        def objective(params):
            try:
                # Build strategy with the given parameters
                strategy = strategy_function(**params)
                
                # Backtest strategy
                results = backtest_function(strategy, price_data)
                
                # Extract metric to optimize (e.g., negative Sharpe ratio)
                # We use negative because the optimizer minimizes
                if 'sharpe_ratio' in results:
                    return -results['sharpe_ratio']
                elif 'returns' in results:
                    return -results['returns']
                else:
                    return -results.get('performance_metric', 0)
                
            except Exception as e:
                self.logger.error(f"Error in trading parameter objective function: {str(e)}")
                return float('inf')
        
        # Run parameter exploration
        results = self.explore(
            parameter_definitions=param_definitions,
            objective_function=objective,
            method=method,
            n_evaluations=n_evaluations,
            verbose=True
        )
        
        # Invert objective values for more intuitive display
        if results:
            results['best_value'] = -results['best_value']
            for eval_idx in range(len(results['evaluations'])):
                results['evaluations'][eval_idx]['value'] = -results['evaluations'][eval_idx]['value']
        
        return results
    
    def recommend_parameter_ranges(self, exploration_results, expansion_factor=0.5):
        """
        Recommend parameter ranges for further exploration based on previous results.
        
        Args:
            exploration_results (dict): Results from previous exploration
            expansion_factor (float): Factor to expand ranges by
            
        Returns:
            list: Recommended parameter definitions
        """
        if not exploration_results:
            self.logger.error("No exploration results provided")
            return None
        
        try:
            # Extract best parameters and parameter space
            best_params = exploration_results['best_params']
            param_space = exploration_results['parameter_space']
            
            # Create new parameter definitions
            new_param_defs = []
            
            for param_name, param_info in param_space.items():
                if param_info['type'] == 'real':
                    # For real parameters, create a range around the best value
                    best_value = best_params[param_name]
                    bounds = param_info['bounds']
                    
                    # Calculate new bounds
                    if param_info.get('transform') == 'log':
                        # For log scale, work in log space
                        best_log = np.log10(best_value)
                        min_log = np.log10(bounds[0])
                        max_log = np.log10(bounds[1])
                        
                        # Calculate new bounds in log space
                        range_log = max_log - min_log
                        new_min_log = max(min_log, best_log - expansion_factor * range_log / 2)
                        new_max_log = min(max_log, best_log + expansion_factor * range_log / 2)
                        
                        # Convert back to linear space
                        new_min = 10 ** new_min_log
                        new_max = 10 ** new_max_log
                    else:
                        # For linear scale, work directly
                        range_val = bounds[1] - bounds[0]
                        new_min = max(bounds[0], best_value - expansion_factor * range_val / 2)
                        new_max = min(bounds[1], best_value + expansion_factor * range_val / 2)
                    
                    # Create new parameter definition
                    new_param_defs.append({
                        'name': param_name,
                        'type': 'real',
                        'bounds': [new_min, new_max],
                        'transform': param_info.get('transform')
                    })
                    
                elif param_info['type'] == 'integer':
                    # For integer parameters, create a range around the best value
                    best_value = best_params[param_name]
                    bounds = param_info['bounds']
                    
                    # Calculate new bounds
                    range_val = bounds[1] - bounds[0]
                    new_min = max(bounds[0], int(best_value - expansion_factor * range_val / 2))
                    new_max = min(bounds[1], int(best_value + expansion_factor * range_val / 2))
                    
                    # Create new parameter definition
                    new_param_defs.append({
                        'name': param_name,
                        'type': 'integer',
                        'bounds': [new_min, new_max]
                    })
                    
                elif param_info['type'] == 'categorical':
                    # For categorical, keep the best value and a few alternatives
                    best_value = best_params[param_name]
                    categories = param_info['categories']
                    
                    if len(categories) > 1:
                        # Keep best value and some alternatives
                        other_categories = [c for c in categories if c != best_value]
                        n_others = min(len(other_categories), 2)  # Keep up to 2 alternatives
                        selected_others = random.sample(other_categories, n_others)
                        
                        new_categories = [best_value] + selected_others
                    else:
                        new_categories = categories
                    
                    # Create new parameter definition
                    new_param_defs.append({
                        'name': param_name,
                        'type': 'categorical',
                        'bounds': new_categories
                    })
            
            self.logger.info(f"Recommended {len(new_param_defs)} parameter ranges for further exploration")
            
            return new_param_defs
            
        except Exception as e:
            self.logger.error(f"Error recommending parameter ranges: {str(e)}")
            return None
