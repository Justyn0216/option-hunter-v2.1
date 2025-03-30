"""
Scenario Analyzer Module

This module provides tools for analyzing "what-if" scenarios in option trading.
It allows experimentation with different market conditions, volatility regimes,
and option strategies to understand potential outcomes.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy
from scipy.stats import norm

class ScenarioAnalyzer:
    """
    Analyzes "what-if" scenarios for option trading strategies.
    
    Features:
    - Monte Carlo simulation of price paths
    - Stress testing of strategies under extreme conditions
    - Sensitivity analysis for option Greeks
    - Strategy comparison under different market regimes
    """
    
    def __init__(self, config, option_pricer, backtest_engine=None):
        """
        Initialize the ScenarioAnalyzer.
        
        Args:
            config (dict): Configuration dictionary
            option_pricer: OptionPricer instance for option pricing
            backtest_engine: BacktestEngine instance for running simulations (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.option_pricer = option_pricer
        self.backtest_engine = backtest_engine
        
        # Create necessary directories
        self.results_dir = "data/scenario_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger.info("ScenarioAnalyzer initialized")
    
    def generate_price_paths(self, S0, days, num_paths=1000, 
                           annual_return=0.05, annual_volatility=0.20,
                           jump_intensity=0.01, jump_mean=-0.03, jump_std=0.10):
        """
        Generate price paths using Geometric Brownian Motion with jumps.
        
        Args:
            S0 (float): Initial stock price
            days (int): Number of days to simulate
            num_paths (int): Number of price paths to generate
            annual_return (float): Expected annual return
            annual_volatility (float): Expected annual volatility
            jump_intensity (float): Jump intensity (lambda)
            jump_mean (float): Jump size mean
            jump_std (float): Jump size standard deviation
            
        Returns:
            numpy.ndarray: Array of price paths, shape (num_paths, days+1)
        """
        self.logger.info(f"Generating {num_paths} price paths for {days} days")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Parameters for GBM
        dt = 1/252  # Daily time step
        mu = annual_return
        sigma = annual_volatility
        
        # Initialize price paths
        paths = np.zeros((num_paths, days+1))
        paths[:, 0] = S0
        
        # Generate paths
        for i in range(num_paths):
            for t in range(1, days+1):
                # Random normal for GBM
                z = np.random.normal(0, 1)
                
                # Random jump (Poisson process)
                jump_occurs = np.random.poisson(jump_intensity * dt) > 0
                jump_size = np.random.normal(jump_mean, jump_std) if jump_occurs else 0
                
                # Calculate price step
                paths[i, t] = paths[i, t-1] * np.exp(
                    (mu - 0.5 * sigma**2) * dt + 
                    sigma * np.sqrt(dt) * z +
                    jump_size
                )
        
        return paths
    
    def simulate_market_scenario(self, scenario_name, stock_price, volatility,
                               days_forward=30, num_simulations=1000,
                               market_drift=0.05, vol_of_vol=0.3):
        """
        Simulate a market scenario with stock price and volatility paths.
        
        Args:
            scenario_name (str): Name of the scenario
            stock_price (float): Initial stock price
            volatility (float): Initial volatility
            days_forward (int): Number of days to simulate
            num_simulations (int): Number of simulations to run
            market_drift (float): Annual market drift (return)
            vol_of_vol (float): Volatility of volatility
            
        Returns:
            dict: Scenario simulation results
        """
        self.logger.info(f"Simulating {scenario_name} scenario with {num_simulations} simulations")
        
        # Generate price paths
        price_paths = self.generate_price_paths(
            stock_price, days_forward, num_simulations, annual_return=market_drift
        )
        
        # Generate volatility paths using mean-reverting process
        # (Ornstein-Uhlenbeck process for volatility)
        vol_paths = np.zeros((num_simulations, days_forward+1))
        vol_paths[:, 0] = volatility
        
        # Volatility mean-reversion parameters
        vol_mean = volatility
        vol_mean_reversion = 2.0  # Speed of mean reversion
        
        for i in range(num_simulations):
            for t in range(1, days_forward+1):
                # Random normal for volatility
                z = np.random.normal(0, 1)
                
                # Calculate volatility step (mean-reverting)
                vol_paths[i, t] = vol_paths[i, t-1] + vol_mean_reversion * (vol_mean - vol_paths[i, t-1]) * (1/252) + \
                                 vol_of_vol * vol_paths[i, t-1] * np.sqrt(1/252) * z
                
                # Ensure volatility remains positive
                vol_paths[i, t] = max(0.01, vol_paths[i, t])
        
        # Calculate option prices across paths
        # We'll use ATM calls and puts with 30-day expiration
        call_prices = np.zeros((num_simulations, days_forward+1))
        put_prices = np.zeros((num_simulations, days_forward+1))
        
        for i in range(num_simulations):
            for t in range(days_forward+1):
                S = price_paths[i, t]
                K = S  # ATM
                T = 30/365  # 30-day option
                r = 0.02  # Risk-free rate
                sigma = vol_paths[i, t]
                
                call_prices[i, t] = self.option_pricer.calculate_option_price(S, K, T, r, sigma, 'call')
                put_prices[i, t] = self.option_pricer.calculate_option_price(S, K, T, r, sigma, 'put')
        
        # Calculate summary statistics
        final_prices = price_paths[:, -1]
        price_change_pct = (final_prices / stock_price) - 1
        
        final_vols = vol_paths[:, -1]
        vol_change_pct = (final_vols / volatility) - 1
        
        final_call_prices = call_prices[:, -1]
        call_price_change_pct = (final_call_prices / call_prices[:, 0]) - 1
        
        final_put_prices = put_prices[:, -1]
        put_price_change_pct = (final_put_prices / put_prices[:, 0]) - 1
        
        # Create results dictionary
        results = {
            'scenario_name': scenario_name,
            'initial_stock_price': stock_price,
            'initial_volatility': volatility,
            'days_forward': days_forward,
            'num_simulations': num_simulations,
            'market_drift': market_drift,
            'vol_of_vol': vol_of_vol,
            'simulation_date': datetime.now().strftime('%Y-%m-%d'),
            'price_paths': {
                'mean': np.mean(price_paths, axis=0).tolist(),
                'std': np.std(price_paths, axis=0).tolist(),
                'min': np.min(price_paths, axis=0).tolist(),
                'max': np.max(price_paths, axis=0).tolist(),
                'percentile_5': np.percentile(price_paths, 5, axis=0).tolist(),
                'percentile_95': np.percentile(price_paths, 95, axis=0).tolist()
            },
            'volatility_paths': {
                'mean': np.mean(vol_paths, axis=0).tolist(),
                'std': np.std(vol_paths, axis=0).tolist(),
                'min': np.min(vol_paths, axis=0).tolist(),
                'max': np.max(vol_paths, axis=0).tolist(),
                'percentile_5': np.percentile(vol_paths, 5, axis=0).tolist(),
                'percentile_95': np.percentile(vol_paths, 95, axis=0).tolist()
            },
            'call_prices': {
                'mean': np.mean(call_prices, axis=0).tolist(),
                'std': np.std(call_prices, axis=0).tolist(),
                'min': np.min(call_prices, axis=0).tolist(),
                'max': np.max(call_prices, axis=0).tolist(),
                'percentile_5': np.percentile(call_prices, 5, axis=0).tolist(),
                'percentile_95': np.percentile(call_prices, 95, axis=0).tolist()
            },
            'put_prices': {
                'mean': np.mean(put_prices, axis=0).tolist(),
                'std': np.std(put_prices, axis=0).tolist(),
                'min': np.min(put_prices, axis=0).tolist(),
                'max': np.max(put_prices, axis=0).tolist(),
                'percentile_5': np.percentile(put_prices, 5, axis=0).tolist(),
                'percentile_95': np.percentile(put_prices, 95, axis=0).tolist()
            },
            'summary': {
                'final_price_mean': np.mean(final_prices),
                'final_price_std': np.std(final_prices),
                'price_change_mean': np.mean(price_change_pct),
                'price_change_std': np.std(price_change_pct),
                'final_vol_mean': np.mean(final_vols),
                'final_vol_std': np.std(final_vols),
                'vol_change_mean': np.mean(vol_change_pct),
                'vol_change_std': np.std(vol_change_pct),
                'call_price_change_mean': np.mean(call_price_change_pct),
                'call_price_change_std': np.std(call_price_change_pct),
                'put_price_change_mean': np.mean(put_price_change_pct),
                'put_price_change_std': np.std(put_price_change_pct)
            }
        }
        
        # Save results
        self._save_scenario_results(results)
        
        return results
    
    def create_stress_test_scenarios(self, base_stock_price=100.0, base_volatility=0.20):
        """
        Create a set of stress test scenarios.
        
        Args:
            base_stock_price (float): Base stock price
            base_volatility (float): Base volatility
            
        Returns:
            dict: Dictionary of scenario results
        """
        scenarios = {}
        
        # Normal market scenario (baseline)
        scenarios['baseline'] = self.simulate_market_scenario(
            'Baseline Market',
            base_stock_price,
            base_volatility,
            market_drift=0.05,
            vol_of_vol=0.2
        )
        
        # Bull market scenario
        scenarios['bull_market'] = self.simulate_market_scenario(
            'Bull Market',
            base_stock_price,
            base_volatility * 0.8,  # Lower volatility
            market_drift=0.15,  # Higher returns
            vol_of_vol=0.15
        )
        
        # Bear market scenario
        scenarios['bear_market'] = self.simulate_market_scenario(
            'Bear Market',
            base_stock_price,
            base_volatility * 1.5,  # Higher volatility
            market_drift=-0.20,  # Negative returns
            vol_of_vol=0.3
        )
        
        # High volatility scenario
        scenarios['high_volatility'] = self.simulate_market_scenario(
            'High Volatility Market',
            base_stock_price,
            base_volatility * 2.0,  # Much higher volatility
            market_drift=0.0,  # Flat market
            vol_of_vol=0.5
        )
        
        # Market crash scenario
        scenarios['market_crash'] = self.simulate_market_scenario(
            'Market Crash',
            base_stock_price,
            base_volatility * 3.0,  # Extreme volatility
            market_drift=-0.4,  # Severe negative returns
            vol_of_vol=0.7
        )
        
        return scenarios
    
    def analyze_option_strategy(self, strategy_func, scenarios):
        """
        Analyze an option strategy across different market scenarios.
        
        Args:
            strategy_func (callable): Strategy function to analyze
            scenarios (dict): Dictionary of scenario results
            
        Returns:
            dict: Strategy performance across scenarios
        """
        if not self.backtest_engine:
            self.logger.error("No backtest engine available for strategy analysis")
            return None
        
        self.logger.info(f"Analyzing option strategy across {len(scenarios)} scenarios")
        
        strategy_results = {}
        
        for scenario_name, scenario_data in scenarios.items():
            self.logger.info(f"Testing strategy in {scenario_name} scenario")
            
            # Set up a custom backtest with this scenario
            self.backtest_engine.reset()
            
            # Create synthetic data for this scenario
            # In a real implementation, we'd create more detailed synthetic data
            # based on the scenario paths
            days = scenario_data['days_forward']
            price_path = scenario_data['price_paths']['mean']
            vol_path = scenario_data['volatility_paths']['mean']
            
            dates = [datetime.now() + timedelta(days=i) for i in range(days+1)]
            
            # Create synthetic stock data for SPY
            spy_data = pd.DataFrame({
                'open': price_path,
                'high': [p * 1.01 for p in price_path],
                'low': [p * 0.99 for p in price_path],
                'close': price_path,
                'volume': [1000000] * len(price_path)
            }, index=dates)
            
            # Run a backtest with this synthetic data
            # In a real implementation, we'd need to more carefully construct
            # option chains and other data needed for the backtest
            
            # For now, just print the summary of what we would do
            print(f"\nScenario: {scenario_name}")
            print(f"Final price: ${price_path[-1]:.2f} from ${price_path[0]:.2f} ({(price_path[-1]/price_path[0]-1)*100:.1f}%)")
            print(f"Final volatility: {vol_path[-1]:.2f} from {vol_path[0]:.2f} ({(vol_path[-1]/vol_path[0]-1)*100:.1f}%)")
            
            # In a full implementation, we would run:
            # results = self.backtest_engine.run_backtest(strategy=strategy_func)
            # strategy_results[scenario_name] = results
            
            # For now, create a placeholder result
            strategy_results[scenario_name] = {
                'scenario_name': scenario_name,
                'initial_price': price_path[0],
                'final_price': price_path[-1],
                'price_change_pct': (price_path[-1]/price_path[0]) - 1,
                'initial_volatility': vol_path[0],
                'final_volatility': vol_path[-1],
                'volatility_change_pct': (vol_path[-1]/vol_path[0]) - 1,
                'strategy_return': None,  # Would come from backtest
                'strategy_sharpe': None,  # Would come from backtest
                'strategy_max_drawdown': None  # Would come from backtest
            }
        
        return strategy_results
    
    def compare_strategies(self, strategies, scenarios):
        """
        Compare multiple strategies across different market scenarios.
        
        Args:
            strategies (dict): Dictionary of strategy functions
            scenarios (dict): Dictionary of scenario results
            
        Returns:
            dict: Comparison results for all strategies across scenarios
        """
        comparison_results = {}
        
        for strategy_name, strategy_func in strategies.items():
            self.logger.info(f"Analyzing strategy: {strategy_name}")
            results = self.analyze_option_strategy(strategy_func, scenarios)
            comparison_results[strategy_name] = results
        
        # Generate comparison report (in a real implementation)
        self.logger.info(f"Completed comparison of {len(strategies)} strategies across {len(scenarios)} scenarios")
        
        return comparison_results
    
    def run_greek_sensitivity_analysis(self, option_type='call', strike_pct=1.0, days_to_expiration=30,
                                     price_range_pct=0.2, vol_range_pct=0.5, steps=20):
        """
        Run sensitivity analysis for option Greeks.
        
        Args:
            option_type (str): Option type ('call' or 'put')
            strike_pct (float): Strike price as percentage of current price
            days_to_expiration (int): Days to expiration
            price_range_pct (float): Price range percentage to analyze
            vol_range_pct (float): Volatility range percentage to analyze
            steps (int): Number of steps in each dimension
            
        Returns:
            dict: Sensitivity analysis results
        """
        self.logger.info(f"Running Greek sensitivity analysis for {option_type} options")
        
        # Base parameters
        S0 = 100.0  # Base stock price
        K = S0 * strike_pct  # Strike price
        T = days_to_expiration / 365  # Time to expiration in years
        r = 0.02  # Risk-free rate
        sigma0 = 0.2  # Base volatility
        
        # Create price and volatility ranges
        price_min = S0 * (1 - price_range_pct)
        price_max = S0 * (1 + price_range_pct)
        prices = np.linspace(price_min, price_max, steps)
        
        vol_min = sigma0 * (1 - vol_range_pct)
        vol_max = sigma0 * (1 + vol_range_pct)
        vols = np.linspace(vol_min, vol_max, steps)
        
        # Initialize result arrays
        option_prices = np.zeros((steps, steps))
        delta_values = np.zeros((steps, steps))
        gamma_values = np.zeros((steps, steps))
        theta_values = np.zeros((steps, steps))
        vega_values = np.zeros((steps, steps))
        
        # Calculate values
        for i, S in enumerate(prices):
            for j, sigma in enumerate(vols):
                # Calculate option price
                option_prices[i, j] = self.option_pricer.calculate_option_price(S, K, T, r, sigma, option_type)
                
                # Calculate Greeks
                greeks = self.option_pricer.calculate_greeks(S, K, T, r, sigma, option_type)
                delta_values[i, j] = greeks['delta']
                gamma_values[i, j] = greeks['gamma']
                theta_values[i, j] = greeks['theta']
                vega_values[i, j] = greeks['vega']
        
        # Create results dictionary
        results = {
            'analysis_type': 'greek_sensitivity',
            'option_type': option_type,
            'strike_pct': strike_pct,
            'days_to_expiration': days_to_expiration,
            'base_price': S0,
            'strike_price': K,
            'base_volatility': sigma0,
            'risk_free_rate': r,
            'price_range': [price_min, price_max],
            'vol_range': [vol_min, vol_max],
            'prices': prices.tolist(),
            'vols': vols.tolist(),
            'option_prices': option_prices.tolist(),
            'delta_values': delta_values.tolist(),
            'gamma_values': gamma_values.tolist(),
            'theta_values': theta_values.tolist(),
            'vega_values': vega_values.tolist()
        }
        
        # Save results
        self._save_sensitivity_results(results)
        
        # Generate some basic visualizations
        self._plot_greek_sensitivity(results)
        
        return results
    
    def _plot_greek_sensitivity(self, results):
        """
        Plot Greek sensitivity analysis results.
        
        Args:
            results (dict): Sensitivity analysis results
            
        Returns:
            None
        """
        try:
            # Extract data
            prices = results['prices']
            vols = results['vols']
            option_prices = np.array(results['option_prices'])
            delta_values = np.array(results['delta_values'])
            gamma_values = np.array(results['gamma_values'])
            theta_values = np.array(results['theta_values'])
            vega_values = np.array(results['vega_values'])
            
            # Create meshgrid for 3D plotting
            X, Y = np.meshgrid(prices, vols)
            
            # Set up multiple subplots
            fig = plt.figure(figsize=(15, 10))
            
            # Plot option prices
            ax1 = fig.add_subplot(221, projection='3d')
            surf1 = ax1.plot_surface(X, Y, option_prices.T, cmap='viridis', alpha=0.8)
            ax1.set_xlabel('Stock Price')
            ax1.set_ylabel('Volatility')
            ax1.set_zlabel('Option Price')
            ax1.set_title('Option Price Sensitivity')
            
            # Plot delta
            ax2 = fig.add_subplot(222, projection='3d')
            surf2 = ax2.plot_surface(X, Y, delta_values.T, cmap='coolwarm', alpha=0.8)
            ax2.set_xlabel('Stock Price')
            ax2.set_ylabel('Volatility')
            ax2.set_zlabel('Delta')
            ax2.set_title('Delta Sensitivity')
            
            # Plot gamma
            ax3 = fig.add_subplot(223, projection='3d')
            surf3 = ax3.plot_surface(X, Y, gamma_values.T, cmap='plasma', alpha=0.8)
            ax3.set_xlabel('Stock Price')
            ax3.set_ylabel('Volatility')
            ax3.set_zlabel('Gamma')
            ax3.set_title('Gamma Sensitivity')
            
            # Plot vega
            ax4 = fig.add_subplot(224, projection='3d')
            surf4 = ax4.plot_surface(X, Y, vega_values.T, cmap='inferno', alpha=0.8)
            ax4.set_xlabel('Stock Price')
            ax4.set_ylabel('Volatility')
            ax4.set_zlabel('Vega')
            ax4.set_title('Vega Sensitivity')
            
            plt.tight_layout()
            
            # Save figure
            plot_file = f"{self.results_dir}/greek_sensitivity_{results['option_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file)
            
            self.logger.info(f"Greek sensitivity plot saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error plotting Greek sensitivity: {str(e)}")
    
    def _save_scenario_results(self, results):
        """
        Save scenario results to file.
        
        Args:
            results (dict): Scenario results
            
        Returns:
            str: Path to saved file
        """
        try:
            # Save as JSON
            scenario_name = results['scenario_name'].replace(' ', '_').lower()
            result_file = f"{self.results_dir}/scenario_{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Scenario results saved to {result_file}")
            return result_file
            
        except Exception as e:
            self.logger.error(f"Error saving scenario results: {str(e)}")
            return None
    
    def _save_sensitivity_results(self, results):
        """
        Save sensitivity analysis results to file.
        
        Args:
            results (dict): Sensitivity analysis results
            
        Returns:
            str: Path to saved file
        """
        try:
            # Save as JSON
            result_file = f"{self.results_dir}/sensitivity_{results['option_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Sensitivity results saved to {result_file}")
            return result_file
            
        except Exception as e:
            self.logger.error(f"Error saving sensitivity results: {str(e)}")
            return None
            
    def analyze_strategy_robustness(self, strategy_func, base_params, param_ranges, num_samples=100):
        """
        Analyze a strategy's robustness to parameter changes.
        
        Args:
            strategy_func (callable): Strategy function to analyze
            base_params (dict): Base parameters for the strategy
            param_ranges (dict): Dictionary mapping parameter names to (min, max) ranges
            num_samples (int): Number of random samples to generate
            
        Returns:
            dict: Robustness analysis results
        """
        if not self.backtest_engine:
            self.logger.error("No backtest engine available for robustness analysis")
            return None
        
        self.logger.info(f"Analyzing strategy robustness with {num_samples} parameter combinations")
        
        # Generate random parameter combinations
        param_combinations = []
        
        for i in range(num_samples):
            params = base_params.copy()
            
            # Randomly select parameters within ranges
            for param, (min_val, max_val) in param_ranges.items():
                params[param] = min_val + np.random.random() * (max_val - min_val)
            
            param_combinations.append(params)
        
        # Run backtests with each parameter combination
        results = []
        
        for i, params in enumerate(param_combinations):
            self.logger.debug(f"Running robustness test {i+1}/{num_samples}")
            
            # In a real implementation, we would:
            # 1. Configure the strategy with these parameters
            # 2. Run the backtest
            # 3. Collect the results
            
            # For now, create placeholder results
            result = {
                'params': params,
                'return': np.random.normal(0.1, 0.2),  # Simulated return
                'sharpe': np.random.normal(1.0, 0.5),  # Simulated Sharpe
                'max_drawdown': np.random.uniform(0.05, 0.3),  # Simulated max drawdown
                'win_rate': np.random.uniform(0.4, 0.7)  # Simulated win rate
            }
            
            results.append(result)
        
        # Analyze parameter sensitivity
        param_names = list(param_ranges.keys())
        param_values = {param: [result['params'][param] for result in results] for param in param_names}
        
        # Extract performance metrics
        returns = [result['return'] for result in results]
        sharpes = [result['sharpe'] for result in results]
        drawdowns = [result['max_drawdown'] for result in results]
        win_rates = [result['win_rate'] for result in results]
        
        # Calculate correlations between parameters and performance
        correlations = {}
        
        for param in param_names:
            param_values_array = np.array(param_values[param])
            
            correlations[param] = {
                'return': np.corrcoef(param_values_array, returns)[0, 1],
                'sharpe': np.corrcoef(param_values_array, sharpes)[0, 1],
                'max_drawdown': np.corrcoef(param_values_array, drawdowns)[0, 1],
                'win_rate': np.corrcoef(param_values_array, win_rates)[0, 1]
            }
        
        # Create final results
        robustness_results = {
            'analysis_type': 'strategy_robustness',
            'num_samples': num_samples,
            'base_params': base_params,
            'param_ranges': param_ranges,
            'results': results,
            'correlations': correlations,
            'summary': {
                'return': {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'min': np.min(returns),
                    'max': np.max(returns)
                },
                'sharpe': {
                    'mean': np.mean(sharpes),
                    'std': np.std(sharpes),
                    'min': np.min(sharpes),
                    'max': np.max(sharpes)
                },
                'max_drawdown': {
                    'mean': np.mean(drawdowns),
                    'std': np.std(drawdowns),
                    'min': np.min(drawdowns),
                    'max': np.max(drawdowns)
                },
                'win_rate': {
                    'mean': np.mean(win_rates),
                    'std': np.std(win_rates),
                    'min': np.min(win_rates),
                    'max': np.max(win_rates)
                }
            }
        }
        
        # Save results
        result_file = f"{self.results_dir}/robustness_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(result_file, 'w') as f:
            json.dump(robustness_results, f, indent=2)
        
        self.logger.info(f"Robustness analysis saved to {result_file}")
        
        return robustness_results
