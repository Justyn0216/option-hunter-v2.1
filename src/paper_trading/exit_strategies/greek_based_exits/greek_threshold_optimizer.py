"""
Greek Threshold Optimizer Module

This module optimizes thresholds for Greek-based exit strategies
using historical trade data and performance metrics.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

class GreekThresholdOptimizer:
    """
    Optimizes thresholds for Greek-based exit strategies to maximize 
    performance metrics like profit factor, Sharpe ratio, or win rate.
    
    Uses historical trade data to find optimal exit thresholds for
    delta, gamma, theta, and vega strategies.
    """
    
    def __init__(self, config, drive_manager=None):
        """
        Initialize the Greek threshold optimizer.
        
        Args:
            config (dict): Configuration dictionary
            drive_manager: Optional GoogleDriveManager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("exit_strategies", {}).get("greek_based_exits", {}).get("greek_threshold_optimizer", {})
        self.drive_manager = drive_manager
        
        # Optimization parameters
        self.optimization_metric = self.config.get("optimization_metric", "profit_factor")  # profit_factor, sharpe, win_rate
        self.optimization_iterations = self.config.get("optimization_iterations", 100)
        self.optimization_window = self.config.get("optimization_window", 60)  # days
        self.min_trades_for_optimization = self.config.get("min_trades_for_optimization", 20)
        self.parallel_processing = self.config.get("parallel_processing", True)
        
        # Define parameter search spaces
        self.parameter_spaces = {
            "delta": {
                "call_high_delta_threshold": (0.7, 0.95, 0.05),  # (min, max, step)
                "call_low_delta_threshold": (0.05, 0.3, 0.05),
                "put_high_delta_threshold": (-0.95, -0.7, 0.05),
                "put_low_delta_threshold": (-0.3, -0.05, 0.05),
                "delta_change_threshold": (0.1, 0.5, 0.05)
            },
            "gamma": {
                "high_gamma_threshold": (0.03, 0.15, 0.01),
                "gamma_acceleration_threshold": (0.005, 0.03, 0.005),
                "gamma_change_threshold": (0.01, 0.1, 0.01)
            },
            "theta": {
                "high_theta_ratio_threshold": (0.01, 0.1, 0.01),
                "theta_acceleration_threshold": (0.001, 0.01, 0.001),
                "theta_percentage_threshold": (1.0, 7.0, 0.5)
            },
            "vega": {
                "iv_drop_threshold": (5.0, 30.0, 2.5),
                "iv_spike_threshold": (10.0, 40.0, 5.0),
                "vega_exposure_threshold": (0.2, 1.0, 0.1)
            }
        }
        
        # Current optimal thresholds
        self.optimal_thresholds = {}
        
        # Optimization history
        self.optimization_history = {
            "delta": [],
            "gamma": [],
            "theta": [],
            "vega": []
        }
        
        # Directory for storing optimization results
        self.results_dir = "data/exit_strategies/threshold_optimization"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load previous optimization results if available
        self._load_optimal_thresholds()
        
        self.logger.info(f"GreekThresholdOptimizer initialized with {self.optimization_metric} as optimization metric")
    
    def _load_optimal_thresholds(self):
        """Load previously optimized thresholds from disk or Google Drive."""
        try:
            # Try local file first
            thresholds_file = f"{self.results_dir}/optimal_thresholds.json"
            if os.path.exists(thresholds_file):
                with open(thresholds_file, "r") as f:
                    self.optimal_thresholds = json.load(f)
                self.logger.info(f"Loaded optimal thresholds for {len(self.optimal_thresholds)} Greek strategies")
                
                # Also load optimization history
                history_file = f"{self.results_dir}/optimization_history.json"
                if os.path.exists(history_file):
                    with open(history_file, "r") as f:
                        self.optimization_history = json.load(f)
            
            elif self.drive_manager and self.drive_manager.file_exists("exit_strategies/threshold_optimization/optimal_thresholds.json"):
                # Try Google Drive if local file not found
                content = self.drive_manager.download_file("exit_strategies/threshold_optimization/optimal_thresholds.json")
                self.optimal_thresholds = json.loads(content)
                
                # Save locally for future use
                with open(thresholds_file, "w") as f:
                    f.write(content)
                    
                self.logger.info(f"Loaded optimal thresholds from Google Drive for {len(self.optimal_thresholds)} Greek strategies")
                
                # Also load optimization history
                if self.drive_manager.file_exists("exit_strategies/threshold_optimization/optimization_history.json"):
                    history_content = self.drive_manager.download_file("exit_strategies/threshold_optimization/optimization_history.json")
                    self.optimization_history = json.loads(history_content)
                    
                    with open(f"{self.results_dir}/optimization_history.json", "w") as f:
                        f.write(history_content)
        
        except Exception as e:
            self.logger.error(f"Error loading optimal thresholds: {str(e)}")
    
    def _save_optimal_thresholds(self):
        """Save optimized thresholds to disk and Google Drive."""
        try:
            # Save to local file
            thresholds_file = f"{self.results_dir}/optimal_thresholds.json"
            with open(thresholds_file, "w") as f:
                json.dump(self.optimal_thresholds, f, indent=2)
            
            # Save optimization history
            history_file = f"{self.results_dir}/optimization_history.json"
            with open(history_file, "w") as f:
                json.dump(self.optimization_history, f, indent=2)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                self.drive_manager.upload_file(
                    "exit_strategies/threshold_optimization/optimal_thresholds.json",
                    json.dumps(self.optimal_thresholds, indent=2),
                    mime_type="application/json"
                )
                
                self.drive_manager.upload_file(
                    "exit_strategies/threshold_optimization/optimization_history.json",
                    json.dumps(self.optimization_history, indent=2),
                    mime_type="application/json"
                )
            
            self.logger.info(f"Saved optimal thresholds for {len(self.optimal_thresholds)} Greek strategies")
            
        except Exception as e:
            self.logger.error(f"Error saving optimal thresholds: {str(e)}")
    
    def optimize_thresholds(self, greek_type, trade_data):
        """
        Optimize thresholds for a specific Greek strategy.
        
        Args:
            greek_type (str): Greek strategy type ('delta', 'gamma', 'theta', 'vega')
            trade_data (list): List of historical trades with Greek data
            
        Returns:
            dict: Optimized thresholds
        """
        if greek_type not in self.parameter_spaces:
            self.logger.error(f"Unknown Greek type: {greek_type}")
            return None
        
        # Filter trades to recent ones within the optimization window
        cutoff_date = datetime.now() - timedelta(days=self.optimization_window)
        recent_trades = []
        
        for trade in trade_data:
            try:
                exit_time = trade.get("exit_time", None)
                if exit_time:
                    exit_dt = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")
                    if exit_dt >= cutoff_date:
                        recent_trades.append(trade)
            except (ValueError, TypeError):
                # Skip trades with invalid date format
                continue
        
        # Ensure we have enough trades for optimization
        if len(recent_trades) < self.min_trades_for_optimization:
            self.logger.warning(f"Not enough trades for {greek_type} optimization: {len(recent_trades)} < {self.min_trades_for_optimization}")
            return self.get_optimal_thresholds(greek_type)
        
        self.logger.info(f"Optimizing {greek_type} thresholds using {len(recent_trades)} trades")
        
        # Get parameter space for this Greek
        param_space = self.parameter_spaces[greek_type]
        
        # Generate parameter combinations for testing
        param_combinations = self._generate_parameter_combinations(param_space)
        
        # Initialize best results
        best_score = -float('inf')
        best_params = None
        all_results = []
        
        # Define the evaluation function
        def evaluate_params(params):
            # Simulate exits using these parameters
            performance = self._simulate_greek_exits(greek_type, recent_trades, params)
            
            # Get the optimization metric value
            if self.optimization_metric == "profit_factor":
                score = performance.get("profit_factor", 0)
            elif self.optimization_metric == "sharpe":
                score = performance.get("sharpe_ratio", 0)
            elif self.optimization_metric == "win_rate":
                score = performance.get("win_rate", 0)
            else:
                # Default to PnL
                score = performance.get("total_pnl", 0)
            
            return params, score, performance
        
        # Run evaluations (with or without parallelization)
        if self.parallel_processing and len(param_combinations) > 10:
            # Determine number of workers (use fewer cores to avoid overwhelming the system)
            num_workers = max(1, min(cpu_count() - 1, 4))
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(evaluate_params, params) for params in param_combinations]
                
                for future in as_completed(futures):
                    try:
                        params, score, performance = future.result()
                        all_results.append((params, score, performance))
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                            self.logger.debug(f"New best {greek_type} score: {best_score:.4f}")
                    except Exception as e:
                        self.logger.error(f"Error in parameter evaluation: {str(e)}")
        else:
            # Serial processing
            for params in param_combinations:
                try:
                    params, score, performance = evaluate_params(params)
                    all_results.append((params, score, performance))
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        self.logger.debug(f"New best {greek_type} score: {best_score:.4f}")
                except Exception as e:
                    self.logger.error(f"Error evaluating parameters {params}: {str(e)}")
        
        # Sort results by score
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Record optimization results
        optimization_result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_params": best_params,
            "best_score": best_score,
            "metric": self.optimization_metric,
            "num_trades": len(recent_trades),
            "top_results": [{"params": r[0], "score": r[1]} for r in all_results[:5]]
        }
        
        self.optimization_history[greek_type].append(optimization_result)
        
        # Limit history size
        if len(self.optimization_history[greek_type]) > 10:
            self.optimization_history[greek_type] = self.optimization_history[greek_type][-10:]
        
        # Update optimal thresholds
        self.optimal_thresholds[greek_type] = best_params
        
        # Save results
        self._save_optimal_thresholds()
        
        self.logger.info(f"Optimized {greek_type} thresholds: {best_params} with {self.optimization_metric}={best_score:.4f}")
        
        return best_params
    
    def _generate_parameter_combinations(self, param_space):
        """
        Generate parameter combinations for testing.
        
        Args:
            param_space (dict): Parameter search space
            
        Returns:
            list: List of parameter combinations
        """
        # Use random sampling for efficient search
        combinations = []
        
        # Generate random combinations
        for _ in range(self.optimization_iterations):
            params = {}
            for param_name, (min_val, max_val, step) in param_space.items():
                # Calculate number of steps
                num_steps = int((max_val - min_val) / step) + 1
                # Choose a random step
                step_idx = random.randint(0, num_steps - 1)
                # Calculate parameter value
                value = min_val + step_idx * step
                params[param_name] = value
            
            combinations.append(params)
        
        # Also include current optimal parameters if available
        if param_space.keys() == self.get_optimal_thresholds(next(iter(param_space))).keys():
            combinations.append(self.get_optimal_thresholds(next(iter(param_space))))
        
        return combinations
    
    def _simulate_greek_exits(self, greek_type, trades, parameters):
        """
        Simulate Greek-based exits with given parameters.
        
        Args:
            greek_type (str): Greek strategy type
            trades (list): Historical trades to simulate
            parameters (dict): Parameters to test
            
        Returns:
            dict: Performance metrics
        """
        # Initialize performance metrics
        total_pnl = 0.0
        win_count = 0
        loss_count = 0
        gross_wins = 0.0
        gross_losses = 0.0
        returns = []
        
        # Process each trade
        for trade in trades:
            # Skip if no Greek history available
            if not trade.get(f"{greek_type}_history"):
                continue
            
            # Get relevant history for the trade
            greek_history = trade.get(f"{greek_type}_history", [])
            
            # Find potential exit signals based on parameters
            exit_time, exit_pnl = self._find_exit_signals(greek_type, greek_history, parameters)
            
            # If found an exit, calculate performance
            if exit_pnl is not None:
                # Add to total P&L
                total_pnl += exit_pnl
                
                # Track win/loss stats
                if exit_pnl > 0:
                    win_count += 1
                    gross_wins += exit_pnl
                else:
                    loss_count += 1
                    gross_losses += abs(exit_pnl)
                
                # Add to returns list for Sharpe ratio
                entry_price = trade.get("entry_price", 0)
                if entry_price > 0:
                    returns.append(exit_pnl / entry_price)
            
        # Calculate performance metrics
        total_trades = win_count + loss_count
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Profit factor
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf') if gross_wins > 0 else 0
        
        # Sharpe ratio (simple version using daily returns)
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if len(returns) > 1 else 0.001
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        # Return combined metrics
        return {
            "total_pnl": total_pnl,
            "win_count": win_count,
            "loss_count": loss_count,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "avg_return": avg_return
        }
    
    def _find_exit_signals(self, greek_type, greek_history, parameters):
        """
        Find exit signals in trade history based on given parameters.
        
        Args:
            greek_type (str): Greek strategy type
            greek_history (list): History of Greek values for the trade
            parameters (dict): Parameters to test
            
        Returns:
            tuple: (exit_time, exit_pnl) or (None, None) if no exit
        """
        if not greek_history:
            return None, None
        
        # The implementation depends on the Greek type
        if greek_type == "delta":
            return self._find_delta_exit(greek_history, parameters)
        elif greek_type == "gamma":
            return self._find_gamma_exit(greek_history, parameters)
        elif greek_type == "theta":
            return self._find_theta_exit(greek_history, parameters)
        elif greek_type == "vega":
            return self._find_vega_exit(greek_history, parameters)
        else:
            return None, None
    
    def _find_delta_exit(self, delta_history, parameters):
        """
        Find delta-based exit in trade history.
        
        Args:
            delta_history (list): History of delta values for the trade
            parameters (dict): Parameters to test
            
        Returns:
            tuple: (exit_time, exit_pnl) or (None, None) if no exit
        """
        for i, entry in enumerate(delta_history):
            # Skip first entry (usually at entry)
            if i == 0:
                continue
                
            delta_value = entry.get("delta", 0)
            option_type = entry.get("option_type", "call")
            pnl = entry.get("pnl", 0)
            
            # Check for threshold crossings
            if option_type == "call":
                if delta_value >= parameters.get("call_high_delta_threshold", 0.85):
                    return entry.get("timestamp"), pnl
                elif delta_value <= parameters.get("call_low_delta_threshold", 0.15):
                    return entry.get("timestamp"), pnl
            else:  # put
                if delta_value <= parameters.get("put_high_delta_threshold", -0.85):
                    return entry.get("timestamp"), pnl
                elif delta_value >= parameters.get("put_low_delta_threshold", -0.15):
                    return entry.get("timestamp"), pnl
            
            # Check for delta changes
            if i >= 2:
                prev_delta = delta_history[i-1].get("delta", 0)
                delta_change = abs(delta_value - prev_delta)
                
                if delta_change >= parameters.get("delta_change_threshold", 0.2):
                    return entry.get("timestamp"), pnl
        
        # No exit signal found
        return None, None
    
    def _find_gamma_exit(self, gamma_history, parameters):
        """
        Find gamma-based exit in trade history.
        
        Args:
            gamma_history (list): History of gamma values for the trade
            parameters (dict): Parameters to test
            
        Returns:
            tuple: (exit_time, exit_pnl) or (None, None) if no exit
        """
        for i, entry in enumerate(gamma_history):
            # Skip first entry (usually at entry)
            if i == 0:
                continue
                
            gamma_value = entry.get("gamma", 0)
            pnl = entry.get("pnl", 0)
            
            # Check for high gamma
            if gamma_value >= parameters.get("high_gamma_threshold", 0.08):
                return entry.get("timestamp"), pnl
            
            # Check for gamma changes
            if i >= 2:
                prev_gamma = gamma_history[i-1].get("gamma", 0)
                gamma_change = abs(gamma_value - prev_gamma)
                
                if gamma_change >= parameters.get("gamma_change_threshold", 0.03):
                    return entry.get("timestamp"), pnl
            
            # Check for gamma acceleration
            if i >= 3:
                gamma_3 = gamma_history[i-2].get("gamma", 0)
                gamma_2 = gamma_history[i-1].get("gamma", 0)
                gamma_1 = gamma_value
                
                # Calculate first differences
                diff_1 = gamma_2 - gamma_3
                diff_2 = gamma_1 - gamma_2
                
                # Calculate acceleration
                acceleration = abs(diff_2 - diff_1)
                
                if acceleration >= parameters.get("gamma_acceleration_threshold", 0.01):
                    return entry.get("timestamp"), pnl
        
        # No exit signal found
        return None, None
    
    def _find_theta_exit(self, theta_history, parameters):
        """
        Find theta-based exit in trade history.
        
        Args:
            theta_history (list): History of theta values for the trade
            parameters (dict): Parameters to test
            
        Returns:
            tuple: (exit_time, exit_pnl) or (None, None) if no exit
        """
        for i, entry in enumerate(theta_history):
            # Skip first entry (usually at entry)
            if i == 0:
                continue
                
            theta_value = entry.get("theta", 0)  # Usually negative for long options
            option_price = entry.get("option_price", 0)
            pnl = entry.get("pnl", 0)
            
            # Check for high theta/price ratio
            if option_price > 0:
                theta_ratio = abs(theta_value) / option_price
                
                if theta_ratio >= parameters.get("high_theta_ratio_threshold", 0.04):
                    return entry.get("timestamp"), pnl
            
            # Check for high theta percentage
            theta_percent = entry.get("theta_percent", 0)
            if theta_percent >= parameters.get("theta_percentage_threshold", 3.0):
                return entry.get("timestamp"), pnl
            
            # Check for theta acceleration
            if i >= 3:
                # Need option prices for all entries to normalize
                price_3 = theta_history[i-2].get("option_price", 0)
                price_2 = theta_history[i-1].get("option_price", 0)
                price_1 = option_price
                
                if price_3 > 0 and price_2 > 0 and price_1 > 0:
                    # Get absolute theta values
                    theta_3 = abs(theta_history[i-2].get("theta", 0))
                    theta_2 = abs(theta_history[i-1].get("theta", 0))
                    theta_1 = abs(theta_value)
                    
                    # Normalize by option price
                    norm_theta_3 = theta_3 / price_3
                    norm_theta_2 = theta_2 / price_2
                    norm_theta_1 = theta_1 / price_1
                    
                    # Calculate first differences
                    diff_1 = norm_theta_2 - norm_theta_3
                    diff_2 = norm_theta_1 - norm_theta_2
                    
                    # Calculate acceleration
                    acceleration = diff_2 - diff_1
                    
                    if acceleration >= parameters.get("theta_acceleration_threshold", 0.005):
                        return entry.get("timestamp"), pnl
        
        # No exit signal found
        return None, None
    
    def _find_vega_exit(self, iv_history, parameters):
        """
        Find vega/IV-based exit in trade history.
        
        Args:
            iv_history (list): History of IV values for the trade
            parameters (dict): Parameters to test
            
        Returns:
            tuple: (exit_time, exit_pnl) or (None, None) if no exit
        """
        if not iv_history or len(iv_history) < 2:
            return None, None
            
        # Get initial IV
        initial_iv = iv_history[0].get("iv", 0)
        
        for i, entry in enumerate(iv_history):
            # Skip first entry (usually at entry)
            if i == 0:
                continue
                
            iv_value = entry.get("iv", 0)
            vega_value = entry.get("vega", 0)
            option_price = entry.get("option_price", 0)
            pnl = entry.get("pnl", 0)
            
            # Check for IV drops
            if initial_iv > 0:
                iv_change_pct = ((iv_value - initial_iv) / initial_iv) * 100
                
                if iv_change_pct <= -parameters.get("iv_drop_threshold", 15.0):
                    return entry.get("timestamp"), pnl
                
                # Check for IV spikes
                if iv_change_pct >= parameters.get("iv_spike_threshold", 25.0):
                    return entry.get("timestamp"), pnl
            
            # Check for vega exposure
            if option_price > 0:
                vega_ratio = vega_value / option_price
                
                if vega_ratio >= parameters.get("vega_exposure_threshold", 0.5):
                    return entry.get("timestamp"), pnl
        
        # No exit signal found
        return None, None
    
    def get_optimal_thresholds(self, greek_type):
        """
        Get optimal thresholds for a Greek strategy.
        
        Args:
            greek_type (str): Greek strategy type ('delta', 'gamma', 'theta', 'vega')
            
        Returns:
            dict: Optimal thresholds or default values if none available
        """
        # Return optimized values if available
        if greek_type in self.optimal_thresholds and self.optimal_thresholds[greek_type]:
            return self.optimal_thresholds[greek_type]
        
        # Otherwise return defaults based on parameter spaces
        default_params = {}
        if greek_type in self.parameter_spaces:
            for param_name, (min_val, max_val, _) in self.parameter_spaces[greek_type].items():
                # Use midpoint as default
                default_params[param_name] = (min_val + max_val) / 2
        
        return default_params
    
    def get_optimization_history(self, greek_type):
        """
        Get optimization history for a Greek strategy.
        
        Args:
            greek_type (str): Greek strategy type
            
        Returns:
            list: Optimization history entries
        """
        return self.optimization_history.get(greek_type, [])
    
    def run_full_optimization(self, trade_data):
        """
        Run optimization for all Greek strategies.
        
        Args:
            trade_data (dict): Dictionary mapping trade IDs to trade data
            
        Returns:
            dict: Optimized thresholds for all strategies
        """
        # Convert dictionary to list if needed
        if isinstance(trade_data, dict):
            trade_list = list(trade_data.values())
        else:
            trade_list = trade_data
        
        self.logger.info(f"Running full optimization with {len(trade_list)} trades")
        
        results = {}
        for greek_type in self.parameter_spaces.keys():
            self.logger.info(f"Optimizing {greek_type} thresholds...")
            thresholds = self.optimize_thresholds(greek_type, trade_list)
            results[greek_type] = thresholds
        
        self.logger.info("Full optimization complete")
        return results
