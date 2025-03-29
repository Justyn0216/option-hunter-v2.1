"""
Model Performance Tracker Module

This module tracks the prediction accuracy and trade profitability of
each model in the system to support adaptive model selection.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class ModelPerformanceTracker:
    """
    Tracks and evaluates the performance of prediction models across
    different market conditions and time periods.
    """
    
    def __init__(self, config, drive_manager=None):
        """
        Initialize the ModelPerformanceTracker.
        
        Args:
            config (dict): Configuration settings
            drive_manager: Optional GoogleDriveManager for data storage
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.drive_manager = drive_manager
        
        # Extract configuration
        self.tracker_config = config.get("meta_learning", {}).get("model_performance_tracker", {})
        
        # Performance history
        self.performance_history = []
        
        # Performance metrics by model
        self.model_metrics = {}
        
        # Last calculation time
        self.last_calculation = None
        
        # Create directories
        os.makedirs("data/meta_learning", exist_ok=True)
        
        # Load performance history
        self._load_performance_history()
        
        self.logger.info("ModelPerformanceTracker initialized")
    
    def _load_performance_history(self):
        """Load performance history from storage."""
        history_file = "data/meta_learning/model_performance_history.json"
        metrics_file = "data/meta_learning/model_metrics.json"
        
        try:
            # Load performance history
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.performance_history = json.load(f)
                
                self.logger.info(f"Loaded {len(self.performance_history)} performance records from file")
                
            elif self.drive_manager and self.drive_manager.file_exists("model_performance_history.json"):
                # Download from Google Drive
                file_data = self.drive_manager.download_file("model_performance_history.json")
                
                # Parse JSON
                self.performance_history = json.loads(file_data)
                
                # Save locally
                with open(history_file, 'w') as f:
                    json.dump(self.performance_history, f, indent=2)
                
                self.logger.info(f"Loaded {len(self.performance_history)} performance records from Google Drive")
            
            # Load model metrics
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                self.model_metrics = metrics_data.get("metrics", {})
                
                if "last_calculation" in metrics_data:
                    self.last_calculation = datetime.strptime(
                        metrics_data["last_calculation"], "%Y-%m-%d %H:%M:%S"
                    )
                
                self.logger.info(f"Loaded metrics for {len(self.model_metrics)} models")
                
            elif self.drive_manager and self.drive_manager.file_exists("model_metrics.json"):
                # Download from Google Drive
                file_data = self.drive_manager.download_file("model_metrics.json")
                
                # Parse JSON
                metrics_data = json.loads(file_data)
                
                self.model_metrics = metrics_data.get("metrics", {})
                
                if "last_calculation" in metrics_data:
                    self.last_calculation = datetime.strptime(
                        metrics_data["last_calculation"], "%Y-%m-%d %H:%M:%S"
                    )
                
                # Save locally
                with open(metrics_file, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                
                self.logger.info(f"Loaded metrics for {len(self.model_metrics)} models from Google Drive")
                
        except Exception as e:
            self.logger.error(f"Error loading performance history: {str(e)}")
            # Start with empty history and metrics
            self.performance_history = []
            self.model_metrics = {}
    
    def _save_performance_history(self):
        """Save performance history to storage."""
        history_file = "data/meta_learning/model_performance_history.json"
        
        try:
            # Limit size of performance history if needed
            max_history = self.tracker_config.get("max_history_records", 10000)
            
            if len(self.performance_history) > max_history:
                # Keep most recent records
                self.performance_history = self.performance_history[-max_history:]
            
            # Save to file
            with open(history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                self.drive_manager.upload_file(
                    "model_performance_history.json",
                    json.dumps(self.performance_history),
                    mime_type="application/json"
                )
                
            self.logger.debug("Saved performance history")
            
        except Exception as e:
            self.logger.error(f"Error saving performance history: {str(e)}")
    
    def _save_model_metrics(self):
        """Save model metrics to storage."""
        metrics_file = "data/meta_learning/model_metrics.json"
        
        try:
            # Create metrics data
            metrics_data = {
                "metrics": self.model_metrics,
                "last_calculation": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to file
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                self.drive_manager.upload_file(
                    "model_metrics.json",
                    json.dumps(metrics_data),
                    mime_type="application/json"
                )
                
            self.logger.debug("Saved model metrics")
            
        except Exception as e:
            self.logger.error(f"Error saving model metrics: {str(e)}")
    
    def track_prediction(self, model_name, prediction_data, market_conditions):
        """
        Track a new prediction for later evaluation.
        
        Args:
            model_name (str): Name of the model
            prediction_data (dict): Prediction data
            market_conditions (dict): Market conditions during prediction
            
        Returns:
            str: Unique tracking ID for this prediction
        """
        try:
            # Generate unique tracking ID
            tracking_id = f"{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.performance_history)}"
            
            # Create tracking record
            tracking_record = {
                "tracking_id": tracking_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": model_name,
                "prediction": prediction_data,
                "market_conditions": market_conditions,
                "outcome": None,  # Will be filled later
                "evaluation": None  # Will be filled later
            }
            
            # Add to history
            self.performance_history.append(tracking_record)
            
            # Periodically save history
            if len(self.performance_history) % 100 == 0:
                self._save_performance_history()
            
            return tracking_id
            
        except Exception as e:
            self.logger.error(f"Error tracking prediction: {str(e)}")
            return None
    
    def record_outcome(self, tracking_id, outcome_data):
        """
        Record the actual outcome for a previously tracked prediction.
        
        Args:
            tracking_id (str): Tracking ID from track_prediction
            outcome_data (dict): Actual outcome data
            
        Returns:
            bool: True if outcome was recorded successfully
        """
        try:
            # Find tracking record
            for i, record in enumerate(self.performance_history):
                if record.get("tracking_id") == tracking_id:
                    # Update with outcome
                    self.performance_history[i]["outcome"] = outcome_data
                    
                    # Evaluate prediction vs. outcome
                    evaluation = self._evaluate_prediction(
                        self.performance_history[i]["prediction"],
                        outcome_data,
                        self.performance_history[i]["model_name"]
                    )
                    
                    # Store evaluation
                    self.performance_history[i]["evaluation"] = evaluation
                    
                    # Save history periodically
                    if i % 20 == 0:
                        self._save_performance_history()
                    
                    # Update metrics if needed
                    self._update_metrics_if_needed()
                    
                    return True
            
            self.logger.warning(f"Tracking ID not found: {tracking_id}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error recording outcome: {str(e)}")
            return False
    
    def _evaluate_prediction(self, prediction, outcome, model_name):
        """
        Evaluate a prediction against the actual outcome.
        
        Args:
            prediction (dict): Prediction data
            outcome (dict): Actual outcome data
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            # Default evaluation
            evaluation = {
                "accuracy": 0.0,
                "error": 0.0,
                "correct": False
            }
            
            # Different evaluation based on model type
            if "entry" in model_name.lower():
                # Entry models predict direction
                prediction_signal = prediction.get("signal", "neutral")
                actual_direction = outcome.get("direction", "neutral")
                
                # Check if prediction was correct
                if prediction_signal == actual_direction:
                    evaluation["correct"] = True
                    evaluation["accuracy"] = 1.0
                elif prediction_signal == "neutral" or actual_direction == "neutral":
                    # Neutral predictions are partly correct
                    evaluation["correct"] = False
                    evaluation["accuracy"] = 0.5
                else:
                    # Wrong direction
                    evaluation["correct"] = False
                    evaluation["accuracy"] = 0.0
                
                # Add P&L data if available
                if "pnl_percent" in outcome:
                    evaluation["pnl_percent"] = outcome["pnl_percent"]
                
            elif "exit" in model_name.lower():
                # Exit models predict whether to exit
                prediction_exit = prediction.get("should_exit", False)
                optimal_exit = outcome.get("optimal_exit", False)
                
                # Check if prediction was correct
                if prediction_exit == optimal_exit:
                    evaluation["correct"] = True
                    evaluation["accuracy"] = 1.0
                else:
                    evaluation["correct"] = False
                    evaluation["accuracy"] = 0.0
                
                # Calculate opportunity cost (if available)
                if "exit_price" in outcome and "optimal_price" in outcome:
                    # For exits, error is price difference relative to optimal
                    price_diff_pct = ((outcome["exit_price"] / outcome["optimal_price"]) - 1) * 100
                    
                    # For long positions, higher exit is better
                    # For short positions, lower exit is better
                    if outcome.get("position_type") == "short":
                        price_diff_pct = -price_diff_pct
                    
                    evaluation["price_difference_pct"] = price_diff_pct
                    
                    # Negative diff means missed opportunity, positive means better than optimal
                    evaluation["opportunity_cost"] = -price_diff_pct if price_diff_pct < 0 else 0
                
            elif "sizing" in model_name.lower():
                # Sizing models predict position size
                predicted_size = prediction.get("position_size", 0.0)
                optimal_size = outcome.get("optimal_size", 0.0)
                
                # Calculate error
                if optimal_size > 0:
                    # Error is percentage difference from optimal
                    size_error_pct = abs((predicted_size / optimal_size) - 1) * 100
                    
                    # Scale accuracy based on error (0% error = 1.0 accuracy, 100% error = 0.0 accuracy)
                    evaluation["error"] = size_error_pct
                    evaluation["accuracy"] = max(0.0, 1.0 - (size_error_pct / 100.0))
                    
                    # Consider "correct" if within 20% of optimal
                    evaluation["correct"] = size_error_pct <= 20.0
                else:
                    # If optimal size is 0, check if prediction was also near 0
                    evaluation["error"] = predicted_size
                    evaluation["accuracy"] = 1.0 if predicted_size < 100 else 0.0
                    evaluation["correct"] = predicted_size < 100
                
                # Add P&L data if available
                if "pnl_percent" in outcome:
                    evaluation["pnl_percent"] = outcome["pnl_percent"]
                    
                    # Calculate risk-adjusted return
                    if predicted_size > 0:
                        evaluation["risk_adjusted_return"] = outcome["pnl_percent"] / predicted_size
                    else:
                        evaluation["risk_adjusted_return"] = 0.0
            
            else:
                # Generic evaluation for other model types
                if "value" in prediction and "actual_value" in outcome:
                    predicted = prediction["value"]
                    actual = outcome["actual_value"]
                    
                    # Calculate error based on data type
                    if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
                        # Numeric error
                        if abs(actual) > 0:
                            error_pct = abs((predicted - actual) / actual) * 100
                        else:
                            error_pct = abs(predicted) * 100
                        
                        evaluation["error"] = error_pct
                        evaluation["accuracy"] = max(0.0, 1.0 - (error_pct / 100.0))
                        evaluation["correct"] = error_pct <= 10.0
                    elif isinstance(predicted, str) and isinstance(actual, str):
                        # String match
                        evaluation["correct"] = predicted == actual
                        evaluation["accuracy"] = 1.0 if predicted == actual else 0.0
                    else:
                        # Boolean or other type
                        evaluation["correct"] = predicted == actual
                        evaluation["accuracy"] = 1.0 if predicted == actual else 0.0
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating prediction: {str(e)}")
            return {"error_message": str(e)}
    
    def _update_metrics_if_needed(self):
        """Update model metrics if enough new data is available."""
        # Check if update is needed
        update_interval_hours = self.tracker_config.get("metrics_update_interval_hours", 6)
        
        if (self.last_calculation is None or 
            (datetime.now() - self.last_calculation).total_seconds() > update_interval_hours * 3600):
            
            # Calculate metrics
            self._calculate_model_metrics()
            
            # Update last calculation time
            self.last_calculation = datetime.now()
            
            # Save metrics
            self._save_model_metrics()
    
    def _calculate_model_metrics(self):
        """Calculate performance metrics for all models."""
        try:
            # Get completed records (those with evaluation)
            completed_records = [r for r in self.performance_history if r.get("evaluation")]
            
            if not completed_records:
                self.logger.info("No completed records for metrics calculation")
                return
            
            # Group by model name
            model_groups = {}
            
            for record in completed_records:
                model_name = record.get("model_name")
                
                if not model_name:
                    continue
                
                if model_name not in model_groups:
                    model_groups[model_name] = []
                
                model_groups[model_name].append(record)
            
            # Calculate metrics for each model
            for model_name, records in model_groups.items():
                # Skip if too few records
                if len(records) < 5:
                    continue
                
                # Calculate based on model type
                if "entry" in model_name.lower():
                    metrics = self._calculate_entry_model_metrics(records)
                elif "exit" in model_name.lower():
                    metrics = self._calculate_exit_model_metrics(records)
                elif "sizing" in model_name.lower():
                    metrics = self._calculate_sizing_model_metrics(records)
                else:
                    metrics = self._calculate_generic_model_metrics(records)
                
                # Add record count and timestamp
                metrics["sample_count"] = len(records)
                metrics["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Store metrics
                self.model_metrics[model_name] = metrics
            
            self.logger.info(f"Calculated metrics for {len(model_groups)} models")
            
        except Exception as e:
            self.logger.error(f"Error calculating model metrics: {str(e)}")
    
    def _calculate_entry_model_metrics(self, records):
        """
        Calculate metrics for entry models.
        
        Args:
            records (list): Prediction records for this model
            
        Returns:
            dict: Performance metrics
        """
        # Basic metrics
        correct_count = sum(1 for r in records if r.get("evaluation", {}).get("correct", False))
        accuracy = correct_count / len(records)
        
        # P&L metrics
        pnl_records = [r for r in records if "pnl_percent" in r.get("evaluation", {})]
        
        if pnl_records:
            pnl_values = [r["evaluation"]["pnl_percent"] for r in pnl_records]
            avg_pnl = sum(pnl_values) / len(pnl_values)
            
            # Calculate profit factor
            gains = sum(pnl for pnl in pnl_values if pnl > 0)
            losses = sum(abs(pnl) for pnl in pnl_values if pnl < 0)
            
            profit_factor = gains / losses if losses > 0 else float('inf')
            
            # Win rate
            win_count = sum(1 for pnl in pnl_values if pnl > 0)
            win_rate = win_count / len(pnl_values)
            
            # Max drawdown calculation for PnL sequence
            pnl_series = pd.Series(pnl_values)
            cumulative_returns = (1 + pnl_series / 100).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = ((cumulative_returns / peak) - 1) * 100
            max_drawdown = abs(drawdown.min())
        else:
            avg_pnl = 0.0
            profit_factor = 1.0
            win_rate = 0.5
            max_drawdown = 0.0
        
        # Market regime breakdown
        regime_metrics = self._calculate_regime_breakdown(records)
        
        return {
            "accuracy": accuracy,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "regime_metrics": regime_metrics
        }
    
    def _calculate_exit_model_metrics(self, records):
        """
        Calculate metrics for exit models.
        
        Args:
            records (list): Prediction records for this model
            
        Returns:
            dict: Performance metrics
        """
        # Basic metrics
        correct_count = sum(1 for r in records if r.get("evaluation", {}).get("correct", False))
        accuracy = correct_count / len(records)
        
        # Opportunity cost analysis
        cost_records = [r for r in records if "opportunity_cost" in r.get("evaluation", {})]
        
        if cost_records:
            opportunity_costs = [r["evaluation"]["opportunity_cost"] for r in cost_records]
            avg_opportunity_cost = sum(opportunity_costs) / len(opportunity_costs)
            
            # Compare to baseline (e.g., always exiting at time-based rule)
            if "baseline_cost" in cost_records[0].get("outcome", {}):
                baseline_costs = [r["outcome"]["baseline_cost"] for r in cost_records]
                avg_baseline_cost = sum(baseline_costs) / len(baseline_costs)
                
                # Calculate improvement over baseline
                improvement = avg_baseline_cost - avg_opportunity_cost
            else:
                improvement = 0.0
        else:
            avg_opportunity_cost = 0.0
            improvement = 0.0
        
        # Timing efficiency (how close to optimal)
        timing_records = [r for r in records if "timing_efficiency" in r.get("evaluation", {})]
        
        if timing_records:
            timing_values = [r["evaluation"]["timing_efficiency"] for r in timing_records]
            avg_timing = sum(timing_values) / len(timing_values)
        else:
            avg_timing = 0.5  # Default 50% efficiency
        
        # Market regime breakdown
        regime_metrics = self._calculate_regime_breakdown(records)
        
        return {
            "accuracy": accuracy,
            "average_opportunity_cost": avg_opportunity_cost,
            "improvement_over_baseline": improvement,
            "timing_efficiency": avg_timing,
            "regime_metrics": regime_metrics
        }
    
    def _calculate_sizing_model_metrics(self, records):
        """
        Calculate metrics for position sizing models.
        
        Args:
            records (list): Prediction records for this model
            
        Returns:
            dict: Performance metrics
        """
        # Calculate sizing error
        error_records = [r for r in records if "error" in r.get("evaluation", {})]
        
        if error_records:
            errors = [r["evaluation"]["error"] for r in error_records]
            avg_error = sum(errors) / len(errors)
            max_error = max(errors)
        else:
            avg_error = 0.0
            max_error = 0.0
        
        # Calculate risk-adjusted return
        rar_records = [r for r in records if "risk_adjusted_return" in r.get("evaluation", {})]
        
        if rar_records:
            rar_values = [r["evaluation"]["risk_adjusted_return"] for r in rar_records]
            avg_rar = sum(rar_values) / len(rar_values)
            
            # Calculate volatility of returns (for Sharpe ratio)
            if len(rar_values) > 1:
                return_stddev = np.std(rar_values)
                sharpe_ratio = avg_rar / return_stddev if return_stddev > 0 else 0.0
            else:
                sharpe_ratio = 0.0
        else:
            avg_rar = 0.0
            sharpe_ratio = 0.0
        
        # Calculate drawdown if PnL available
        pnl_records = [r for r in records if "pnl_percent" in r.get("evaluation", {})]
        
        if pnl_records:
            # Get PnL values
            pnl_values = [r["evaluation"]["pnl_percent"] for r in pnl_records]
            
            # Calculate drawdown
            if len(pnl_values) > 1:
                pnl_series = pd.Series(pnl_values)
                cumulative_returns = (1 + pnl_series / 100).cumprod()
                peak = cumulative_returns.expanding().max()
                drawdown = ((cumulative_returns / peak) - 1) * 100
                max_drawdown = abs(drawdown.min())
            else:
                max_drawdown = 0.0
        else:
            max_drawdown = 0.0
        
        # Market regime breakdown
        regime_metrics = self._calculate_regime_breakdown(records)
        
        return {
            "average_error": avg_error,
            "max_error": max_error,
            "risk_adjusted_return": avg_rar,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "regime_metrics": regime_metrics
        }
    
    def _calculate_generic_model_metrics(self, records):
        """
        Calculate metrics for generic prediction models.
        
        Args:
            records (list): Prediction records for this model
            
        Returns:
            dict: Performance metrics
        """
        # Basic metrics
        correct_count = sum(1 for r in records if r.get("evaluation", {}).get("correct", False))
        accuracy = correct_count / len(records)
        
        # Error calculation for numeric predictions
        error_records = [r for r in records if "error" in r.get("evaluation", {})]
        
        if error_records:
            errors = [r["evaluation"]["error"] for r in error_records]
            avg_error = sum(errors) / len(errors)
            mae = sum(abs(e) for e in errors) / len(errors)  # Mean Absolute Error
        else:
            avg_error = 0.0
            mae = 0.0
        
        # Market regime breakdown
        regime_metrics = self._calculate_regime_breakdown(records)
        
        return {
            "accuracy": accuracy,
            "average_error": avg_error,
            "mean_absolute_error": mae,
            "regime_metrics": regime_metrics
        }
    
    def _calculate_regime_breakdown(self, records):
        """
        Calculate metrics breakdown by market regime.
        
        Args:
            records (list): Prediction records
            
        Returns:
            dict: Metrics by regime
        """
        # Group by market regime
        regime_groups = {}
        volatility_groups = {}
        
        for record in records:
            # Extract market conditions
            market_conditions = record.get("market_conditions", {})
            
            # Get regime
            market_regime = market_conditions.get("market_regime", "unknown")
            volatility_regime = market_conditions.get("volatility_regime", "unknown")
            
            # Add to regime groups
            if market_regime not in regime_groups:
                regime_groups[market_regime] = []
            
            regime_groups[market_regime].append(record)
            
            # Add to volatility groups
            if volatility_regime not in volatility_groups:
                volatility_groups[volatility_regime] = []
            
            volatility_groups[volatility_regime].append(record)
        
        # Calculate basic metrics for each regime
        regime_metrics = {}
        
        for regime, regime_records in regime_groups.items():
            if len(regime_records) < 3:
                continue
                
            # Calculate accuracy
            correct_count = sum(1 for r in regime_records if r.get("evaluation", {}).get("correct", False))
            accuracy = correct_count / len(regime_records)
            
            # Calculate win rate and PnL if available
            pnl_records = [r for r in regime_records if "pnl_percent" in r.get("evaluation", {})]
            
            if pnl_records:
                pnl_values = [r["evaluation"]["pnl_percent"] for r in pnl_records]
                avg_pnl = sum(pnl_values) / len(pnl_values)
                
                # Win rate
                win_count = sum(1 for pnl in pnl_values if pnl > 0)
                win_rate = win_count / len(pnl_values)
            else:
                avg_pnl = 0.0
                win_rate = 0.0
            
            regime_metrics[regime] = {
                "accuracy": accuracy,
                "sample_count": len(regime_records),
                "win_rate": win_rate,
                "avg_pnl": avg_pnl
            }
        
        # Calculate for volatility regimes
        volatility_metrics = {}
        
        for regime, regime_records in volatility_groups.items():
            if len(regime_records) < 3:
                continue
                
            # Calculate accuracy
            correct_count = sum(1 for r in regime_records if r.get("evaluation", {}).get("correct", False))
            accuracy = correct_count / len(regime_records)
            
            # Calculate win rate and PnL if available
            pnl_records = [r for r in regime_records if "pnl_percent" in r.get("evaluation", {})]
            
            if pnl_records:
                pnl_values = [r["evaluation"]["pnl_percent"] for r in pnl_records]
                avg_pnl = sum(pnl_values) / len(pnl_values)
                
                # Win rate
                win_count = sum(1 for pnl in pnl_values if pnl > 0)
                win_rate = win_count / len(pnl_values)
            else:
                avg_pnl = 0.0
                win_rate = 0.0
            
            volatility_metrics[regime] = {
                "accuracy": accuracy,
                "sample_count": len(regime_records),
                "win_rate": win_rate,
                "avg_pnl": avg_pnl
            }
        
        return {
            "market_regime": regime_metrics,
            "volatility_regime": volatility_metrics
        }
    
    def get_model_performance(self, model_name=None, market_regime=None, volatility_regime=None):
        """
        Get performance metrics for a model.
        
        Args:
            model_name (str, optional): Name of the model, or None for all models
            market_regime (str, optional): Filter by market regime
            volatility_regime (str, optional): Filter by volatility regime
            
        Returns:
            dict: Performance metrics
        """
        try:
            # If metrics need updating, do it now
            self._update_metrics_if_needed()
            
            # Return metrics for specific model if requested
            if model_name:
                if model_name in self.model_metrics:
                    metrics = self.model_metrics[model_name]
                    
                    # Filter by regime if requested
                    if market_regime or volatility_regime:
                        regime_metrics = metrics.get("regime_metrics", {})
                        
                        if market_regime and market_regime in regime_metrics.get("market_regime", {}):
                            return regime_metrics["market_regime"][market_regime]
                        
                        if volatility_regime and volatility_regime in regime_metrics.get("volatility_regime", {}):
                            return regime_metrics["volatility_regime"][volatility_regime]
                        
                        # If regime not found, return None
                        return None
                    
                    return metrics
                else:
                    return None
            
            # Return all model metrics
            return self.model_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting model performance: {str(e)}")
            return None
    
    def get_model_comparison(self, model_type=None):
        """
        Compare performance of models of the same type.
        
        Args:
            model_type (str, optional): Type of models to compare ('entry', 'exit', 'sizing')
            
        Returns:
            dict: Comparison results
        """
        try:
            # Filter models by type if specified
            if model_type:
                models = {name: metrics for name, metrics in self.model_metrics.items() 
                         if model_type.lower() in name.lower()}
            else:
                models = self.model_metrics
            
            if not models:
                return {"error": f"No models found of type: {model_type}"}
            
            # Group models by type
            model_groups = {}
            
            for name, metrics in models.items():
                # Determine type from name
                if "entry" in name.lower():
                    group = "entry"
                elif "exit" in name.lower():
                    group = "exit"
                elif "sizing" in name.lower():
                    group = "sizing"
                else:
                    group = "other"
                
                if group not in model_groups:
                    model_groups[group] = {}
                
                model_groups[group][name] = metrics
            
            # Create comparison for each group
            comparison = {}
            
            for group, group_models in model_groups.items():
                # Skip if only one model in group
                if len(group_models) <= 1:
                    comparison[group] = {"models": list(group_models.keys())}
                    continue
                
                # Different comparison metrics by group
                if group == "entry":
                    # Compare on win rate, profit factor, accuracy
                    model_metrics = []
                    
                    for name, metrics in group_models.items():
                        model_metrics.append({
                            "name": name,
                            "win_rate": metrics.get("win_rate", 0.0),
                            "profit_factor": metrics.get("profit_factor", 1.0),
                            "accuracy": metrics.get("accuracy", 0.0),
                            "avg_pnl": metrics.get("avg_pnl", 0.0),
                            "sample_count": metrics.get("sample_count", 0)
                        })
                    
                    # Sort by combined score
                    for metrics in model_metrics:
                        metrics["combined_score"] = (
                            metrics["win_rate"] * 0.4 + 
                            (metrics["profit_factor"] - 1.0) * 0.3 + 
                            metrics["accuracy"] * 0.3
                        )
                    
                    model_metrics.sort(key=lambda x: x["combined_score"], reverse=True)
                    
                    comparison[group] = {
                        "models": model_metrics,
                        "best_model": model_metrics[0]["name"] if model_metrics else None,
                        "comparison_metric": "combined_score"
                    }
                
                elif group == "exit":
                    # Compare on accuracy, opportunity cost, timing efficiency
                    model_metrics = []
                    
                    for name, metrics in group_models.items():
                        model_metrics.append({
                            "name": name,
                            "accuracy": metrics.get("accuracy", 0.0),
                            "opportunity_cost": metrics.get("average_opportunity_cost", 0.0),
                            "improvement": metrics.get("improvement_over_baseline", 0.0),
                            "timing_efficiency": metrics.get("timing_efficiency", 0.0),
                            "sample_count": metrics.get("sample_count", 0)
                        })
                    
                    # Sort by combined score
                    for metrics in model_metrics:
                        metrics["combined_score"] = (
                            metrics["accuracy"] * 0.3 + 
                            metrics["improvement"] * 0.4 + 
                            metrics["timing_efficiency"] * 0.3
                        )
                    
                    model_metrics.sort(key=lambda x: x["combined_score"], reverse=True)
                    
                    comparison[group] = {
                        "models": model_metrics,
                        "best_model": model_metrics[0]["name"] if model_metrics else None,
                        "comparison_metric": "combined_score"
                    }
                
                elif group == "sizing":
                    # Compare on error, risk-adjusted return, Sharpe ratio
                    model_metrics = []
                    
                    for name, metrics in group_models.items():
                        model_metrics.append({
                            "name": name,
                            "average_error": metrics.get("average_error", 0.0),
                            "risk_adjusted_return": metrics.get("risk_adjusted_return", 0.0),
                            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                            "max_drawdown": metrics.get("max_drawdown", 0.0),
                            "sample_count": metrics.get("sample_count", 0)
                        })
                    
                    # Sort by combined score
                    for metrics in model_metrics:
                        metrics["combined_score"] = (
                            metrics["sharpe_ratio"] * 0.5 + 
                            metrics["risk_adjusted_return"] * 0.3 -
                            metrics["average_error"] * 0.01 -  # Lower error is better
                            metrics["max_drawdown"] * 0.01   # Lower drawdown is better
                        )
                    
                    model_metrics.sort(key=lambda x: x["combined_score"], reverse=True)
                    
                    comparison[group] = {
                        "models": model_metrics,
                        "best_model": model_metrics[0]["name"] if model_metrics else None,
                        "comparison_metric": "combined_score"
                    }
                
                else:
                    # Generic comparison on accuracy
                    model_metrics = []
                    
                    for name, metrics in group_models.items():
                        model_metrics.append({
                            "name": name,
                            "accuracy": metrics.get("accuracy", 0.0),
                            "error": metrics.get("average_error", 0.0),
                            "sample_count": metrics.get("sample_count", 0)
                        })
                    
                    model_metrics.sort(key=lambda x: x["accuracy"], reverse=True)
                    
                    comparison[group] = {
                        "models": model_metrics,
                        "best_model": model_metrics[0]["name"] if model_metrics else None,
                        "comparison_metric": "accuracy"
                    }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            return {"error": str(e)}
    
    def get_prediction_history(self, model_name=None, limit=100):
        """
        Get recent prediction history for a model.
        
        Args:
            model_name (str, optional): Name of the model, or None for all models
            limit (int): Maximum number of records to return
            
        Returns:
            list: Recent prediction history
        """
        try:
            # Filter records
            if model_name:
                filtered_records = [r for r in self.performance_history 
                                   if r.get("model_name") == model_name and r.get("evaluation")]
            else:
                filtered_records = [r for r in self.performance_history if r.get("evaluation")]
            
            # Sort by timestamp (most recent first)
            sorted_records = sorted(
                filtered_records,
                key=lambda x: datetime.strptime(x.get("timestamp", "2000-01-01"), "%Y-%m-%d %H:%M:%S"),
                reverse=True
            )
            
            # Limit results
            limited_records = sorted_records[:limit]
            
            # Extract relevant fields for response
            result = []
            
            for record in limited_records:
                result.append({
                    "tracking_id": record.get("tracking_id"),
                    "timestamp": record.get("timestamp"),
                    "model_name": record.get("model_name"),
                    "prediction": record.get("prediction"),
                    "outcome": record.get("outcome"),
                    "accuracy": record.get("evaluation", {}).get("accuracy", 0.0),
                    "correct": record.get("evaluation", {}).get("correct", False)
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting prediction history: {str(e)}")
            return []
    
    def get_model_learning_curve(self, model_name, window_size=20):
        """
        Get learning curve data showing model improvement over time.
        
        Args:
            model_name (str): Name of the model
            window_size (int): Window size for rolling averages
            
        Returns:
            dict: Learning curve data
        """
        try:
            # Get model records
            model_records = [r for r in self.performance_history 
                            if r.get("model_name") == model_name and r.get("evaluation")]
            
            if not model_records:
                return {"error": f"No data found for model: {model_name}"}
            
            # Sort by timestamp
            sorted_records = sorted(
                model_records,
                key=lambda x: datetime.strptime(x.get("timestamp", "2000-01-01"), "%Y-%m-%d %H:%M:%S")
            )
            
            # Calculate rolling metrics
            timestamps = []
            accuracy_values = []
            
            for i, record in enumerate(sorted_records):
                if i < window_size - 1:
                    continue
                
                # Calculate metrics for window
                window = sorted_records[i-window_size+1:i+1]
                
                # Calculate accuracy
                correct_count = sum(1 for r in window if r.get("evaluation", {}).get("correct", False))
                accuracy = correct_count / window_size
                
                # Add to data
                timestamps.append(record.get("timestamp"))
                accuracy_values.append(accuracy)
            
            # Create learning curve data
            curve_data = {
                "model_name": model_name,
                "window_size": window_size,
                "timestamps": timestamps,
                "accuracy": accuracy_values
            }
            
            # Add win rate if available
            win_rate_values = []
            has_win_rate = any("pnl_percent" in r.get("evaluation", {}) for r in sorted_records)
            
            if has_win_rate:
                for i, record in enumerate(sorted_records):
                    if i < window_size - 1:
                        continue
                    
                    # Calculate win rate for window
                    window = sorted_records[i-window_size+1:i+1]
                    pnl_records = [r for r in window if "pnl_percent" in r.get("evaluation", {})]
                    
                    if pnl_records:
                        win_count = sum(1 for r in pnl_records 
                                       if r.get("evaluation", {}).get("pnl_percent", 0.0) > 0)
                        win_rate = win_count / len(pnl_records)
                    else:
                        win_rate = 0.0
                    
                    win_rate_values.append(win_rate)
                
                curve_data["win_rate"] = win_rate_values
            
            return curve_data
            
        except Exception as e:
            self.logger.error(f"Error getting learning curve: {str(e)}")
            return {"error": str(e)}
