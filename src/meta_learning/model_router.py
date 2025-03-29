"""
Model Router Module

This module routes decisions to the best models for current market conditions
and handles ensemble predictions from multiple models.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class ModelRouter:
    """
    Routes decisions to the best models based on current market conditions
    and handles ensemble predictions.
    """
    
    def __init__(self, config, model_performance_tracker=None, drive_manager=None):
        """
        Initialize the ModelRouter.
        
        Args:
            config (dict): Configuration settings
            model_performance_tracker: Optional ModelPerformanceTracker instance
            drive_manager: Optional GoogleDriveManager for data storage
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model_performance_tracker = model_performance_tracker
        self.drive_manager = drive_manager
        
        # Extract configuration
        self.router_config = config.get("meta_learning", {}).get("model_router", {})
        
        # Available models grouped by task
        self.available_models = {
            # Models for entry decisions
            "entry": self.router_config.get("entry_models", [
                "ml_trend", "ml_volatility", "ml_sentiment", "rl_entry"
            ]),
            
            # Models for exit decisions
            "exit": self.router_config.get("exit_models", [
                "rl_exit_timing", "exit_condition_detector", "greek_exit"
            ]),
            
            # Models for position sizing
            "sizing": self.router_config.get("sizing_models", [
                "fixed_sizing", "kelly_criterion", "ml_risk_sizing"
            ])
        }
        
        # Current model weights
        self.model_weights = {task: {model: 1.0 / len(models) for model in models} 
                             for task, models in self.available_models.items()}
        
        # Cached model performance stats
        self.model_stats = {}
        
        # Last update time for weights
        self.last_weight_update = None
        
        # Create directories
        os.makedirs("data/meta_learning", exist_ok=True)
        
        # Load model weights
        self._load_model_weights()
        
        self.logger.info("ModelRouter initialized")
    
    def _load_model_weights(self):
        """Load model weights from storage."""
        weights_file = "data/meta_learning/model_weights.json"
        
        try:
            if os.path.exists(weights_file):
                with open(weights_file, 'r') as f:
                    weights_data = json.load(f)
                
                self.model_weights = weights_data.get("weights", self.model_weights)
                
                if "last_update" in weights_data:
                    self.last_weight_update = datetime.strptime(
                        weights_data["last_update"], "%Y-%m-%d %H:%M:%S"
                    )
                
                self.logger.info("Loaded model weights from file")
                
            elif self.drive_manager and self.drive_manager.file_exists("model_weights.json"):
                # Download from Google Drive
                file_data = self.drive_manager.download_file("model_weights.json")
                
                # Parse JSON
                weights_data = json.loads(file_data)
                
                self.model_weights = weights_data.get("weights", self.model_weights)
                
                if "last_update" in weights_data:
                    self.last_weight_update = datetime.strptime(
                        weights_data["last_update"], "%Y-%m-%d %H:%M:%S"
                    )
                
                # Save locally
                with open(weights_file, 'w') as f:
                    json.dump(weights_data, f, indent=2)
                
                self.logger.info("Loaded model weights from Google Drive")
                
        except Exception as e:
            self.logger.error(f"Error loading model weights: {str(e)}")
            # Keep default weights
    
    def _save_model_weights(self):
        """Save model weights to storage."""
        weights_file = "data/meta_learning/model_weights.json"
        
        try:
            # Create weights data
            weights_data = {
                "weights": self.model_weights,
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to file
            with open(weights_file, 'w') as f:
                json.dump(weights_data, f, indent=2)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                self.drive_manager.upload_file(
                    "model_weights.json",
                    json.dumps(weights_data),
                    mime_type="application/json"
                )
                
            self.logger.info("Saved model weights")
            
        except Exception as e:
            self.logger.error(f"Error saving model weights: {str(e)}")
    
    def route_decision(self, task, market_conditions, predictions=None):
        """
        Route a decision to the appropriate model(s) based on market conditions.
        
        Args:
            task (str): Decision task ('entry', 'exit', 'sizing')
            market_conditions (dict): Current market conditions
            predictions (dict, optional): Predictions from various models
            
        Returns:
            dict: Routing decision with selected models and weights
        """
        try:
            # Check if task is valid
            if task not in self.available_models:
                self.logger.error(f"Invalid task: {task}")
                return {"error": f"Invalid task: {task}"}
            
            # Update model weights if needed
            self._update_weights_if_needed()
            
            # Get market regime
            market_regime = market_conditions.get("market_regime", "unknown")
            volatility_regime = market_conditions.get("volatility_regime", "unknown")
            
            # Get adjusted weights for current conditions
            adjusted_weights = self._adjust_weights_for_conditions(
                task, market_regime, volatility_regime
            )
            
            # If predictions are provided, create weighted ensemble
            if predictions:
                ensemble_result = self._create_ensemble_prediction(task, predictions, adjusted_weights)
                return {
                    "task": task,
                    "market_regime": market_regime,
                    "volatility_regime": volatility_regime,
                    "models": adjusted_weights,
                    "ensemble_result": ensemble_result
                }
            
            # Otherwise, just return model selection
            return {
                "task": task,
                "market_regime": market_regime,
                "volatility_regime": volatility_regime,
                "models": adjusted_weights
            }
            
        except Exception as e:
            self.logger.error(f"Error routing decision for task {task}: {str(e)}")
            return {"error": f"Error routing decision: {str(e)}"}
    
    def _update_weights_if_needed(self):
        """Update model weights based on performance if needed."""
        # Check if update is needed
        update_interval_hours = self.router_config.get("weight_update_interval_hours", 24)
        
        if (self.last_weight_update is None or 
            (datetime.now() - self.last_weight_update).total_seconds() > update_interval_hours * 3600):
            
            # Update model weights
            self._update_model_weights()
            
            # Update last update time
            self.last_weight_update = datetime.now()
            
            # Save weights
            self._save_model_weights()
    
    def _update_model_weights(self):
        """Update model weights based on recent performance."""
        # Skip if no performance tracker available
        if self.model_performance_tracker is None:
            return
        
        try:
            # Get recent model performance stats
            for task in self.available_models:
                task_models = self.available_models[task]
                
                # Get performance for each model
                model_performance = {}
                
                for model in task_models:
                    # Get model stats
                    stats = self.model_performance_tracker.get_model_performance(model)
                    
                    if stats:
                        # Extract performance metrics
                        if task == "entry":
                            # For entry models, use win rate and profit factor
                            win_rate = stats.get("win_rate", 0.5)
                            profit_factor = stats.get("profit_factor", 1.0)
                            
                            # Combined score
                            score = win_rate * 0.5 + (profit_factor - 1) * 0.5
                            
                        elif task == "exit":
                            # For exit models, use improvement over baseline and win rate
                            improvement = stats.get("improvement_over_baseline", 0.0)
                            win_rate = stats.get("win_rate", 0.5)
                            
                            # Combined score
                            score = improvement * 0.7 + win_rate * 0.3
                            
                        elif task == "sizing":
                            # For sizing models, use sharpe ratio and max drawdown
                            sharpe = stats.get("sharpe_ratio", 1.0)
                            max_dd = stats.get("max_drawdown", 20.0)
                            
                            # Combined score (higher is better)
                            score = sharpe * 0.6 - (max_dd / 100) * 0.4
                            
                        else:
                            # Default score
                            score = 0.5
                        
                        # Store score
                        model_performance[model] = max(0.1, score)  # Ensure positive score
                    else:
                        # No stats, use default score
                        model_performance[model] = 0.5
                
                # Calculate new weights
                if model_performance:
                    # Normalize scores to sum to 1
                    total_score = sum(model_performance.values())
                    
                    if total_score > 0:
                        new_weights = {model: score / total_score for model, score in model_performance.items()}
                        
                        # Apply soft update
                        current_weights = self.model_weights[task]
                        update_rate = self.router_config.get("weight_update_rate", 0.2)
                        
                        # Update weights
                        for model in task_models:
                            if model in new_weights and model in current_weights:
                                current_weights[model] = (
                                    (1 - update_rate) * current_weights[model] + 
                                    update_rate * new_weights[model]
                                )
                        
                        # Normalize again to ensure sum to 1
                        weight_sum = sum(current_weights.values())
                        if weight_sum > 0:
                            self.model_weights[task] = {
                                model: weight / weight_sum for model, weight in current_weights.items()
                            }
            
            self.logger.info("Updated model weights based on performance")
            
        except Exception as e:
            self.logger.error(f"Error updating model weights: {str(e)}")
    
    def _adjust_weights_for_conditions(self, task, market_regime, volatility_regime):
        """
        Adjust model weights based on current market conditions.
        
        Args:
            task (str): Decision task
            market_regime (str): Current market regime
            volatility_regime (str): Current volatility regime
            
        Returns:
            dict: Adjusted model weights
        """
        # Start with base weights
        base_weights = self.model_weights.get(task, {})
        
        # Apply regime-specific adjustments
        try:
            # Only apply adjustments if we have regime-specific data
            if self.model_performance_tracker:
                # Get per-regime performance
                regime_adjustments = {}
                
                for model in base_weights:
                    # Get model performance in this regime
                    regime_stats = self.model_performance_tracker.get_model_performance(
                        model, market_regime=market_regime, volatility_regime=volatility_regime
                    )
                    
                    # Only apply adjustment if we have regime-specific stats
                    if regime_stats and regime_stats.get("sample_count", 0) >= 10:
                        # Calculate regime-specific score similar to _update_model_weights
                        if task == "entry":
                            win_rate = regime_stats.get("win_rate", 0.5)
                            profit_factor = regime_stats.get("profit_factor", 1.0)
                            score = win_rate * 0.5 + (profit_factor - 1) * 0.5
                        elif task == "exit":
                            improvement = regime_stats.get("improvement_over_baseline", 0.0)
                            win_rate = regime_stats.get("win_rate", 0.5)
                            score = improvement * 0.7 + win_rate * 0.3
                        elif task == "sizing":
                            sharpe = regime_stats.get("sharpe_ratio", 1.0)
                            max_dd = regime_stats.get("max_drawdown", 20.0)
                            score = sharpe * 0.6 - (max_dd / 100) * 0.4
                        else:
                            score = 0.5
                        
                        # Store adjustment factor (1.0 means no adjustment)
                        # Higher score means higher weight
                        regime_adjustments[model] = max(0.5, min(2.0, score * 2))
                
                # Apply adjustments
                if regime_adjustments:
                    adjusted_weights = {
                        model: weight * regime_adjustments.get(model, 1.0)
                        for model, weight in base_weights.items()
                    }
                    
                    # Normalize to sum to 1
                    weight_sum = sum(adjusted_weights.values())
                    if weight_sum > 0:
                        return {
                            model: weight / weight_sum 
                            for model, weight in adjusted_weights.items()
                        }
            
            # If no adjustments applied, return base weights
            return base_weights
            
        except Exception as e:
            self.logger.error(f"Error adjusting weights for conditions: {str(e)}")
            return base_weights
    
    def _create_ensemble_prediction(self, task, predictions, weights):
        """
        Create an ensemble prediction from multiple models.
        
        Args:
            task (str): Decision task
            predictions (dict): Predictions from various models
            weights (dict): Model weights
            
        Returns:
            dict: Ensemble prediction
        """
        try:
            # Filter to models that have predictions
            available_predictions = {model: pred for model, pred in predictions.items() if model in weights}
            
            if not available_predictions:
                return {"error": "No predictions available from weighted models"}
            
            # Different ensemble methods based on task
            if task == "entry":
                return self._ensemble_entry_prediction(available_predictions, weights)
            elif task == "exit":
                return self._ensemble_exit_prediction(available_predictions, weights)
            elif task == "sizing":
                return self._ensemble_sizing_prediction(available_predictions, weights)
            else:
                return {"error": f"Unsupported task for ensemble: {task}"}
                
        except Exception as e:
            self.logger.error(f"Error creating ensemble prediction: {str(e)}")
            return {"error": f"Ensemble creation failed: {str(e)}"}
    
    def _ensemble_entry_prediction(self, predictions, weights):
        """
        Create an ensemble entry prediction.
        
        Args:
            predictions (dict): Entry predictions from models
            weights (dict): Model weights
            
        Returns:
            dict: Ensemble prediction
        """
        # Entry predictions typically have:
        # - signal (bullish, bearish, neutral)
        # - confidence (0-1)
        # - score (can be different scales)
        
        # Normalize weights for available models
        available_models = list(predictions.keys())
        available_weights = {model: weights[model] for model in available_models}
        weight_sum = sum(available_weights.values())
        
        if weight_sum == 0:
            return {"signal": "neutral", "confidence": 0.0}
        
        normalized_weights = {m: w / weight_sum for m, w in available_weights.items()}
        
        # Count weighted votes for each signal
        signal_votes = {
            "bullish": 0.0,
            "bearish": 0.0,
            "neutral": 0.0
        }
        
        # Total confidence score
        weighted_confidence = 0.0
        
        for model, prediction in predictions.items():
            # Get model weight
            weight = normalized_weights.get(model, 0.0)
            
            # Extract signal and confidence
            signal = prediction.get("signal", "neutral")
            confidence = prediction.get("confidence", 0.5)
            
            # Add weighted vote
            if signal in signal_votes:
                signal_votes[signal] += weight * confidence
            else:
                signal_votes["neutral"] += weight * confidence
            
            # Add to total confidence
            weighted_confidence += weight * confidence
        
        # Determine ensemble signal (highest weighted vote)
        ensemble_signal = max(signal_votes.items(), key=lambda x: x[1])[0]
        
        # Calculate ensemble confidence
        if ensemble_signal == "neutral":
            # For neutral, confidence is lower
            ensemble_confidence = weighted_confidence * 0.8
        else:
            # For directional signals, confidence is the difference between
            # the winning signal and the sum of others
            other_signals = sum(v for s, v in signal_votes.items() if s != ensemble_signal)
            signal_strength = signal_votes[ensemble_signal] - other_signals
            ensemble_confidence = max(0.0, min(1.0, signal_strength + 0.5))
        
        return {
            "signal": ensemble_signal,
            "confidence": ensemble_confidence,
            "signal_votes": signal_votes,
            "models_used": len(predictions),
            "ensemble_type": "weighted_vote"
        }
    
    def _ensemble_exit_prediction(self, predictions, weights):
        """
        Create an ensemble exit prediction.
        
        Args:
            predictions (dict): Exit predictions from models
            weights (dict): Model weights
            
        Returns:
            dict: Ensemble prediction
        """
        # Exit predictions typically have:
        # - should_exit (boolean)
        # - confidence (0-1)
        # - reason (string)
        
        # Normalize weights for available models
        available_models = list(predictions.keys())
        available_weights = {model: weights[model] for model in available_models}
        weight_sum = sum(available_weights.values())
        
        if weight_sum == 0:
            return {"should_exit": False, "confidence": 0.0}
        
        normalized_weights = {m: w / weight_sum for m, w in available_weights.items()}
        
        # Calculate weighted exit signal
        exit_score = 0.0
        exit_reasons = []
        
        for model, prediction in predictions.items():
            # Get model weight
            weight = normalized_weights.get(model, 0.0)
            
            # Extract exit signal and confidence
            should_exit = prediction.get("should_exit", False)
            confidence = prediction.get("confidence", 0.5)
            reason = prediction.get("reason", "")
            
            # Add to exit score if model says exit
            if should_exit:
                exit_score += weight * confidence
                
                # Add reason with model name
                if reason:
                    exit_reasons.append(f"{model}: {reason}")
        
        # Determine if we should exit
        # Use a threshold that can be configured
        exit_threshold = self.router_config.get("exit_threshold", 0.6)
        ensemble_should_exit = exit_score >= exit_threshold
        
        return {
            "should_exit": ensemble_should_exit,
            "confidence": exit_score,
            "threshold": exit_threshold,
            "reasons": exit_reasons,
            "models_used": len(predictions),
            "ensemble_type": "weighted_score"
        }
    
    def _ensemble_sizing_prediction(self, predictions, weights):
        """
        Create an ensemble position sizing prediction.
        
        Args:
            predictions (dict): Sizing predictions from models
            weights (dict): Model weights
            
        Returns:
            dict: Ensemble prediction
        """
        # Sizing predictions typically have:
        # - position_size (dollar amount or percentage)
        # - confidence (0-1)
        # - max_size (optional maximum)
        
        # Normalize weights for available models
        available_models = list(predictions.keys())
        available_weights = {model: weights[model] for model in available_models}
        weight_sum = sum(available_weights.values())
        
        if weight_sum == 0:
            return {"position_size": 0.0, "confidence": 0.0}
        
        normalized_weights = {m: w / weight_sum for m, w in available_weights.items()}
        
        # Track weighted sizes
        weighted_size_sum = 0.0
        weighted_confidence_sum = 0.0
        max_sizes = []
        
        for model, prediction in predictions.items():
            # Get model weight
            weight = normalized_weights.get(model, 0.0)
            
            # Extract position size and confidence
            size = prediction.get("position_size", 0.0)
            confidence = prediction.get("confidence", 0.5)
            
            # Add weighted size
            weighted_size_sum += weight * size
            weighted_confidence_sum += weight * confidence
            
            # Track max size constraints
            if "max_size" in prediction:
                max_sizes.append(prediction["max_size"])
        
        # Apply maximum size constraint if any
        if max_sizes:
            ensemble_size = min(weighted_size_sum, min(max_sizes))
        else:
            ensemble_size = weighted_size_sum
        
        return {
            "position_size": ensemble_size,
            "confidence": weighted_confidence_sum,
            "models_used": len(predictions),
            "ensemble_type": "weighted_average"
        }
    
    def update_model_performance(self, model_name, performance_metrics):
        """
        Update the cached model performance metrics.
        
        Args:
            model_name (str): Name of the model
            performance_metrics (dict): Performance metrics
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Store metrics in cache
            self.model_stats[model_name] = performance_metrics
            
            # Update model weights if we have enough updates
            update_count = self.router_config.get("weight_update_min_samples", 5)
            
            if len(self.model_stats) >= update_count:
                self._update_model_weights()
                self._save_model_weights()
                
                # Clear cache
                self.model_stats = {}
                
                # Update last update time
                self.last_weight_update = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating model performance: {str(e)}")
            return False
    
    def get_model_weights(self, task=None):
        """
        Get current model weights.
        
        Args:
            task (str, optional): Task to get weights for, or None for all tasks
            
        Returns:
            dict: Current model weights
        """
        if task is not None:
            if task in self.model_weights:
                return {task: self.model_weights[task]}
            else:
                return {}
        else:
            return self.model_weights
    
    def get_routing_history(self, days=7):
        """
        Get history of routing decisions.
        
        Args:
            days (int): Number of days of history
            
        Returns:
            list: Routing decision history
        """
        # In a real implementation, this would query a database of routing decisions
        # Here we just return a placeholder
        return []
    
    def set_task_weights(self, task, weights):
        """
        Manually set weights for a task.
        
        Args:
            task (str): Task to set weights for
            weights (dict): Model weights
            
        Returns:
            bool: True if weights were set successfully
        """
        try:
            if task not in self.available_models:
                self.logger.error(f"Invalid task: {task}")
                return False
            
            # Validate models
            for model in weights:
                if model not in self.available_models[task]:
                    self.logger.error(f"Invalid model for task {task}: {model}")
                    return False
            
            # Normalize weights
            weight_sum = sum(weights.values())
            
            if weight_sum <= 0:
                self.logger.error("Sum of weights must be positive")
                return False
            
            normalized_weights = {model: weight / weight_sum for model, weight in weights.items()}
            
            # Add any missing models with zero weight
            for model in self.available_models[task]:
                if model not in normalized_weights:
                    normalized_weights[model] = 0.0
            
            # Update weights
            self.model_weights[task] = normalized_weights
            
            # Save weights
            self._save_model_weights()
            
            self.logger.info(f"Manually set weights for task {task}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting task weights: {str(e)}")
            return False
