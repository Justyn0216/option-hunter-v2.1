"""
Strategy Selector Module

This module provides meta-learning capabilities to learn which trading strategies
work best in which market conditions, adapting strategy selection over time.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class StrategySelector:
    """
    Meta-learning system that learns which strategies work best in which conditions.
    """
    
    def __init__(self, config, drive_manager=None):
        """
        Initialize the StrategySelector.
        
        Args:
            config (dict): Configuration settings
            drive_manager: Optional GoogleDriveManager for data storage
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.drive_manager = drive_manager
        
        # Extract configuration
        self.meta_config = config.get("meta_learning", {}).get("strategy_selector", {})
        
        # Available trading strategies
        self.strategies = self.meta_config.get("strategies", [
            "momentum", "mean_reversion", "volatility_expansion", "volatility_contraction",
            "trend_following", "options_premium", "gamma_scalping", "earnings_volatility"
        ])
        
        # Market condition features
        self.market_features = [
            "market_regime", "volatility_regime", "term_structure_slope", 
            "price_trend", "volume_trend", "sentiment_score", "sector_trend"
        ]
        
        # Initialize model
        self.model = None
        self.performance_history = []
        self.last_training_date = None
        
        # Feature preprocessing
        self.categorical_features = ["market_regime", "volatility_regime", "sector_trend"]
        self.numeric_features = ["term_structure_slope", "price_trend", "volume_trend", "sentiment_score"]
        
        self.categorical_encoder = None
        self.numeric_scaler = None
        
        # Create directories
        os.makedirs("data/meta_learning", exist_ok=True)
        
        # Load or create model
        self._initialize_model()
        
        self.logger.info("StrategySelector initialized")
    
    def _initialize_model(self):
        """Initialize or load the strategy selection model."""
        model_file = "data/meta_learning/strategy_selector.joblib"
        
        try:
            # Check if model file exists locally
            if os.path.exists(model_file):
                model_data = joblib.load(model_file)
                self.model = model_data['model']
                self.categorical_encoder = model_data['encoder']
                self.numeric_scaler = model_data['scaler']
                self.last_training_date = model_data.get('last_training_date')
                
                self.logger.info("Loaded strategy selector model from file")
                
            # Check if model exists in Google Drive
            elif self.drive_manager and self.drive_manager.file_exists("strategy_selector.joblib"):
                # Download and load model
                file_data = self.drive_manager.download_file_binary("strategy_selector.joblib")
                
                # Save locally
                with open(model_file, "wb") as f:
                    f.write(file_data)
                
                # Load model
                model_data = joblib.load(model_file)
                self.model = model_data['model']
                self.categorical_encoder = model_data['encoder']
                self.numeric_scaler = model_data['scaler']
                self.last_training_date = model_data.get('last_training_date')
                
                self.logger.info("Loaded strategy selector model from Google Drive")
                
            else:
                # Initialize new model
                self._create_initial_model()
                
            # Load performance history
            self._load_performance_history()
                
        except Exception as e:
            self.logger.error(f"Error initializing strategy selector model: {str(e)}")
            # Fall back to new model
            self._create_initial_model()
    
    def _create_initial_model(self):
        """Create initial strategy selection model."""
        self.logger.info("Creating initial strategy selector model")
        
        # Initialize preprocessing components
        self.categorical_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.numeric_scaler = StandardScaler()
        
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Fit encoder and scaler with default data
        # For encoder, provide examples of each category
        sample_data = {
            "market_regime": ["bullish", "bearish", "sideways", "volatile", "trending"],
            "volatility_regime": ["low", "normal", "high", "extreme"],
            "sector_trend": ["bullish", "bearish", "neutral"]
        }
        
        sample_df = pd.DataFrame({
            feature: np.random.choice(values, size=10) 
            for feature, values in sample_data.items()
        })
        
        self.categorical_encoder.fit(sample_df)
        
        # For scaler, provide reasonable ranges
        sample_numeric = pd.DataFrame({
            "term_structure_slope": np.random.uniform(-0.05, 0.05, 10),
            "price_trend": np.random.uniform(-0.1, 0.1, 10),
            "volume_trend": np.random.uniform(-0.2, 0.2, 10),
            "sentiment_score": np.random.uniform(-1, 1, 10)
        })
        
        self.numeric_scaler.fit(sample_numeric)
        
        # Model isn't trained yet - will be trained when enough data is collected
        self.last_training_date = None
    
    def _load_performance_history(self):
        """Load strategy performance history from storage."""
        history_file = "data/meta_learning/strategy_performance.json"
        
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.performance_history = json.load(f)
                
                self.logger.info(f"Loaded {len(self.performance_history)} performance history records")
                
            elif self.drive_manager and self.drive_manager.file_exists("strategy_performance.json"):
                # Download from Google Drive
                file_data = self.drive_manager.download_file("strategy_performance.json")
                
                # Parse JSON
                self.performance_history = json.loads(file_data)
                
                # Save locally
                with open(history_file, 'w') as f:
                    json.dump(self.performance_history, f, indent=2)
                
                self.logger.info(f"Loaded {len(self.performance_history)} performance history records from Google Drive")
                
            else:
                self.logger.info("No performance history found, starting with empty history")
                
        except Exception as e:
            self.logger.error(f"Error loading performance history: {str(e)}")
            # Start with empty history
            self.performance_history = []
    
    def select_strategy(self, market_conditions):
        """
        Select the best strategy for current market conditions.
        
        Args:
            market_conditions (dict): Current market conditions
            
        Returns:
            dict: Selected strategy and confidence scores
        """
        try:
            # Check if we have a trained model
            if self.model is None or not hasattr(self.model, 'classes_'):
                # No trained model, return default strategy with exploration
                return self._default_strategy_selection()
            
            # Prepare features
            X = self._prepare_features(market_conditions)
            
            # Predict strategy probabilities
            probabilities = self.model.predict_proba(X)[0]
            
            # Map probabilities to strategies
            strategy_probs = {
                strategy: prob for strategy, prob in zip(self.model.classes_, probabilities)
            }
            
            # Get strategies sorted by probability
            sorted_strategies = sorted(
                strategy_probs.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Apply exploration-exploitation tradeoff
            exploration_rate = self.meta_config.get("exploration_rate", 0.1)
            
            if np.random.random() < exploration_rate:
                # Exploration: randomly pick a strategy with probability proportional to its score
                # Convert probabilities to a distribution that sums to 1
                total_prob = sum(prob for _, prob in sorted_strategies)
                normalized_probs = [prob / total_prob for _, prob in sorted_strategies]
                
                # Randomly select based on probabilities
                selected_idx = np.random.choice(len(sorted_strategies), p=normalized_probs)
                selected_strategy = sorted_strategies[selected_idx][0]
                confidence = sorted_strategies[selected_idx][1]
                
                selection_type = "exploration"
            else:
                # Exploitation: pick the highest probability strategy
                selected_strategy = sorted_strategies[0][0]
                confidence = sorted_strategies[0][1]
                
                selection_type = "exploitation"
            
            return {
                "strategy": selected_strategy,
                "confidence": confidence,
                "all_strategies": sorted_strategies,
                "selection_type": selection_type
            }
            
        except Exception as e:
            self.logger.error(f"Error selecting strategy: {str(e)}")
            return self._default_strategy_selection()
    
    def _default_strategy_selection(self):
        """
        Return a default strategy when model not available.
        
        Returns:
            dict: Default strategy selection
        """
        # Randomly select from available strategies
        selected_strategy = np.random.choice(self.strategies)
        
        # Assign equal probabilities to all strategies
        probs = {strategy: 1.0 / len(self.strategies) for strategy in self.strategies}
        sorted_strategies = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "strategy": selected_strategy,
            "confidence": 1.0 / len(self.strategies),
            "all_strategies": sorted_strategies,
            "selection_type": "random"
        }
    
    def _prepare_features(self, market_conditions):
        """
        Prepare features for model prediction.
        
        Args:
            market_conditions (dict): Market condition features
            
        Returns:
            ndarray: Prepared feature vector
        """
        # Extract categorical features
        categorical_data = {}
        for feature in self.categorical_features:
            categorical_data[feature] = [market_conditions.get(feature, "unknown")]
        
        categorical_df = pd.DataFrame(categorical_data)
        
        # Extract numeric features
        numeric_data = {}
        for feature in self.numeric_features:
            numeric_data[feature] = [market_conditions.get(feature, 0.0)]
        
        numeric_df = pd.DataFrame(numeric_data)
        
        # Transform features
        try:
            # Transform categorical features
            categorical_transformed = self.categorical_encoder.transform(categorical_df)
            
            # Transform numeric features
            numeric_transformed = self.numeric_scaler.transform(numeric_df)
            
            # Combine all features
            X = np.hstack([categorical_transformed, numeric_transformed])
            
            return X
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            # Return zeros array as fallback
            feature_count = len(self.categorical_encoder.get_feature_names_out()) + len(self.numeric_features)
            return np.zeros((1, feature_count))
    
    def update_performance(self, strategy, market_conditions, performance_metrics):
        """
        Update strategy performance history.
        
        Args:
            strategy (str): Strategy that was used
            market_conditions (dict): Market conditions during strategy use
            performance_metrics (dict): Performance metrics achieved
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Create performance record
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "strategy": strategy,
                "market_conditions": market_conditions,
                "performance": performance_metrics
            }
            
            # Add to history
            self.performance_history.append(record)
            
            # Save history
            self._save_performance_history()
            
            # Check if retraining is needed
            self._check_retrain_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating performance history: {str(e)}")
            return False
    
    def _save_performance_history(self):
        """Save performance history to storage."""
        history_file = "data/meta_learning/strategy_performance.json"
        
        try:
            with open(history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                self.drive_manager.upload_file(
                    "strategy_performance.json",
                    json.dumps(self.performance_history),
                    mime_type="application/json"
                )
                
        except Exception as e:
            self.logger.error(f"Error saving performance history: {str(e)}")
    
    def _check_retrain_model(self):
        """Check if model retraining is needed based on new data."""
        # Check if we have enough data
        min_samples = self.meta_config.get("min_training_samples", 50)
        
        if len(self.performance_history) < min_samples:
            self.logger.debug(f"Not enough samples for training: {len(self.performance_history)}/{min_samples}")
            return False
        
        # Check training frequency
        training_interval_days = self.meta_config.get("training_interval_days", 7)
        
        if (self.last_training_date is not None and 
            (datetime.now().date() - self.last_training_date).days < training_interval_days):
            self.logger.debug("Training interval not reached")
            return False
        
        # Perform training
        return self.train_model()
    
    def train_model(self):
        """
        Train the strategy selection model using performance history.
        
        Returns:
            bool: True if training was successful
        """
        try:
            self.logger.info("Training strategy selector model")
            
            if len(self.performance_history) < 10:
                self.logger.warning("Not enough data to train strategy selector")
                return False
            
            # Prepare training data
            training_data = []
            
            # Extract data from performance history
            for record in self.performance_history:
                # Skip records with missing data
                if not all(k in record for k in ["strategy", "market_conditions", "performance"]):
                    continue
                
                # Extract relevant features and target
                strategy = record["strategy"]
                market_conditions = record["market_conditions"]
                performance = record["performance"]
                
                # Only include if we have profit/loss data
                if "pnl_percent" not in performance and "profit_factor" not in performance:
                    continue
                
                # Create a training record
                training_record = {
                    "strategy": strategy,
                    **{k: market_conditions.get(k, "unknown") for k in self.categorical_features},
                    **{k: market_conditions.get(k, 0.0) for k in self.numeric_features},
                    "pnl_percent": performance.get("pnl_percent", 0.0),
                    "profit_factor": performance.get("profit_factor", 1.0),
                    "win_rate": performance.get("win_rate", 0.0)
                }
                
                training_data.append(training_record)
            
            # Convert to DataFrame
            df = pd.DataFrame(training_data)
            
            if len(df) < 10:
                self.logger.warning("Not enough valid data to train strategy selector")
                return False
            
            # Find the best strategy for each market condition combination
            # Group by market condition features
            group_features = self.categorical_features + self.numeric_features
            
            # Calculate average performance by strategy and conditions
            performance_by_strategy = df.groupby(["strategy"] + group_features)[["pnl_percent", "profit_factor", "win_rate"]].mean().reset_index()
            
            # Find best strategy for each condition set
            best_strategy_data = []
            
            for _, group in performance_by_strategy.groupby(group_features):
                # Skip if only one strategy in group
                if len(group) <= 1:
                    continue
                
                # Rank strategies by combined performance metric
                group = group.copy()
                group["combined_score"] = (
                    group["pnl_percent"] * 0.4 +  # 40% weight on PnL
                    group["profit_factor"] * 0.3 +  # 30% weight on profit factor
                    group["win_rate"] * 100 * 0.3  # 30% weight on win rate (scaled to 0-100)
                )
                
                # Get best strategy in this condition
                best_strategy = group.loc[group["combined_score"].idxmax()]
                
                # Only include if performance is positive
                if best_strategy["combined_score"] > 0:
                    best_strategy_data.append(best_strategy)
            
            if len(best_strategy_data) < 5:
                self.logger.warning("Not enough diverse conditions for training")
                return False
            
            # Create training dataset
            best_strategy_df = pd.DataFrame(best_strategy_data)
            
            # Features and target
            X_categorical = best_strategy_df[self.categorical_features]
            X_numeric = best_strategy_df[self.numeric_features]
            y = best_strategy_df["strategy"]
            
            # Fit preprocessing components
            self.categorical_encoder.fit(X_categorical)
            self.numeric_scaler.fit(X_numeric)
            
            # Transform features
            X_categorical_transformed = self.categorical_encoder.transform(X_categorical)
            X_numeric_transformed = self.numeric_scaler.transform(X_numeric)
            
            # Combine features
            X = np.hstack([X_categorical_transformed, X_numeric_transformed])
            
            # Train model
            self.model.fit(X, y)
            
            # Update last training date
            self.last_training_date = datetime.now().date()
            
            # Save model
            self._save_model()
            
            self.logger.info(f"Successfully trained strategy selector with {len(best_strategy_df)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training strategy selector model: {str(e)}")
            return False
    
    def _save_model(self):
        """Save model to storage."""
        model_file = "data/meta_learning/strategy_selector.joblib"
        
        try:
            # Create model data dictionary
            model_data = {
                'model': self.model,
                'encoder': self.categorical_encoder,
                'scaler': self.numeric_scaler,
                'last_training_date': self.last_training_date
            }
            
            # Save to file
            joblib.dump(model_data, model_file)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                with open(model_file, 'rb') as f:
                    self.drive_manager.upload_file(
                        "strategy_selector.joblib",
                        f.read(),
                        mime_type="application/octet-stream"
                    )
                
            self.logger.info("Strategy selector model saved")
            
        except Exception as e:
            self.logger.error(f"Error saving strategy selector model: {str(e)}")
    
    def get_strategy_performance(self, strategy=None, market_regime=None):
        """
        Get performance statistics for strategies.
        
        Args:
            strategy (str, optional): Filter by specific strategy
            market_regime (str, optional): Filter by market regime
            
        Returns:
            dict: Performance statistics
        """
        try:
            # Filter history
            filtered_history = self.performance_history
            
            if strategy:
                filtered_history = [r for r in filtered_history if r.get("strategy") == strategy]
                
            if market_regime:
                filtered_history = [r for r in filtered_history 
                                   if r.get("market_conditions", {}).get("market_regime") == market_regime]
            
            if not filtered_history:
                return {"strategy": strategy, "market_regime": market_regime, "data": []}
            
            # Extract performance metrics
            performance_data = []
            
            for record in filtered_history:
                if "performance" not in record:
                    continue
                    
                perf = record["performance"]
                
                # Only include records with performance data
                if "pnl_percent" not in perf and "profit_factor" not in perf:
                    continue
                
                performance_data.append({
                    "timestamp": record.get("timestamp", ""),
                    "strategy": record.get("strategy", ""),
                    "market_regime": record.get("market_conditions", {}).get("market_regime", "unknown"),
                    "pnl_percent": perf.get("pnl_percent", 0.0),
                    "profit_factor": perf.get("profit_factor", 1.0),
                    "win_rate": perf.get("win_rate", 0.0)
                })
            
            # Calculate summary statistics
            if performance_data:
                pnl_values = [p.get("pnl_percent", 0.0) for p in performance_data]
                profit_factors = [p.get("profit_factor", 1.0) for p in performance_data]
                win_rates = [p.get("win_rate", 0.0) for p in performance_data]
                
                summary = {
                    "count": len(performance_data),
                    "avg_pnl": sum(pnl_values) / len(pnl_values),
                    "avg_profit_factor": sum(profit_factors) / len(profit_factors),
                    "avg_win_rate": sum(win_rates) / len(win_rates),
                    "max_pnl": max(pnl_values),
                    "min_pnl": min(pnl_values)
                }
            else:
                summary = {
                    "count": 0,
                    "avg_pnl": 0.0,
                    "avg_profit_factor": 0.0,
                    "avg_win_rate": 0.0,
                    "max_pnl": 0.0,
                    "min_pnl": 0.0
                }
            
            return {
                "strategy": strategy,
                "market_regime": market_regime,
                "summary": summary,
                "data": performance_data
            }
            
        except Exception as e:
            self.logger.error(f"Error getting strategy performance: {str(e)}")
            return {"strategy": strategy, "market_regime": market_regime, "data": []}
    
    def get_strategy_effectiveness_by_condition(self, condition_feature):
        """
        Analyze strategy effectiveness across different values of a condition.
        
        Args:
            condition_feature (str): Market condition feature to analyze
            
        Returns:
            dict: Effectiveness analysis
        """
        try:
            # Validate condition feature
            if condition_feature not in self.categorical_features and condition_feature not in self.numeric_features:
                return {"error": f"Unknown condition feature: {condition_feature}"}
            
            # Extract data from performance history
            performance_data = []
            
            for record in self.performance_history:
                if "performance" not in record or "market_conditions" not in record:
                    continue
                
                # Extract condition value
                condition_value = record.get("market_conditions", {}).get(condition_feature, "unknown")
                
                # Extract performance metrics
                perf = record["performance"]
                
                performance_data.append({
                    "strategy": record.get("strategy", ""),
                    "condition_value": condition_value,
                    "pnl_percent": perf.get("pnl_percent", 0.0),
                    "profit_factor": perf.get("profit_factor", 1.0),
                    "win_rate": perf.get("win_rate", 0.0)
                })
            
            # Convert to DataFrame
            df = pd.DataFrame(performance_data)
            
            if df.empty:
                return {"condition_feature": condition_feature, "data": []}
            
            # Group by strategy and condition value
            grouped = df.groupby(["strategy", "condition_value"]).agg({
                "pnl_percent": "mean",
                "profit_factor": "mean",
                "win_rate": "mean"
            }).reset_index()
            
            # Convert to list of records
            result_data = grouped.to_dict(orient="records")
            
            # Find best strategy for each condition value
            best_strategies = {}
            
            for condition_value, group in df.groupby("condition_value"):
                # Calculate combined score
                group = group.copy()
                group["combined_score"] = (
                    group["pnl_percent"] * 0.4 +
                    group["profit_factor"] * 0.3 +
                    group["win_rate"] * 100 * 0.3
                )
                
                # Group by strategy
                strategy_scores = group.groupby("strategy")["combined_score"].mean()
                
                # Get best strategy
                if not strategy_scores.empty:
                    best_strategy = strategy_scores.idxmax()
                    best_score = strategy_scores.max()
                    
                    best_strategies[str(condition_value)] = {
                        "strategy": best_strategy,
                        "score": best_score
                    }
            
            return {
                "condition_feature": condition_feature,
                "data": result_data,
                "best_strategies": best_strategies
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing strategy effectiveness: {str(e)}")
            return {"condition_feature": condition_feature, "data": []}
    
    def get_feature_importance(self):
        """
        Get feature importance from the model.
        
        Returns:
            dict: Feature importance scores
        """
        try:
            if self.model is None or not hasattr(self.model, 'feature_importances_'):
                return {"error": "Model not trained or feature importances not available"}
            
            # Get feature names
            categorical_features = self.categorical_encoder.get_feature_names_out()
            numeric_features = self.numeric_features
            
            # Combine feature names
            feature_names = np.concatenate([categorical_features, numeric_features])
            
            # Get feature importances
            importances = self.model.feature_importances_
            
            # Create result dictionary
            result = {name: float(importance) for name, importance in zip(feature_names, importances)}
            
            # Sort by importance
            sorted_result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
            
            return sorted_result
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return {"error": f"Failed to get feature importance: {str(e)}"}
    
    def recommend_strategy_for_symbol(self, symbol, market_conditions):
        """
        Recommend strategy for a specific symbol based on performance history.
        
        Args:
            symbol (str): Trading symbol
            market_conditions (dict): Current market conditions
            
        Returns:
            dict: Strategy recommendation
        """
        try:
            # Filter history for this symbol
            symbol_history = [r for r in self.performance_history 
                             if r.get("performance", {}).get("symbol") == symbol]
            
            if not symbol_history:
                # No history for this symbol, use general recommendation
                return self.select_strategy(market_conditions)
            
            # Extract market regime from conditions
            market_regime = market_conditions.get("market_regime", "unknown")
            
            # Filter by similar market regime
            regime_history = [r for r in symbol_history 
                             if r.get("market_conditions", {}).get("market_regime") == market_regime]
            
            if not regime_history:
                # No history for this regime, use all symbol history
                regime_history = symbol_history
            
            # Calculate performance by strategy
            strategy_performance = {}
            
            for record in regime_history:
                strategy = record.get("strategy", "")
                perf = record.get("performance", {})
                
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {
                        "count": 0,
                        "total_pnl": 0.0,
                        "total_profit_factor": 0.0,
                        "total_win_rate": 0.0
                    }
                
                # Update statistics
                strategy_performance[strategy]["count"] += 1
                strategy_performance[strategy]["total_pnl"] += perf.get("pnl_percent", 0.0)
                strategy_performance[strategy]["total_profit_factor"] += perf.get("profit_factor", 1.0)
                strategy_performance[strategy]["total_win_rate"] += perf.get("win_rate", 0.0)
            
            # Calculate averages
            for strategy in strategy_performance:
                count = strategy_performance[strategy]["count"]
                if count > 0:
                    strategy_performance[strategy]["avg_pnl"] = strategy_performance[strategy]["total_pnl"] / count
                    strategy_performance[strategy]["avg_profit_factor"] = strategy_performance[strategy]["total_profit_factor"] / count
                    strategy_performance[strategy]["avg_win_rate"] = strategy_performance[strategy]["total_win_rate"] / count
                    
                    # Calculate combined score
                    strategy_performance[strategy]["score"] = (
                        strategy_performance[strategy]["avg_pnl"] * 0.4 +
                        strategy_performance[strategy]["avg_profit_factor"] * 0.3 +
                        strategy_performance[strategy]["avg_win_rate"] * 100 * 0.3
                    )
            
            # Sort strategies by score
            sorted_strategies = sorted(
                strategy_performance.items(),
                key=lambda x: x[1].get("score", 0),
                reverse=True
            )
            
            # Apply exploration-exploitation
            exploration_rate = self.meta_config.get("symbol_exploration_rate", 0.2)
            
            if np.random.random() < exploration_rate or not sorted_strategies:
                # Exploration or no data: fall back to general model
                return self.select_strategy(market_conditions)
            else:
                # Exploitation: use best strategy for this symbol
                best_strategy = sorted_strategies[0][0]
                score = sorted_strategies[0][1].get("score", 0)
                
                return {
                    "strategy": best_strategy,
                    "confidence": min(1.0, score / 100),  # Scale to 0-1
                    "all_strategies": [(s, d.get("score", 0)) for s, d in sorted_strategies],
                    "selection_type": "symbol_specific",
                    "symbol": symbol
                }
                
        except Exception as e:
            self.logger.error(f"Error recommending strategy for symbol {symbol}: {str(e)}")
            return self.select_strategy(market_conditions)
