"""
Adaptation Controller Module

This module controls how quickly the meta-learning system adapts to
new market conditions, balancing stability and responsiveness.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class AdaptationController:
    """
    Controls the adaptation rate of the meta-learning system in response
    to changing market conditions, preventing over-adaptation to noise.
    """
    
    def __init__(self, config, drive_manager=None):
        """
        Initialize the AdaptationController.
        
        Args:
            config (dict): Configuration settings
            drive_manager: Optional GoogleDriveManager for data storage
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.drive_manager = drive_manager
        
        # Extract configuration
        self.controller_config = config.get("meta_learning", {}).get("adaptation_controller", {})
        
        # Default parameters
        self.base_learning_rate = self.controller_config.get("base_learning_rate", 0.1)
        self.min_learning_rate = self.controller_config.get("min_learning_rate", 0.01)
        self.max_learning_rate = self.controller_config.get("max_learning_rate", 0.5)
        
        # Market regime settings
        self.regime_learning_rates = self.controller_config.get("regime_learning_rates", {
            "trending": 0.2,      # Faster adaptation in trending markets
            "volatile": 0.05,     # Slower adaptation in volatile markets
            "sideways": 0.1,      # Moderate adaptation in sideways markets
            "bullish": 0.15,      # Slightly faster in bullish markets
            "bearish": 0.1        # Moderate adaptation in bearish markets
        })
        
        # Performance tracking
        self.performance_history = []
        self.current_learning_rates = {}
        self.stability_measures = {}
        
        # Load adaptation state
        self._load_adaptation_state()
        
        self.logger.info("AdaptationController initialized")
    
    def _load_adaptation_state(self):
        """Load adaptation state from storage."""
        state_file = "data/meta_learning/adaptation_state.json"
        
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Load learning rates and stability measures
                self.current_learning_rates = state_data.get("learning_rates", {})
                self.stability_measures = state_data.get("stability_measures", {})
                
                # Load recent performance history
                if "performance_history" in state_data:
                    self.performance_history = state_data["performance_history"]
                
                self.logger.info("Loaded adaptation state from file")
                
            elif self.drive_manager and self.drive_manager.file_exists("adaptation_state.json"):
                # Download from Google Drive
                file_data = self.drive_manager.download_file("adaptation_state.json")
                
                # Parse JSON
                state_data = json.loads(file_data)
                
                # Load learning rates and stability measures
                self.current_learning_rates = state_data.get("learning_rates", {})
                self.stability_measures = state_data.get("stability_measures", {})
                
                # Load recent performance history
                if "performance_history" in state_data:
                    self.performance_history = state_data["performance_history"]
                
                # Save locally
                with open(state_file, 'w') as f:
                    json.dump(state_data, f, indent=2)
                
                self.logger.info("Loaded adaptation state from Google Drive")
                
        except Exception as e:
            self.logger.error(f"Error loading adaptation state: {str(e)}")
            # Start with default learning rates
            self._initialize_default_rates()
    
    def _initialize_default_rates(self):
        """Initialize default learning rates for all components."""
        # Set default learning rates for main system components
        self.current_learning_rates = {
            "strategy_selector": self.base_learning_rate,
            "model_router": self.base_learning_rate,
            "parameter_hub": self.base_learning_rate,
            "entry_models": self.base_learning_rate,
            "exit_models": self.base_learning_rate,
            "sizing_models": self.base_learning_rate
        }
        
        # Initialize stability measures
        self.stability_measures = {
            "performance_variance": 0.0,
            "regime_transitions": 0,
            "strategy_changes": 0
        }
    
    def _save_adaptation_state(self):
        """Save adaptation state to storage."""
        state_file = "data/meta_learning/adaptation_state.json"
        
        try:
            # Create state data
            state_data = {
                "learning_rates": self.current_learning_rates,
                "stability_measures": self.stability_measures,
                "performance_history": self.performance_history[-100:],  # Keep only recent history
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Create directory if needed
            os.makedirs("data/meta_learning", exist_ok=True)
            
            # Save to file
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                self.drive_manager.upload_file(
                    "adaptation_state.json",
                    json.dumps(state_data),
                    mime_type="application/json"
                )
                
            self.logger.debug("Saved adaptation state")
            
        except Exception as e:
            self.logger.error(f"Error saving adaptation state: {str(e)}")
    
    def get_learning_rate(self, component, market_conditions=None):
        """
        Get the current learning rate for a component.
        
        Args:
            component (str): System component
            market_conditions (dict, optional): Current market conditions
            
        Returns:
            float: Current learning rate
        """
        try:
            # Get base rate for component
            base_rate = self.current_learning_rates.get(component, self.base_learning_rate)
            
            # If no market conditions provided, return base rate
            if not market_conditions:
                return base_rate
            
            # Adjust for market regime
            market_regime = market_conditions.get("market_regime", "unknown")
            volatility_regime = market_conditions.get("volatility_regime", "unknown")
            
            # Get regime multiplier
            regime_multiplier = self.regime_learning_rates.get(market_regime, 1.0)
            
            # Adjust for volatility regime
            vol_multipliers = {
                "low": 1.5,     # Learn faster in low volatility
                "normal": 1.0,  # Normal learning in normal volatility
                "high": 0.7,    # Learn slower in high volatility
                "extreme": 0.5  # Learn much slower in extreme volatility
            }
            
            vol_multiplier = vol_multipliers.get(volatility_regime, 1.0)
            
            # Calculate adjusted rate
            adjusted_rate = base_rate * regime_multiplier * vol_multiplier
            
            # Ensure within bounds
            adjusted_rate = max(self.min_learning_rate, min(adjusted_rate, self.max_learning_rate))
            
            return adjusted_rate
            
        except Exception as e:
            self.logger.error(f"Error getting learning rate: {str(e)}")
            return self.base_learning_rate
    
    def update_learning_rates(self, performance_data, market_conditions, stability_metrics=None):
        """
        Update learning rates based on current performance and market conditions.
        
        Args:
            performance_data (dict): Recent performance metrics
            market_conditions (dict): Current market conditions
            stability_metrics (dict, optional): System stability metrics
            
        Returns:
            dict: Updated learning rates
        """
        try:
            # Add to performance history
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "performance": performance_data,
                "market_conditions": market_conditions
            }
            
            self.performance_history.append(record)
            
            # Trim history if needed
            max_history = self.controller_config.get("max_history", 500)
            if len(self.performance_history) > max_history:
                self.performance_history = self.performance_history[-max_history:]
            
            # Update stability measures
            self._update_stability_measures(stability_metrics)
            
            # Adjust learning rates based on performance and stability
            self._adjust_learning_rates()
            
            # Save updated state
            self._save_adaptation_state()
            
            return self.current_learning_rates
            
        except Exception as e:
            self.logger.error(f"Error updating learning rates: {str(e)}")
            return self.current_learning_rates
    
    def _update_stability_measures(self, stability_metrics=None):
        """
        Update system stability measures.
        
        Args:
            stability_metrics (dict, optional): Explicit stability metrics
        """
        try:
            # Update from provided metrics if available
            if stability_metrics:
                for key, value in stability_metrics.items():
                    self.stability_measures[key] = value
            
            # Calculate performance variance from history
            if len(self.performance_history) >= 10:
                # Get recent performance
                recent_records = self.performance_history[-10:]
                
                # Extract PnL values
                pnl_values = []
                for record in recent_records:
                    if "pnl_percent" in record.get("performance", {}):
                        pnl_values.append(record["performance"]["pnl_percent"])
                
                # Calculate variance if enough data
                if len(pnl_values) >= 5:
                    variance = np.var(pnl_values)
                    self.stability_measures["performance_variance"] = variance
            
            # Count recent regime transitions
            if len(self.performance_history) >= 20:
                recent_records = self.performance_history[-20:]
                
                # Count regime changes
                regime_changes = 0
                prev_regime = None
                
                for record in recent_records:
                    regime = record.get("market_conditions", {}).get("market_regime")
                    
                    if regime and prev_regime and regime != prev_regime:
                        regime_changes += 1
                    
                    prev_regime = regime
                
                self.stability_measures["regime_transitions"] = regime_changes
            
        except Exception as e:
            self.logger.error(f"Error updating stability measures: {str(e)}")
    
    def _adjust_learning_rates(self):
        """Adjust learning rates based on stability measures and performance."""
        try:
            # Get stability factors
            variance = self.stability_measures.get("performance_variance", 0.0)
            regime_transitions = self.stability_measures.get("regime_transitions", 0)
            
            # Calculate stability score (0-1, higher means less stable)
            # Normalize variance (typical range 0-2500 for percentage returns)
            norm_variance = min(1.0, variance / 2500)
            
            # Normalize regime transitions (0-10 scale)
            norm_transitions = min(1.0, regime_transitions / 10)
            
            # Overall stability score (weighted)
            stability_score = norm_variance * 0.7 + norm_transitions * 0.3
            
            # Adjust learning rates based on stability
            # Less stable = slower learning
            stability_factor = 1.0 - (stability_score * 0.8)  # Range 0.2-1.0
            
            # Apply adjustments to each component
            for component in self.current_learning_rates:
                # Get current rate
                current_rate = self.current_learning_rates[component]
                
                # Different adjustment logic for different components
                if component == "parameter_hub":
                    # Parameters need more stability
                    adj_factor = stability_factor * 0.8
                elif component in ["entry_models", "exit_models"]:
                    # Models need moderate stability
                    adj_factor = stability_factor * 0.9
                else:
                    # Other components use standard adjustment
                    adj_factor = stability_factor
                
                # Calculate target rate
                target_rate = self.base_learning_rate * adj_factor
                
                # Smooth transition to target rate
                smoothing = self.controller_config.get("rate_smoothing", 0.2)
                new_rate = current_rate * (1 - smoothing) + target_rate * smoothing
                
                # Ensure within bounds
                new_rate = max(self.min_learning_rate, min(new_rate, self.max_learning_rate))
                
                # Update learning rate
                self.current_learning_rates[component] = new_rate
            
        except Exception as e:
            self.logger.error(f"Error adjusting learning rates: {str(e)}")
    
    def get_adaptation_state(self):
        """
        Get current adaptation state information.
        
        Returns:
            dict: Adaptation state data
        """
        return {
            "learning_rates": self.current_learning_rates,
            "stability_measures": self.stability_measures,
            "history_length": len(self.performance_history)
        }
    
    def set_adaptation_speed(self, speed, component=None):
        """
        Manually set adaptation speed.
        
        Args:
            speed (str): Speed setting ('slow', 'medium', 'fast', or 'adaptive')
            component (str, optional): Specific component, or None for all
            
        Returns:
            dict: Updated learning rates
        """
        try:
            # Map speed settings to learning rates
            speed_mapping = {
                "slow": self.min_learning_rate,
                "medium": self.base_learning_rate,
                "fast": self.max_learning_rate,
                "adaptive": None  # Use automatic adjustment
            }
            
            if speed not in speed_mapping:
                self.logger.error(f"Invalid speed setting: {speed}")
                return self.current_learning_rates
            
            # If adaptive, don't change rates
            if speed == "adaptive":
                return self.current_learning_rates
            
            # Get rate for selected speed
            target_rate = speed_mapping[speed]
            
            # Update specific component or all components
            if component:
                if component in self.current_learning_rates:
                    self.current_learning_rates[component] = target_rate
                else:
                    self.logger.warning(f"Unknown component: {component}")
            else:
                # Update all components
                for comp in self.current_learning_rates:
                    self.current_learning_rates[comp] = target_rate
            
            # Save updated state
            self._save_adaptation_state()
            
            return self.current_learning_rates
            
        except Exception as e:
            self.logger.error(f"Error setting adaptation speed: {str(e)}")
            return self.current_learning_rates
    
    def analyze_adaptation_efficiency(self, days=30):
        """
        Analyze how efficiently the system has been adapting.
        
        Args:
            days (int): Number of days to analyze
            
        Returns:
            dict: Adaptation efficiency metrics
        """
        try:
            # Filter to recent history
            cutoff_time = datetime.now() - timedelta(days=days)
            recent_records = []
            
            for record in self.performance_history:
                try:
                    record_time = datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M:%S")
                    if record_time >= cutoff_time:
                        recent_records.append(record)
                except (ValueError, KeyError):
                    continue
            
            if not recent_records:
                return {"error": "No recent data available for analysis"}
            
            # Group by market regime
            regime_groups = {}
            
            for record in recent_records:
                regime = record.get("market_conditions", {}).get("market_regime", "unknown")
                
                if regime not in regime_groups:
                    regime_groups[regime] = []
                
                regime_groups[regime].append(record)
            
            # Calculate adaptation metrics for each regime
            regime_metrics = {}
            
            for regime, records in regime_groups.items():
                if len(records) < 5:
                    continue
                
                # Extract performance over time
                timestamps = []
                performance_values = []
                
                for record in sorted(records, key=lambda x: x["timestamp"]):
                    if "pnl_percent" in record.get("performance", {}):
                        timestamps.append(record["timestamp"])
                        performance_values.append(record["performance"]["pnl_percent"])
                
                if len(performance_values) < 5:
                    continue
                
                # Calculate adaptation metrics
                perf_mean = np.mean(performance_values)
                perf_std = np.std(performance_values)
                perf_trend = np.polyfit(range(len(performance_values)), performance_values, 1)[0]
                
                # Calculate adaptation speed (how quickly performance improves)
                # Positive trend means improving performance
                adaptation_speed = max(0.0, perf_trend)
                
                # Calculate adaptation stability (inverse of volatility)
                # Lower std means more stable adaptation
                adaptation_stability = 1.0 / (1.0 + perf_std / 10.0)
                
                # Overall efficiency (speed and stability)
                adaptation_efficiency = adaptation_speed * 0.7 + adaptation_stability * 0.3
                
                regime_metrics[regime] = {
                    "samples": len(performance_values),
                    "mean_performance": perf_mean,
                    "performance_volatility": perf_std,
                    "performance_trend": perf_trend,
                    "adaptation_speed": adaptation_speed,
                    "adaptation_stability": adaptation_stability,
                    "adaptation_efficiency": adaptation_efficiency
                }
            
            # Calculate overall metrics
            if not regime_metrics:
                return {"error": "Insufficient data for analysis"}
            
            overall_efficiency = np.mean([m["adaptation_efficiency"] for m in regime_metrics.values()])
            overall_speed = np.mean([m["adaptation_speed"] for m in regime_metrics.values()])
            overall_stability = np.mean([m["adaptation_stability"] for m in regime_metrics.values()])
            
            return {
                "overall_efficiency": overall_efficiency,
                "overall_speed": overall_speed,
                "overall_stability": overall_stability,
                "regime_metrics": regime_metrics,
                "analyzed_days": days,
                "sample_count": len(recent_records)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing adaptation efficiency: {str(e)}")
            return {"error": str(e)}
    
    def get_recommended_learning_rates(self, market_conditions):
        """
        Get recommended learning rates for current market conditions.
        
        Args:
            market_conditions (dict): Current market conditions
            
        Returns:
            dict: Recommended learning rates
        """
        try:
            # Get market regime
            market_regime = market_conditions.get("market_regime", "unknown")
            volatility_regime = market_conditions.get("volatility_regime", "unknown")
            
            # Get regime multiplier
            regime_multiplier = self.regime_learning_rates.get(market_regime, 1.0)
            
            # Adjust for volatility regime
            vol_multipliers = {
                "low": 1.5,     # Learn faster in low volatility
                "normal": 1.0,  # Normal learning in normal volatility
                "high": 0.7,    # Learn slower in high volatility
                "extreme": 0.5  # Learn much slower in extreme volatility
            }
            
            vol_multiplier = vol_multipliers.get(volatility_regime, 1.0)
            
            # Calculate recommended rates for each component
            recommended_rates = {}
            
            for component in self.current_learning_rates:
                # Different base rates for different components
                if component == "parameter_hub":
                    base = self.base_learning_rate * 0.8  # Slower for parameters
                elif component in ["strategy_selector", "model_router"]:
                    base = self.base_learning_rate * 1.2  # Faster for meta components
                else:
                    base = self.base_learning_rate
                
                # Calculate adjusted rate
                recommended_rate = base * regime_multiplier * vol_multiplier
                
                # Ensure within bounds
                recommended_rate = max(self.min_learning_rate, min(recommended_rate, self.max_learning_rate))
                
                recommended_rates[component] = recommended_rate
            
            return {
                "recommended_rates": recommended_rates,
                "current_rates": self.current_learning_rates,
                "market_regime": market_regime,
                "volatility_regime": volatility_regime,
                "regime_multiplier": regime_multiplier,
                "volatility_multiplier": vol_multiplier
            }
            
        except Exception as e:
            self.logger.error(f"Error getting recommended learning rates: {str(e)}")
            return {"error": str(e)}
