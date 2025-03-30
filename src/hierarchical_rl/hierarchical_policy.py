"""
Hierarchical Policy Module

This module manages policies across the hierarchy of controllers in the
reinforcement learning system, coordinating strategy selection, entry/exit
decisions, and position sizing.
"""

import logging
import numpy as np
import time
from datetime import datetime
import os
import json

class HierarchicalPolicy:
    """
    Coordinates the hierarchical policy structure, managing the meta-controller
    and sub-controllers to create an integrated trading system.
    """
    
    def __init__(self, config, meta_controller, entry_controller, exit_controller, 
                 position_sizing_controller, state_abstractor=None, reward_shaper=None, drive_manager=None):
        """
        Initialize the HierarchicalPolicy.
        
        Args:
            config (dict): Configuration dictionary
            meta_controller: MetaController instance
            entry_controller: EntryController instance
            exit_controller: ExitController instance
            position_sizing_controller: PositionSizingController instance
            state_abstractor: StateAbstractor instance (optional)
            reward_shaper: RewardShaper instance (optional)
            drive_manager: GoogleDriveManager instance (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("hierarchical_rl", {}).get("hierarchical_policy", {})
        self.drive_manager = drive_manager
        
        # Store controllers
        self.meta_controller = meta_controller
        self.entry_controller = entry_controller
        self.exit_controller = exit_controller
        self.position_sizing_controller = position_sizing_controller
        self.state_abstractor = state_abstractor
        self.reward_shaper = reward_shaper
        
        # Initialize state
        self.current_strategy = None
        self.active_trades = {}
        self.strategy_stats = {}
        self.cumulative_reward = 0.0
        self.episode_count = 0
        self.train_mode = config.get("training_mode", False)
        
        # Training parameters
        self.train_frequency = self.config.get("train_frequency", 10)
        self.train_frequency_meta = self.config.get("train_frequency_meta", 50)
        self.update_counter = 0
        
        # Performance tracking
        self.strategy_performance = {
            "momentum": {"count": 0, "rewards": [], "avg_reward": 0.0},
            "mean_reversion": {"count": 0, "rewards": [], "avg_reward": 0.0},
            "volatility_breakout": {"count": 0, "rewards": [], "avg_reward": 0.0},
            "gamma_scalping": {"count": 0, "rewards": [], "avg_reward": 0.0},
            "theta_harvesting": {"count": 0, "rewards": [], "avg_reward": 0.0}
        }
        
        # Directory for saving stats
        self.stats_dir = "data/hierarchical_rl/policy_stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        
        # Load previous stats if available
        self._load_stats()
        
        self.logger.info("HierarchicalPolicy initialized")
    
    def _load_stats(self):
        """Load policy stats if available."""
        try:
            stats_file = f"{self.stats_dir}/policy_stats.json"
            
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                self.strategy_performance = stats.get("strategy_performance", self.strategy_performance)
                self.cumulative_reward = stats.get("cumulative_reward", 0.0)
                self.episode_count = stats.get("episode_count", 0)
                
                self.logger.info(f"Loaded policy stats from {stats_file}")
                
            elif self.drive_manager and self.drive_manager.file_exists("hierarchical_rl/policy_stats/policy_stats.json"):
                stats_data = self.drive_manager.download_file("hierarchical_rl/policy_stats/policy_stats.json")
                stats = json.loads(stats_data)
                
                self.strategy_performance = stats.get("strategy_performance", self.strategy_performance)
                self.cumulative_reward = stats.get("cumulative_reward", 0.0)
                self.episode_count = stats.get("episode_count", 0)
                
                # Save locally
                with open(stats_file, 'w') as f:
                    f.write(stats_data)
                
                self.logger.info("Loaded policy stats from Google Drive")
                
        except Exception as e:
            self.logger.error(f"Error loading policy stats: {str(e)}")
    
    def _save_stats(self):
        """Save policy stats."""
        try:
            stats_file = f"{self.stats_dir}/policy_stats.json"
            
            stats = {
                "strategy_performance": self.strategy_performance,
                "cumulative_reward": self.cumulative_reward,
                "episode_count": self.episode_count,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Upload to Google Drive if available
            if self.drive_manager:
                self.drive_manager.upload_file(
                    "hierarchical_rl/policy_stats/policy_stats.json", 
                    json.dumps(stats, indent=2),
                    mime_type="application/json"
                )
            
            self.logger.info("Saved policy stats")
            
        except Exception as e:
            self.logger.error(f"Error saving policy stats: {str(e)}")
    
    def select_strategy(self, state_features, evaluate=False):
        """
        Select trading strategy using the meta-controller.
        
        Args:
            state_features (dict): Current market state features
            evaluate (bool): If True, use greedy policy
            
        Returns:
            str: Selected trading strategy
        """
        # Use meta-controller to select strategy
        strategy = self.meta_controller.select_strategy(state_features, evaluate)
        self.current_strategy = strategy
        
        # Load strategy-specific models for sub-controllers
        self.entry_controller.load_strategy_specific_model(strategy)
        self.exit_controller.load_strategy_specific_model(strategy)
        self.position_sizing_controller.load_strategy_specific_model(strategy)
        
        self.logger.info(f"Selected strategy: {strategy}")
        return strategy
    
    def decide_entry(self, state_features, evaluate=False):
        """
        Decide whether to enter a trade.
        
        Args:
            state_features (dict): Current market state features
            evaluate (bool): If True, use greedy policy
            
        Returns:
            tuple: (action, position_size_pct)
                action: 0=no entry, 1=enter call, 2=enter put
                position_size_pct: Position size as percentage of account
        """
        if self.current_strategy is None:
            self.select_strategy(state_features, evaluate)
        
        # Get entry decision from entry controller
        entry_action = self.entry_controller.decide_entry(
            state_features, self.current_strategy, evaluate
        )
        
        # If entering a position, determine position size
        if entry_action > 0:  # 1=call, 2=put
            # Get account info
            account_info = state_features.get("account_info", {})
            
            # Get position size from position sizing controller
            position_size_pct = self.position_sizing_controller.decide_position_size(
                state_features, self.current_strategy, account_info, evaluate
            )
            
            # Create trade ID
            trade_id = f"{self.current_strategy}_{entry_action}_{int(time.time())}"
            
            # Store in active trades
            self.active_trades[trade_id] = {
                "strategy": self.current_strategy,
                "entry_action": entry_action,
                "position_size_pct": position_size_pct,
                "entry_state": state_features,
                "entry_time": time.time(),
                "rewards": []
            }
            
            # Increment strategy count
            self.strategy_performance[self.current_strategy]["count"] += 1
            
            return entry_action, position_size_pct, trade_id
        
        return entry_action, 0.0, None
    
    def decide_exit(self, trade_id, state_features, trade_info, evaluate=False):
        """
        Decide whether to exit a trade.
        
        Args:
            trade_id (str): ID of the trade to evaluate
            state_features (dict): Current market state features
            trade_info (dict): Information about the current trade
            evaluate (bool): If True, use greedy policy
            
        Returns:
            int: Exit action (0=hold, 1=exit, 2=partial exit)
        """
        if trade_id not in self.active_trades:
            self.logger.warning(f"Trade {trade_id} not found in active trades")
            return 1  # Exit if trade not found
        
        trade = self.active_trades[trade_id]
        strategy = trade["strategy"]
        
        # Get exit decision from exit controller
        exit_action = self.exit_controller.decide_exit(
            state_features, strategy, trade_info, evaluate
        )
        
        return exit_action
    
    def record_reward(self, trade_id, reward, state_features=None, done=False, final_pnl=None):
        """
        Record reward for a trade and update controllers.
        
        Args:
            trade_id (str): ID of the trade
            reward (float): Reward value
            state_features (dict, optional): Current state features
            done (bool): Whether the trade is completed
            final_pnl (float, optional): Final P&L percentage for the trade
        """
        if trade_id not in self.active_trades:
            self.logger.warning(f"Trade {trade_id} not found in active trades when recording reward")
            return
        
        trade = self.active_trades[trade_id]
        strategy = trade["strategy"]
        
        # Add reward to trade history
        trade["rewards"].append(reward)
        
        # Update strategy performance
        if strategy in self.strategy_performance:
            self.strategy_performance[strategy]["rewards"].append(reward)
            # Calculate new average
            rewards = self.strategy_performance[strategy]["rewards"]
            self.strategy_performance[strategy]["avg_reward"] = sum(rewards) / len(rewards)
        
        # Update cumulative reward
        self.cumulative_reward += reward
        
        # If we have state features, record experience for controllers
        if state_features is not None:
            # Record for meta-controller
            self.meta_controller.record_experience(state_features, reward, done)
            
            # For exit controller, we need trade info
            if "trade_info" in state_features:
