"""
Trade Lifecycle Tracker Module

This module tracks trades through their entire lifecycle, monitoring key metrics,
state transitions, and inflection points to optimize exit decisions.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import time
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class TradeLifecycleTracker:
    """
    Tracks the full lifecycle of option trades, recording key metrics and state changes.
    
    This tracker helps identify optimal exit points by monitoring:
    - Price action and technical indicators
    - Greek value changes over time
    - Profit/loss trajectories
    - Market condition shifts
    - Relative performance against benchmark
    """
    
    def __init__(self, config, drive_manager=None):
        """
        Initialize the trade lifecycle tracker.
        
        Args:
            config (dict): Configuration dictionary
            drive_manager: GoogleDriveManager for saving data (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config.get("exit_strategies", {}).get("trade_lifecycle_tracker", {})
        self.drive_manager = drive_manager
        
        # Configuration parameters
        self.tracking_frequency = self.config.get("tracking_frequency_seconds", 60)
        self.max_metrics_history = self.config.get("max_metrics_history", 1000)
        self.generate_charts = self.config.get("generate_charts", True)
        self.save_data_frequency = self.config.get("save_data_frequency_minutes", 30)
        self.auto_analyze = self.config.get("auto_analyze", True)
        
        # Data storage
        self.active_trades = {}
        self.completed_trades = {}
        self.global_trade_stats = {
            "completed_count": 0,
            "profitable_count": 0,
            "total_pnl": 0.0,
            "avg_duration": 0.0,
            "best_exit_reasons": {},
            "worst_exit_reasons": {}
        }
        
        # Directory for data storage
        self.data_dir = "data/exit_strategies/trade_lifecycle"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(f"{self.data_dir}/charts", exist_ok=True)
        
        # Load previous data if available
        self._load_data()
        
        # Last save timestamp
        self.last_save_time = time.time()
        
        self.logger.info("TradeLifecycleTracker initialized")
    
    def _load_data(self):
        """Load trade history and stats if available."""
        try:
            # Load global stats
            stats_file = f"{self.data_dir}/global_trade_stats.json"
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    self.global_trade_stats = json.load(f)
                self.logger.info(f"Loaded global trade stats with {self.global_trade_stats['completed_count']} completed trades")
            
            # Load completed trades
            completed_file = f"{self.data_dir}/completed_trades.json"
            if os.path.exists(completed_file):
                with open(completed_file, 'r') as f:
                    trades_data = json.load(f)
                    # Convert string timestamps back to float
                    for trade_id, trade in trades_data.items():
                        if 'metrics_history' in trade:
                            trade['metrics_history'] = {float(ts): metrics for ts, metrics in trade['metrics_history'].items()}
                    self.completed_trades = trades_data
                self.logger.info(f"Loaded {len(self.completed_trades)} completed trade histories")
                
            # Try Google Drive if local files not found
            elif self.drive_manager and self.drive_manager.file_exists("exit_strategies/trade_lifecycle/completed_trades.json"):
                content = self.drive_manager.download_file("exit_strategies/trade_lifecycle/completed_trades.json")
                trades_data = json.loads(content)
                # Convert string timestamps back to float
                for trade_id, trade in trades_data.items():
                    if 'metrics_history' in trade:
                        trade['metrics_history'] = {float(ts): metrics for ts, metrics in trade['metrics_history'].items()}
                self.completed_trades = trades_data
                # Save locally
                with open(completed_file, 'w') as f:
                    json.dump(trades_data, f)
                self.logger.info(f"Loaded {len(self.completed_trades)} completed trade histories from Google Drive")
                
        except Exception as e:
            self.logger.error(f"Error loading trade lifecycle data: {str(e)}")
    
    def _save_data(self, force=False):
        """
        Save trade data to disk and Google Drive.
        
        Args:
            force (bool): If True, save regardless of timing
        """
        # Check if it's time to save data
        current_time = time.time()
        time_since_last_save = (current_time - self.last_save_time) / 60  # minutes
        
        if not force and time_since_last_save < self.save_data_frequency:
            return
            
        try:
            # Save global stats
            stats_file = f"{self.data_dir}/global_trade_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(self.global_trade_stats, f, indent=2)
            
            # Save completed trades (limit to most recent 1000)
            completed_keys = list(self.completed_trades.keys())
            if len(completed_keys) > 1000:
                # Sort by completion time and keep the most recent
                sorted_trades = sorted(
                    self.completed_trades.items(), 
                    key=lambda x: x[1].get('exit_time', 0),
                    reverse=True
                )
                self.completed_trades = dict(sorted_trades[:1000])
            
            # Prepare for JSON serialization (convert timestamp keys to strings)
            trades_data = {}
            for trade_id, trade in self.completed_trades.items():
                trade_copy = trade.copy()
                if 'metrics_history' in trade_copy:
                    trade_copy['metrics_history'] = {str(ts): metrics for ts, metrics in trade_copy['metrics_history'].items()}
                trades_data[trade_id] = trade_copy
            
            completed_file = f"{self.data_dir}/completed_trades.json"
            with open(completed_file, 'w') as f:
                json.dump(trades_data, f, indent=2)
            
            # Save to Google Drive if available
            if self.drive_manager:
                self.drive_manager.upload_file(
                    "exit_strategies/trade_lifecycle/global_trade_stats.json",
                    json.dumps(self.global_trade_stats, indent=2),
                    mime_type="application/json"
                )
                
                self.drive_manager.upload_file(
                    "exit_strategies/trade_lifecycle/completed_trades.json",
                    json.dumps(trades_data, indent=2),
                    mime_type="application/json"
                )
            
            self.last_save_time = current_time
            self.logger.info(f"Saved trade lifecycle data with {len(self.completed_trades)} completed trades")
            
        except Exception as e:
            self.logger.error(f"Error saving trade lifecycle data: {str(e)}")
    
    def start_tracking(self, trade_id, trade_data):
        """
        Start tracking a new trade.
        
        Args:
            trade_id (str): Unique identifier for the trade
            trade_data (dict): Initial trade data
            
        Returns:
            bool: Success flag
        """
        if trade_id in self.active_trades:
            self.logger.warning(f"Trade {trade_id} is already being tracked")
            return False
        
        # Create trade tracking structure
        trade_track = {
            "trade_id": trade_id,
            "symbol": trade_data.get("symbol"),
            "option_symbol": trade_data.get("option_symbol"),
            "option_type": trade_data.get("option_type"),
            "strike": trade_data.get("strike"),
            "expiration": trade_data.get("expiration"),
            "entry_price": trade_data.get("entry_price"),
            "entry_time": trade_data.get("entry_time"),
            "quantity": trade_data.get("quantity"),
            "strategy": trade_data.get("strategy", "unknown"),
            "initial_greeks": trade_data.get("greeks", {}),
            "max_profit": 0.0,
            "max_profit_time": None,
            "max_loss": 0.0,
            "max_loss_time": None,
            "entry_iv": trade_data.get("implied_volatility", 0.0),
            "metrics_history": {},
            "state_changes": [],
            "inflection_points": [],
            "last_update_time": time.time()
        }
        
        # Add the initial metrics
        current_metrics = self._extract_current_metrics(trade_data)
        trade_track["metrics_history"][time.time()] = current_metrics
        
        # Record state
        self._record_state_change(trade_track, "entry", "Trade entered", current_metrics)
        
        # Store the trade
        self.active_trades[trade_id] = trade_track
        
        self.logger.info(f"Started tracking trade {trade_id} for {trade_data.get('symbol')} {trade_data.get('option_type')}")
        return True
    
    def update_trade(self, trade_id, trade_data):
        """
        Update metrics for a tracked trade.
        
        Args:
            trade_id (str): Trade identifier
            trade_data (dict): Updated trade data
            
        Returns:
            dict: Updated trade tracking data
        """
        if trade_id not in self.active_trades:
            self.logger.warning(f"Trade {trade_id} is not being tracked")
            return None
        
        trade_track = self.active_trades[trade_id]
        current_time = time.time()
        
        # Check if it's time to update metrics
        seconds_since_last_update = current_time - trade_track["last_update_time"]
        if seconds_since_last_update < self.tracking_frequency:
            return trade_track
        
        # Extract current metrics
        current_metrics = self._extract_current_metrics(trade_data)
        
        # Update metrics history
        trade_track["metrics_history"][current_time] = current_metrics
        
        # Limit the history size
        if len(trade_track["metrics_history"]) > self.max_metrics_history:
            # Remove oldest entries
            oldest_keys = sorted(trade_track["metrics_history"].keys())[:len(trade_track["metrics_history"]) - self.max_metrics_history]
            for key in oldest_keys:
                del trade_track["metrics_history"][key]
        
        # Update max profit/loss
        current_pnl = current_metrics.get("pnl", 0.0)
        current_pnl_pct = current_metrics.get("pnl_percent", 0.0)
        
        if current_pnl > trade_track["max_profit"]:
            trade_track["max_profit"] = current_pnl
            trade_track["max_profit_time"] = current_time
        
        if current_pnl < trade_track["max_loss"]:
            trade_track["max_loss"] = current_pnl
            trade_track["max_loss_time"] = current_time
        
        # Check for inflection points
        self._check_inflection_points(trade_track, current_metrics, current_time)
        
        # Detect state changes
        self._detect_state_changes(trade_track, current_metrics)
        
        # Update last update time
        trade_track["last_update_time"] = current_time
        
        return trade_track
    
    def _extract_current_metrics(self, trade_data):
        """
        Extract current metrics from trade data.
        
        Args:
            trade_data (dict): Current trade data
            
        Returns:
            dict: Extracted metrics
        """
        metrics = {
            "current_price": trade_data.get("current_price", 0.0),
            "underlying_price": trade_data.get("underlying_price", 0.0),
            "pnl": trade_data.get("current_pnl", 0.0),
            "pnl_percent": trade_data.get("current_pnl_percent", 0.0),
            "implied_volatility": trade_data.get("implied_volatility", 0.0),
            "days_to_expiration": trade_data.get("days_to_expiration", 0.0),
            "time_in_trade": trade_data.get("time_in_trade", 0.0),
            "volume": trade_data.get("volume", 0),
            "open_interest": trade_data.get("open_interest", 0),
            "bid": trade_data.get("bid", 0.0),
            "ask": trade_data.get("ask", 0.0),
            "spread": trade_data.get("ask", 0.0) - trade_data.get("bid", 0.0),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Extract greeks if available
        greeks = trade_data.get("greeks", {})
        if greeks:
            metrics.update({
                "delta": greeks.get("delta", 0.0),
                "gamma": greeks.get("gamma", 0.0),
                "theta": greeks.get("theta", 0.0),
                "vega": greeks.get("vega", 0.0),
                "rho": greeks.get("rho", 0.0)
            })
        
        # Add market condition data if available
        if "market_conditions" in trade_data:
            metrics["market_regime"] = trade_data["market_conditions"].get("regime", "unknown")
            metrics["volatility_regime"] = trade_data["market_conditions"].get("volatility", "unknown")
            metrics["sentiment"] = trade_data["market_conditions"].get("sentiment", "neutral")
        
        return metrics
    
    def _check_inflection_points(self, trade_track, current_metrics, current_time):
        """
        Check for significant inflection points in trade metrics.
        
        Args:
            trade_track (dict): Trade tracking data
            current_metrics (dict): Current metrics
            current_time (float): Current timestamp
        """
        # Need at least a few data points to detect inflection points
        if len(trade_track["metrics_history"]) < 3:
            return
        
        # Get metrics history as a list, sorted by time
        history = sorted(trade_track["metrics_history"].items())
        
        # Check for PnL inflection points
        if len(history) >= 3:
            prev_2_pnl = history[-3][1].get("pnl_percent", 0)
            prev_1_pnl = history[-2][1].get("pnl_percent", 0)
            current_pnl = current_metrics.get("pnl_percent", 0)
            
            # Check for reversal (profit to loss)
            if prev_2_pnl < prev_1_pnl and prev_1_pnl > current_pnl and prev_1_pnl > 5:
                # Significant profit that's starting to decline
                inflection = {
                    "type": "profit_reversal",
                    "time": current_time,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "value": prev_1_pnl,
                    "current_value": current_pnl,
                    "description": f"Profit reversal detected: {prev_1_pnl:.2f}% -> {current_pnl:.2f}%"
                }
                trade_track["inflection_points"].append(inflection)
                self.logger.info(f"Detected profit reversal for {trade_track['trade_id']}: {inflection['description']}")
            
            # Check for acceleration (loss to bigger loss)
            if prev_2_pnl > prev_1_pnl and prev_1_pnl > current_pnl and current_pnl < -5:
                # Accelerating loss
                inflection = {
                    "type": "loss_acceleration",
                    "time": current_time,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "value": current_pnl,
                    "previous_value": prev_1_pnl,
                    "description": f"Loss acceleration detected: {prev_1_pnl:.2f}% -> {current_pnl:.2f}%"
                }
                trade_track["inflection_points"].append(inflection)
                self.logger.info(f"Detected loss acceleration for {trade_track['trade_id']}: {inflection['description']}")
        
        # Check for Greek inflection points
        if "delta" in current_metrics and len(history) >= 3:
            prev_2_delta = history[-3][1].get("delta", 0)
            prev_1_delta = history[-2][1].get("delta", 0)
            current_delta = current_metrics.get("delta", 0)
            
            # Significant delta change
            delta_change = abs(current_delta - prev_1_delta)
            if delta_change > 0.1:
                inflection = {
                    "type": "delta_change",
                    "time": current_time,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "value": current_delta,
                    "change": delta_change,
                    "description": f"Significant delta change: {prev_1_delta:.2f} -> {current_delta:.2f}"
                }
                trade_track["inflection_points"].append(inflection)
                self.logger.info(f"Detected delta change for {trade_track['trade_id']}: {inflection['description']}")
        
        # Check for IV inflection points
        if len(history) >= 3:
            prev_2_iv = history[-3][1].get("implied_volatility", 0)
            prev_1_iv = history[-2][1].get("implied_volatility", 0)
            current_iv = current_metrics.get("implied_volatility", 0)
            
            # IV trend reversal
            if (prev_2_iv < prev_1_iv and prev_1_iv > current_iv) or \
               (prev_2_iv > prev_1_iv and prev_1_iv < current_iv):
                # IV trend changed
                iv_change_pct = abs(current_iv - prev_1_iv) / prev_1_iv * 100 if prev_1_iv else 0
                if iv_change_pct > 5:  # Only report significant changes
                    inflection = {
                        "type": "iv_reversal",
                        "time": current_time,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "value": current_iv,
                        "previous_value": prev_1_iv,
                        "description": f"IV trend reversal: {prev_1_iv:.2f} -> {current_iv:.2f}"
                    }
                    trade_track["inflection_points"].append(inflection)
                    self.logger.info(f"Detected IV reversal for {trade_track['trade_id']}: {inflection['description']}")
    
    def _detect_state_changes(self, trade_track, current_metrics):
        """
        Detect significant state changes in the trade.
        
        Args:
            trade_track (dict): Trade tracking data
            current_metrics (dict): Current metrics
        """
        # Check for ITM/OTM transitions
        option_type = trade_track["option_type"]
        strike = trade_track["strike"]
        underlying_price = current_metrics.get("underlying_price", 0)
        
        # Determine if option is in the money
        is_itm = (option_type == "call" and underlying_price > strike) or \
                 (option_type == "put" and underlying_price < strike)
        
        # Check last state
        last_state = None
        for state in reversed(trade_track["state_changes"]):
            if state["state"] in ["itm", "otm"]:
                last_state = state["state"]
                break
        
        # Record state change if changed
        if is_itm and last_state != "itm":
            self._record_state_change(
                trade_track, 
                "itm", 
                f"Option moved in-the-money: {underlying_price} vs {strike}", 
                current_metrics
            )
        elif not is_itm and last_state != "otm":
            self._record_state_change(
                trade_track, 
                "otm", 
                f"Option moved out-of-the-money: {underlying_price} vs {strike}", 
                current_metrics
            )
        
        # Check for significant profit/loss states
        pnl_percent = current_metrics.get("pnl_percent", 0)
        
        # Check profit milestones
        profit_milestones = [25, 50, 100]
        for milestone in profit_milestones:
            milestone_state = f"profit_{milestone}"
            if pnl_percent >= milestone:
                # Check if we've already recorded this milestone
                if not any(s["state"] == milestone_state for s in trade_track["state_changes"]):
                    self._record_state_change(
                        trade_track,
                        milestone_state,
                        f"Profit milestone reached: {pnl_percent:.2f}% > {milestone}%",
                        current_metrics
                    )
        
        # Check loss milestones
        loss_milestones = [-25, -50, -75]
        for milestone in loss_milestones:
            milestone_state = f"loss_{abs(milestone)}"
            if pnl_percent <= milestone:
                # Check if we've already recorded this milestone
                if not any(s["state"] == milestone_state for s in trade_track["state_changes"]):
                    self._record_state_change(
                        trade_track,
                        milestone_state,
                        f"Loss milestone reached: {pnl_percent:.2f}% < {milestone}%",
                        current_metrics
                    )
        
        # Check for time-based states
        days_to_expiration = current_metrics.get("days_to_expiration", 0)
        
        # Close to expiration warning
        if days_to_expiration <= 3 and not any(s["state"] == "near_expiration" for s in trade_track["state_changes"]):
            self._record_state_change(
                trade_track,
                "near_expiration",
                f"Option approaching expiration: {days_to_expiration:.1f} days left",
                current_metrics
            )
        
        # Check for changes in market regime
        if "market_regime" in current_metrics:
            current_regime = current_metrics["market_regime"]
            
            # Find the last recorded market regime
            last_regime = None
            for state in reversed(trade_track["state_changes"]):
                if state["state"].startswith("market_regime_"):
                    last_regime = state["state"].replace("market_regime_", "")
                    break
            
            # Record change if regime changed
            if current_regime != last_regime:
                self._record_state_change(
                    trade_track,
                    f"market_regime_{current_regime}",
                    f"Market regime changed to: {current_regime}",
                    current_metrics
                )
    
    def _record_state_change(self, trade_track, state, description, metrics):
        """
        Record a state change in the trade lifecycle.
        
        Args:
            trade_track (dict): Trade tracking data
            state (str): New state
            description (str): State change description
            metrics (dict): Current metrics
        """
        state_change = {
            "state": state,
            "time": time.time(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": description,
            "metrics": metrics.copy()
        }
        
        trade_track["state_changes"].append(state_change)
        self.logger.info(f"Trade {trade_track['trade_id']} state change: {description}")
    
    def complete_trade(self, trade_id, exit_data):
        """
        Complete tracking for a trade that has been exited.
        
        Args:
            trade_id (str): Trade identifier
            exit_data (dict): Exit data including price, time, etc.
            
        Returns:
            dict: Complete trade lifecycle data
        """
        if trade_id not in self.active_trades:
            self.logger.warning(f"Trade {trade_id} is not being tracked")
            return None
        
        trade_track = self.active_trades[trade_id]
        
        # Add exit information
        trade_track["exit_price"] = exit_data.get("exit_price", 0.0)
        trade_track["exit_time"] = exit_data.get("exit_time")
        trade_track["exit_reason"] = exit_data.get("exit_reason", "unknown")
        
        # Calculate final metrics
        final_pnl = exit_data.get("pnl", 0.0)
        final_pnl_pct = exit_data.get("pnl_percent", 0.0)
        
        trade_track["final_pnl"] = final_pnl
        trade_track["final_pnl_percent"] = final_pnl_pct
        
        # Extract final greeks if available
        if "greeks" in exit_data:
            trade_track["exit_greeks"] = exit_data["greeks"]
        
        trade_track["exit_iv"] = exit_data.get("implied_volatility", 0.0)
        
        # Calculate trade duration
        if trade_track["entry_time"] and trade_track["exit_time"]:
            try:
                entry_dt = datetime.strptime(trade_track["entry_time"], "%Y-%m-%d %H:%M:%S")
                exit_dt = datetime.strptime(trade_track["exit_time"], "%Y-%m-%d %H:%M:%S")
                duration = exit_dt - entry_dt
                trade_track["duration"] = str(duration)
                trade_track["duration_hours"] = duration.total_seconds() / 3600
            except (ValueError, TypeError):
                trade_track["duration"] = "unknown"
                trade_track["duration_hours"] = 0
        
        # Add final metrics
        final_metrics = self._extract_current_metrics(exit_data)
        trade_track["metrics_history"][time.time()] = final_metrics
        
        # Record exit state
        self._record_state_change(trade_track, "exit", f"Trade exited: {exit_data.get('exit_reason')}", final_metrics)
        
        # Move to completed trades
        self.completed_trades[trade_id] = trade_track
        del self.active_trades[trade_id]
        
        # Update global stats
        self.global_trade_stats["completed_count"] += 1
        self.global_trade_stats["total_pnl"] += final_pnl
        
        if final_pnl > 0:
            self.global_trade_stats["profitable_count"] += 1
        
        # Track exit reasons
        exit_reason = exit_data.get("exit_reason", "unknown")
        if exit_reason in self.global_trade_stats["best_exit_reasons"]:
            self.global_trade_stats["best_exit_reasons"][exit_reason] += 1
        else:
            self.global_trade_stats["best_exit_reasons"][exit_reason] = 1
        
        # Calculate average duration
        if "duration_hours" in trade_track and trade_track["duration_hours"] > 0:
            total_trades = self.global_trade_stats["completed_count"]
            current_avg = self.global_trade_stats["avg_duration"]
            new_duration = trade_track["duration_hours"]
            
            # Update running average
            self.global_trade_stats["avg_duration"] = (current_avg * (total_trades - 1) + new_duration) / total_trades
        
        # Generate trade analysis charts
        if self.generate_charts:
            self._generate_trade_charts(trade_id, trade_track)
        
        # Automatically analyze trade if enabled
        if self.auto_analyze:
            self._analyze_trade(trade_id, trade_track)
        
        # Save data
        self._save_data(force=True)
        
        self.logger.info(f"Completed tracking for trade {trade_id} with P&L: ${final_pnl:.2f} ({final_pnl_pct:.2f}%)")
        return trade_track
    
    def _generate_trade_charts(self, trade_id, trade_track):
        """
        Generate charts visualizing the trade lifecycle.
        
        Args:
            trade_id (str): Trade identifier
            trade_track (dict): Trade tracking data
        """
        try:
            # Prepare data
            timestamps = []
            prices = []
            pnls = []
            deltas = []
            gammas = []
            thetas = []
            vegas = []
            ivs = []
            
            # Sort by timestamp
            sorted_metrics = sorted(trade_track["metrics_history"].items())
            for ts, metrics in sorted_metrics:
                dt = datetime.fromtimestamp(ts)
                
                timestamps.append(dt)
                prices.append(metrics.get("current_price", 0))
                pnls.append(metrics.get("pnl_percent", 0))
                deltas.append(metrics.get("delta", 0))
                gammas.append(metrics.get("gamma", 0))
                thetas.append(metrics.get("theta", 0))
                vegas.append(metrics.get("vega", 0))
                ivs.append(metrics.get("implied_volatility", 0))
            
            if not timestamps:
                self.logger.warning(f"No data to generate charts for trade {trade_id}")
                return
            
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(12, 16))
            
            # Add title
            option_type = trade_track["option_type"].upper()
            strike = trade_track["strike"]
            symbol = trade_track["symbol"]
            fig.suptitle(f"{symbol} {option_type} {strike} - Trade Lifecycle\nP&L: {trade_track.get('final_pnl_percent', 0):.2f}%", fontsize=16)
            
            # Price and P&L chart
            ax1 = fig.add_subplot(4, 1, 1)
            ax1.plot(timestamps, prices, 'b-', label='Option Price')
            ax1.set_ylabel('Price ($)')
            ax1.set_title('Option Price')
            ax1.grid(True)
            ax1.legend(loc='upper left')
            
            # Format x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Add annotations for entry and exit
            ax1.axhline(y=trade_track.get("entry_price", 0), color='g', linestyle='--', alpha=0.7, label='Entry Price')
            ax1.axhline(y=trade_track.get("exit_price", 0), color='r', linestyle='--', alpha=0.7, label='Exit Price')
            
            # Twin axis for P&L percentage
            ax1b = ax1.twinx()
            ax1b.plot(timestamps, pnls, 'r-', label='P&L %')
            ax1b.set_ylabel('P&L %')
            ax1b.legend(loc='upper right')
            
            # Greek charts
            ax2 = fig.add_subplot(4, 1, 2)
            ax2.plot(timestamps, deltas, 'g-', label='Delta')
            ax2.set_ylabel('Delta')
            ax2.set_title('Delta')
            ax2.grid(True)
            ax2.legend(loc='upper left')
            
            ax3 = fig.add_subplot(4, 1, 3)
            lines = []
            lines.append(ax3.plot(timestamps, gammas, 'c-', label='Gamma')[0])
            lines.append(ax3.plot(timestamps, vegas, 'm-', label='Vega')[0])
            ax3.set_ylabel('Gamma/Vega')
            ax3.set_title('Gamma and Vega')
            ax3.grid(True)
            
            # Twin axis for Theta (often on different scale)
            ax3b = ax3.twinx()
            lines.append(ax3b.plot(timestamps, thetas, 'y-', label='Theta')[0])
            ax3b.set_ylabel('Theta')
            
            # Combined legend
            ax3.legend(lines, [l.get_label() for l in lines], loc='upper left')
            
            # IV chart
            ax4 = fig.add_subplot(4, 1, 4)
            ax4.plot(timestamps, ivs, 'b-', label='IV')
            ax4.set_ylabel('Implied Volatility')
            ax4.set_title('Implied Volatility')
            ax4.set_xlabel('Time')
            ax4.grid(True)
            ax4.legend(loc='upper left')
            
            # Format x-axis
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            
            # Add inflection points
            for inflection in trade_track.get("inflection_points", []):
                inflection_time = datetime.fromtimestamp(inflection.get("time", 0))
                
                # Only add if time is within our chart range
                if min(timestamps) <= inflection_time <= max(timestamps):
                    # Add marker based on inflection type
                    if "profit" in inflection.get("type", ""):
                        ax1b.plot(inflection_time, inflection.get("value", 0), 'ro', markersize=8)
                    elif "loss" in inflection.get("type", ""):
                        ax1b.plot(inflection_time, inflection.get("value", 0), 'rx', markersize=8)
                    elif "delta" in inflection.get("type", ""):
                        ax2.plot(inflection_time, inflection.get("value", 0), 'go', markersize=8)
                    elif "iv" in inflection.get("type", ""):
                        ax4.plot(inflection_time, inflection.get("value", 0), 'bo', markersize=8)
            
            # Add state changes
            for state_change in trade_track.get("state_changes", []):
                state_time = datetime.fromtimestamp(state_change.get("time", 0))
                
                # Only add if time is within our chart range
                if min(timestamps) <= state_time <= max(timestamps):
                    state = state_change.get("state", "")
                    
                    # Add different markers based on state type
                    if state == "entry":
                        ax1.axvline(x=state_time, color='g', linestyle='-', alpha=0.5)
                    elif state == "exit":
                        ax1.axvline(x=state_time, color='r', linestyle='-', alpha=0.5)
                    elif state == "itm":
                        ax1.axvline(x=state_time, color='g', linestyle=':', alpha=0.5)
                    elif state == "otm":
                        ax1.axvline(x=state_time, color='r', linestyle=':', alpha=0.5)
                    elif state.startswith("profit_"):
                        ax1.axvline(x=state_time, color='g', linestyle='--', alpha=0.3)
                    elif state.startswith("loss_"):
                        ax1.axvline(x=state_time, color='r', linestyle='--', alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            # Save chart
            chart_path = f"{self.data_dir}/charts/{trade_id}_lifecycle.png"
            plt.savefig(chart_path)
            plt.close(fig)
            
            self.logger.info(f"Generated trade lifecycle chart for {trade_id}: {chart_path}")
            
            # Upload to Google Drive if available
            if self.drive_manager:
                with open(chart_path, 'rb') as f:
                    self.drive_manager.upload_file(
                        f"exit_strategies/trade_lifecycle/charts/{trade_id}_lifecycle.png",
                        f.read(),
                        mime_type="image/png"
                    )
            
        except Exception as e:
            self.logger.error(f"Error generating trade charts for {trade_id}: {str(e)}")
    
    def _analyze_trade(self, trade_id, trade_track):
        """
        Analyze a completed trade to extract learning insights.
        
        Args:
            trade_id (str): Trade identifier
            trade_track (dict): Trade tracking data
            
        Returns:
            dict: Analysis results
        """
        try:
            # Prepare analysis result
            analysis = {
                "trade_id": trade_id,
                "symbol": trade_track.get("symbol"),
                "option_type": trade_track.get("option_type"),
                "strategy": trade_track.get("strategy"),
                "entry_time": trade_track.get("entry_time"),
                "exit_time": trade_track.get("exit_time"),
                "duration": trade_track.get("duration"),
                "pnl": trade_track.get("final_pnl", 0),
                "pnl_percent": trade_track.get("final_pnl_percent", 0),
                "exit_reason": trade_track.get("exit_reason"),
                "optimal_exit": False,
                "exit_timing_score": 0.0,
                "missed_profit": 0.0,
                "key_insights": [],
                "improvement_suggestions": []
            }
            
            # Check if we exited at or near maximum profit
            max_profit = trade_track.get("max_profit", 0)
            final_pnl = trade_track.get("final_pnl", 0)
            
            # Calculate exit timing score (0-100)
            if max_profit <= 0:
                # No profit opportunity, any exit is fine
                exit_timing_score = 50
            else:
                # Calculate as percentage of maximum profit captured
                exit_timing_score = min(100, (final_pnl / max_profit) * 100)
            
            analysis["exit_timing_score"] = exit_timing_score
            
            # Determine if this was an optimal exit (within 90% of max profit)
            analysis["optimal_exit"] = exit_timing_score >= 90
            
            # Calculate missed profit
            analysis["missed_profit"] = max(0, max_profit - final_pnl)
            
            # Generate key insights
            insights = []
            
            # Analyze profit capture
            if max_profit > 0:
                if exit_timing_score >= 90:
                    insights.append(f"Excellent exit timing, captured {exit_timing_score:.1f}% of maximum profit")
                elif exit_timing_score >= 70:
                    insights.append(f"Good exit timing, captured {exit_timing_score:.1f}% of maximum profit")
                elif exit_timing_score >= 50:
                    insights.append(f"Average exit timing, captured {exit_timing_score:.1f}% of maximum profit")
                else:
                    insights.append(f"Poor exit timing, only captured {exit_timing_score:.1f}% of maximum profit")
                    insights.append(f"Missed ${analysis['missed_profit']:.2f} in potential profit")
            
            # Analyze holding time
            duration_hours = trade_track.get("duration_hours", 0)
            if final_pnl > 0:
                # For profitable trades
                if duration_hours < 24:
                    insights.append(f"Quick profitable trade (held for {duration_hours:.1f} hours)")
                elif duration_hours > 72:
                    insights.append(f"Long-term profitable trade (held for {duration_hours:.1f} hours)")
            else:
                # For losing trades
                if duration_hours > 48:
                    insights.append(f"Held losing position too long ({duration_hours:.1f} hours)")
            
            # Analyze inflection points
            if trade_track.get("inflection_points"):
                profit_reversals = [p for p in trade_track.get("inflection_points", []) if p.get("type") == "profit_reversal"]
                if profit_reversals and not analysis["optimal_exit"]:
                    insights.append(f"Missed {len(profit_reversals)} profit-taking opportunities")
                    # Add specifics for the largest missed reversal
                    if profit_reversals:
                        largest_reversal = max(profit_reversals, key=lambda x: x.get("value", 0))
                        insights.append(f"Largest missed profit-taking opportunity: {largest_reversal.get('value', 0):.2f}%")
            
            # Analyze Greek changes
            greek_insights = self._analyze_greek_changes(trade_track)
            if greek_insights:
                insights.extend(greek_insights)
            
            # Add insights to analysis
            analysis["key_insights"] = insights
            
            # Generate improvement suggestions
            suggestions = []
            
            # Suggest improvements based on exit timing
            if exit_timing_score < 70:
                suggestions.append("Consider using trailing stop-loss to capture more profit")
                
                # Check if there were clear signals to exit
                if profit_reversals:
                    suggestions.append("Watch for profit reversals as exit signals")
                
                # If IV dropped significantly, suggest vega-based exit
                iv_drop = self._check_iv_change(trade_track)
                if iv_drop < -15:  # 15% IV drop
                    suggestions.append(f"Consider IV-based exits when implied volatility drops significantly (dropped {iv_drop:.1f}%)")
            
            # Suggestions for losing trades
            if final_pnl < 0:
                # Check for stop loss discipline
                if abs(final_pnl) > abs(max_profit) * 2:
                    suggestions.append("Implement stricter stop-loss discipline")
                
                # Check if held too long
                if duration_hours > 48:
                    suggestions.append("Consider shorter holding periods for losing trades")
                
                # Check if delta exposure was too high
                max_delta = self._get_max_metric(trade_track, "delta", abs)
                if max_delta > 0.7:
                    suggestions.append(f"Consider reducing position size for high delta ({max_delta:.2f}) trades")
            
            # Add suggestions to analysis
            analysis["improvement_suggestions"] = suggestions
            
            # Store analysis in trade track
            trade_track["analysis"] = analysis
            
            self.logger.info(f"Completed analysis for trade {trade_id} with exit timing score: {exit_timing_score:.1f}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing trade {trade_id}: {str(e)}")
            return None
    
    def _analyze_greek_changes(self, trade_track):
        """
        Analyze significant changes in option Greeks during trade lifecycle.
        
        Args:
            trade_track (dict): Trade tracking data
            
        Returns:
            list: Insights about Greek changes
        """
        insights = []
        
        # Need history to analyze
        if not trade_track.get("metrics_history"):
            return insights
        
        # Get metrics history as sorted list
        history = sorted(trade_track["metrics_history"].items())
        
        # Skip if not enough data points
        if len(history) < 3:
            return insights
        
        # Analyze Delta trends
        if "delta" in history[0][1]:
            initial_delta = history[0][1]["delta"]
            final_delta = history[-1][1]["delta"]
            max_delta = max([m["delta"] for _, m in history if "delta" in m])
            min_delta = min([m["delta"] for _, m in history if "delta" in m])
            delta_range = max_delta - min_delta
            
            if abs(delta_range) > 0.3:
                insights.append(f"Large delta range during trade: {delta_range:.2f} (min: {min_delta:.2f}, max: {max_delta:.2f})")
            
            if abs(final_delta - initial_delta) > 0.2:
                insights.append(f"Significant delta change: {initial_delta:.2f} → {final_delta:.2f}")
        
        # Analyze Gamma spikes
        if "gamma" in history[0][1]:
            max_gamma = max([m["gamma"] for _, m in history if "gamma" in m])
            avg_gamma = sum([m["gamma"] for _, m in history if "gamma" in m]) / len(history)
            
            if max_gamma > avg_gamma * 2 and max_gamma > 0.05:
                insights.append(f"Significant gamma spike detected (max: {max_gamma:.4f}, avg: {avg_gamma:.4f})")
        
        # Analyze Theta decay
        if "theta" in history[0][1]:
            initial_theta = history[0][1]["theta"]
            final_theta = history[-1][1]["theta"]
            max_theta = min([m["theta"] for _, m in history if "theta" in m])  # Theta is negative
            
            if abs(final_theta) > abs(initial_theta) * 1.5:
                insights.append(f"Accelerating theta decay: {initial_theta:.4f} → {final_theta:.4f}")
            
            if abs(max_theta) > abs(initial_theta) * 2:
                insights.append(f"Theta decay spike detected (max: {max_theta:.4f})")
        
        # Analyze Vega changes with IV
        if "vega" in history[0][1] and "implied_volatility" in history[0][1]:
            initial_vega = history[0][1]["vega"]
            final_vega = history[-1][1]["vega"]
            
            initial_iv = history[0][1]["implied_volatility"]
            final_iv = history[-1][1]["implied_volatility"]
            
            # Calculate percentage change in IV
            iv_change_pct = ((final_iv - initial_iv) / initial_iv) * 100 if initial_iv else 0
            
            if abs(iv_change_pct) > 15:
                insights.append(f"Large IV change during trade: {iv_change_pct:.1f}% ({initial_iv:.2f} → {final_iv:.2f})")
            
            if abs(final_vega - initial_vega) / (initial_vega or 0.01) > 0.3:
                insights.append(f"Significant vega change: {initial_vega:.4f} → {final_vega:.4f}")
        
        return insights
    
    def _check_iv_change(self, trade_track):
        """
        Calculate percentage change in implied volatility during trade.
        
        Args:
            trade_track (dict): Trade tracking data
            
        Returns:
            float: Percentage change in IV
        """
        entry_iv = trade_track.get("entry_iv", 0)
        exit_iv = trade_track.get("exit_iv", 0)
        
        if entry_iv == 0:
            return 0
            
        return ((exit_iv - entry_iv) / entry_iv) * 100
    
    def _get_max_metric(self, trade_track, metric_name, transform=None):
        """
        Get maximum value of a metric from trade history.
        
        Args:
            trade_track (dict): Trade tracking data
            metric_name (str): Metric to analyze
            transform (function, optional): Transform function to apply to values
            
        Returns:
            float: Maximum value
        """
        values = []
        
        for _, metrics in trade_track.get("metrics_history", {}).items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if transform:
                    value = transform(value)
                values.append(value)
        
        if not values:
            return 0
            
        return max(values)
    
    def get_lifecycle_data(self, trade_id):
        """
        Get complete lifecycle data for a specific trade.
        
        Args:
            trade_id (str): Trade identifier
            
        Returns:
            dict: Trade lifecycle data
        """
        # Check active trades first
        if trade_id in self.active_trades:
            return self.active_trades[trade_id]
        
        # Check completed trades
        if trade_id in self.completed_trades:
            return self.completed_trades[trade_id]
        
        return None
    
    def get_active_trades(self):
        """
        Get all currently active trade lifecycles.
        
        Returns:
            dict: Active trades
        """
        return self.active_trades
    
    def get_trade_count(self):
        """
        Get count of active and completed trades.
        
        Returns:
            dict: Trade counts
        """
        return {
            "active_count": len(self.active_trades),
            "completed_count": len(self.completed_trades),
            "total_completed": self.global_trade_stats["completed_count"]
        }
    
    def get_global_stats(self):
        """
        Get global trade statistics.
        
        Returns:
            dict: Global statistics
        """
        # Calculate some additional metrics
        if self.global_trade_stats["completed_count"] > 0:
            win_rate = (self.global_trade_stats["profitable_count"] / 
                         self.global_trade_stats["completed_count"]) * 100
            avg_pnl = self.global_trade_stats["total_pnl"] / self.global_trade_stats["completed_count"]
        else:
            win_rate = 0.0
            avg_pnl = 0.0
        
        stats = self.global_trade_stats.copy()
        stats["win_rate"] = win_rate
        stats["avg_pnl"] = avg_pnl
        
        return stats
    
    def get_best_exit_strategies(self, top_n=5, min_trades=5):
        """
        Get the most successful exit strategies based on average P&L.
        
        Args:
            top_n (int): Number of strategies to return
            min_trades (int): Minimum number of trades to consider
            
        Returns:
            list: Best exit strategies with stats
        """
        # Group trades by exit reason
        exit_stats = {}
        
        for trade_id, trade in self.completed_trades.items():
            exit_reason = trade.get("exit_reason", "unknown")
            pnl = trade.get("final_pnl", 0)
            
            if exit_reason not in exit_stats:
                exit_stats[exit_reason] = {
                    "count": 0,
                    "total_pnl": 0,
                    "profitable_count": 0
                }
            
            exit_stats[exit_reason]["count"] += 1
            exit_stats[exit_reason]["total_pnl"] += pnl
            
            if pnl > 0:
                exit_stats[exit_reason]["profitable_count"] += 1
        
        # Calculate averages and win rates
        for reason, stats in exit_stats.items():
            if stats["count"] >= min_trades:
                stats["avg_pnl"] = stats["total_pnl"] / stats["count"]
                stats["win_rate"] = (stats["profitable_count"] / stats["count"]) * 100
            else:
                # Not enough samples
                stats["avg_pnl"] = 0
                stats["win_rate"] = 0
        
        # Filter and sort by average P&L
        valid_exits = [(reason, stats) for reason, stats in exit_stats.items() 
                      if stats["count"] >= min_trades]
        
        sorted_exits = sorted(valid_exits, key=lambda x: x[1]["avg_pnl"], reverse=True)
        
        # Format result
        result = []
        for reason, stats in sorted_exits[:top_n]:
            result.append({
                "exit_reason": reason,
                "count": stats["count"],
                "avg_pnl": stats["avg_pnl"],
                "win_rate": stats["win_rate"],
                "total_pnl": stats["total_pnl"]
            })
        
        return result
    
    def get_worst_exit_strategies(self, bottom_n=5, min_trades=5):
        """
        Get the least successful exit strategies based on average P&L.
        
        Args:
            bottom_n (int): Number of strategies to return
            min_trades (int): Minimum number of trades to consider
            
        Returns:
            list: Worst exit strategies with stats
        """
        # Group trades by exit reason
        exit_stats = {}
        
        for trade_id, trade in self.completed_trades.items():
            exit_reason = trade.get("exit_reason", "unknown")
            pnl = trade.get("final_pnl", 0)
            
            if exit_reason not in exit_stats:
                exit_stats[exit_reason] = {
                    "count": 0,
                    "total_pnl": 0,
                    "profitable_count": 0
                }
            
            exit_stats[exit_reason]["count"] += 1
            exit_stats[exit_reason]["total_pnl"] += pnl
            
            if pnl > 0:
                exit_stats[exit_reason]["profitable_count"] += 1
        
        # Calculate averages and win rates
        for reason, stats in exit_stats.items():
            if stats["count"] >= min_trades:
                stats["avg_pnl"] = stats["total_pnl"] / stats["count"]
                stats["win_rate"] = (stats["profitable_count"] / stats["count"]) * 100
            else:
                # Not enough samples
                stats["avg_pnl"] = 0
                stats["win_rate"] = 0
        
        # Filter and sort by average P&L (ascending)
        valid_exits = [(reason, stats) for reason, stats in exit_stats.items() 
                      if stats["count"] >= min_trades]
        
        sorted_exits = sorted(valid_exits, key=lambda x: x[1]["avg_pnl"])
        
        # Format result
        result = []
        for reason, stats in sorted_exits[:bottom_n]:
            result.append({
                "exit_reason": reason,
                "count": stats["count"],
                "avg_pnl": stats["avg_pnl"],
                "win_rate": stats["win_rate"],
                "total_pnl": stats["total_pnl"]
            })
        
        return result
