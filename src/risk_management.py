"""
Risk Management Module

This module provides risk management functionality for the Option Hunter trading system.
It includes position sizing, portfolio exposure management, drawdown controls,
and various risk limitation strategies.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import math

class RiskManager:
    """
    Risk management system that monitors and controls trading risks.
    
    Features:
    - Position sizing based on volatility and account balance
    - Portfolio exposure controls
    - Maximum drawdown protection
    - Risk per trade limits
    - Correlated position detection
    - Dynamic risk adjustment based on market conditions
    """
    
    def __init__(self, config, tradier_api=None, parameter_hub=None):
        """
        Initialize the RiskManager.
        
        Args:
            config (dict): Risk management configuration
            tradier_api: Tradier API instance for market data
            parameter_hub: MasterParameterHub instance for parameter lookups
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.tradier_api = tradier_api
        self.parameter_hub = parameter_hub
        
        # Set default risk parameters if not in config
        self.risk_params = config.get("risk_management", {})
        
        # Default risk parameters
        self.max_account_risk = self.risk_params.get("max_account_risk_percent", 2.0) / 100
        self.max_position_size = self.risk_params.get("max_position_size_percent", 5.0) / 100
        self.max_sector_exposure = self.risk_params.get("max_sector_exposure_percent", 20.0) / 100
        self.max_correlation_exposure = self.risk_params.get("max_correlation_exposure_percent", 15.0) / 100
        self.max_drawdown = self.risk_params.get("max_drawdown_percent", 25.0) / 100
        self.position_sizing_atr_multiplier = self.risk_params.get("position_sizing_atr_multiplier", 1.0)
        self.allow_overnight_positions = self.risk_params.get("allow_overnight_positions", True)
        self.risk_increases_near_close = self.risk_params.get("risk_increases_near_close", True)
        
        # Tracking variables
        self.current_positions = {}
        self.trade_history = []
        self.sector_exposure = {}
        self.portfolio_value = 0.0
        self.initial_portfolio_value = 0.0
        self.peak_portfolio_value = 0.0
        self.current_drawdown = 0.0
        
        # Load sector mappings
        self.sector_mappings = self._load_sector_mappings()
        
        # Create logs directory
        self.logs_dir = "logs/risk_management"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.logger.info("RiskManager initialized")
    
    def _load_sector_mappings(self):
        """
        Load sector mappings for symbols.
        
        Returns:
            dict: Mapping of symbols to sectors
        """
        # Default to empty mappings
        mappings = {}
        
        # Try to load from file
        try:
            mappings_file = self.risk_params.get("sector_mappings_file", "data/sector_mappings.json")
            if os.path.exists(mappings_file):
                with open(mappings_file, 'r') as f:
                    mappings = json.load(f)
                self.logger.info(f"Loaded sector mappings for {len(mappings)} symbols")
            else:
                self.logger.warning(f"Sector mappings file not found: {mappings_file}")
        except Exception as e:
            self.logger.error(f"Error loading sector mappings: {str(e)}")
        
        return mappings
    
    def update_portfolio_value(self, portfolio_value):
        """
        Update the current portfolio value.
        
        Args:
            portfolio_value (float): Current portfolio value
        """
        # Initialize initial and peak values if not set
        if self.initial_portfolio_value == 0.0:
            self.initial_portfolio_value = portfolio_value
            self.peak_portfolio_value = portfolio_value
        
        # Update current value
        self.portfolio_value = portfolio_value
        
        # Update peak value if current value is higher
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        # Calculate current drawdown
        if self.peak_portfolio_value > 0:
            self.current_drawdown = 1.0 - (portfolio_value / self.peak_portfolio_value)
        else:
            self.current_drawdown = 0.0
        
        # Log significant drawdowns
        if self.current_drawdown > self.max_drawdown * 0.5:
            self.logger.warning(f"Significant drawdown detected: {self.current_drawdown:.2%}, max allowed: {self.max_drawdown:.2%}")
    
    def register_position(self, position):
        """
        Register a new position with the risk manager.
        
        Args:
            position (dict): Position information
        """
        # Extract position details
        symbol = position.get('symbol')
        option_symbol = position.get('option_symbol')
        position_id = option_symbol or symbol
        
        if not position_id:
            self.logger.error("Position missing symbol identifiers")
            return
        
        # Store position
        self.current_positions[position_id] = position
        
        # Update sector exposure
        self._update_sector_exposure()
        
        self.logger.info(f"Registered position: {position_id}")
        
        # Check if position exceeds risk limits
        self._check_position_risk(position)
    
    def unregister_position(self, position_id):
        """
        Unregister a position from the risk manager.
        
        Args:
            position_id (str): Position identifier
        """
        if position_id in self.current_positions:
            # Add to trade history before removing
            position = self.current_positions[position_id]
            self.trade_history.append(position)
            
            # Remove from current positions
            del self.current_positions[position_id]
            
            # Update sector exposure
            self._update_sector_exposure()
            
            self.logger.info(f"Unregistered position: {position_id}")
        else:
            self.logger.warning(f"Position not found for unregistering: {position_id}")
    
    def update_position(self, position_id, updates):
        """
        Update an existing position.
        
        Args:
            position_id (str): Position identifier
            updates (dict): Position updates
        """
        if position_id in self.current_positions:
            # Update position
            self.current_positions[position_id].update(updates)
            
            # Check risk if price updated
            if 'current_price' in updates:
                self._check_position_risk(self.current_positions[position_id])
            
            self.logger.debug(f"Updated position: {position_id}")
        else:
            self.logger.warning(f"Position not found for updating: {position_id}")
    
    def _update_sector_exposure(self):
        """
        Update sector exposure based on current positions.
        """
        # Reset sector exposure
        self.sector_exposure = {}
        
        # Calculate exposure per sector
        for position_id, position in self.current_positions.items():
            symbol = position.get('symbol', '')
            sector = self.sector_mappings.get(symbol, 'Unknown')
            position_size = position.get('position_size', 0.0)
            
            if sector not in self.sector_exposure:
                self.sector_exposure[sector] = 0.0
            
            self.sector_exposure[sector] += position_size
    
    def _check_position_risk(self, position):
        """
        Check if a position exceeds risk limits.
        
        Args:
            position (dict): Position information
        """
        # Check position size
        position_size = position.get('position_size', 0.0)
        max_allowed_size = self.portfolio_value * self.max_position_size
        
        if position_size > max_allowed_size:
            self.logger.warning(
                f"Position size exceeds limit: {position_size:.2f} > {max_allowed_size:.2f} "
                f"({position.get('symbol', 'Unknown')})"
            )
        
        # Check sector exposure
        symbol = position.get('symbol', '')
        sector = self.sector_mappings.get(symbol, 'Unknown')
        sector_exposure = self.sector_exposure.get(sector, 0.0)
        max_sector_exposure = self.portfolio_value * self.max_sector_exposure
        
        if sector_exposure > max_sector_exposure:
            self.logger.warning(
                f"Sector exposure exceeds limit: {sector_exposure:.2f} > {max_sector_exposure:.2f} "
                f"(Sector: {sector})"
            )
    
    def calculate_position_size(self, symbol, entry_price, stop_loss_price=None, market_regime=None):
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            symbol (str): Symbol to trade
            entry_price (float): Entry price
            stop_loss_price (float, optional): Stop loss price
            market_regime (str, optional): Current market regime
            
        Returns:
            float: Recommended position size in dollars
        """
        # Get risk per trade from parameter hub if available
        if self.parameter_hub:
            risk_per_trade = self.parameter_hub.get_parameter(
                "max_risk_per_trade_percentage", 
                self.risk_params.get("max_risk_per_trade_percent", 1.0)
            ) / 100
        else:
            risk_per_trade = self.risk_params.get("max_risk_per_trade_percent", 1.0) / 100
        
        # Calculate maximum risk amount
        max_risk_amount = self.portfolio_value * risk_per_trade
        
        # Adjust based on market regime if provided
        if market_regime:
            if market_regime == 'volatile':
                # Reduce position size in volatile markets
                max_risk_amount *= 0.7
            elif market_regime == 'trending':
                # Increase position size in trending markets
                max_risk_amount *= 1.2
        
        # Check for drawdown protection
        if self.current_drawdown > 0.1:
            # Reduce position size when in drawdown
            drawdown_factor = 1.0 - (self.current_drawdown * 2)
            drawdown_factor = max(0.25, drawdown_factor)  # Don't go below 25%
            max_risk_amount *= drawdown_factor
        
        # Calculate position size based on stop loss distance or fixed percentage
        if stop_loss_price and stop_loss_price > 0:
            # Calculate based on stop loss
            risk_per_share = abs(entry_price - stop_loss_price)
            if risk_per_share > 0:
                # For options, each contract is 100 shares
                position_size = max_risk_amount / risk_per_share * 100
            else:
                # Fallback to fixed percentage
                position_size = max_risk_amount / 0.1
        else:
            # Use ATR for position sizing if available
            if self.tradier_api:
                try:
                    # Get historical data
                    historical_data = self.tradier_api.get_historical_data(
                        symbol=symbol,
                        interval='daily',
                        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                        end_date=datetime.now().strftime('%Y-%m-%d')
                    )
                    
                    # Calculate ATR
                    if not historical_data.empty and len(historical_data) > 5:
                        # Calculate true range
                        historical_data['high_low'] = historical_data['high'] - historical_data['low']
                        historical_data['high_close'] = abs(historical_data['high'] - historical_data['close'].shift(1))
                        historical_data['low_close'] = abs(historical_data['low'] - historical_data['close'].shift(1))
                        historical_data['true_range'] = historical_data[['high_low', 'high_close', 'low_close']].max(axis=1)
                        
                        # Calculate ATR (14-day)
                        atr = historical_data['true_range'].rolling(window=14).mean().iloc[-1]
                        
                        # Use ATR for risk calculation (typically 1-2 ATRs)
                        risk_per_share = atr * self.position_sizing_atr_multiplier
                        position_size = max_risk_amount / risk_per_share * 100
                    else:
                        # Fallback to fixed percentage
                        position_size = max_risk_amount / (entry_price * 0.1)
                except Exception as e:
                    self.logger.error(f"Error calculating ATR: {str(e)}")
                    # Fallback to fixed percentage
                    position_size = max_risk_amount / (entry_price * 0.1)
            else:
                # Fallback to fixed percentage
                position_size = max_risk_amount / (entry_price * 0.1)
        
        # Apply max position size constraint
        max_allowed_position = self.portfolio_value * self.max_position_size
        position_size = min(position_size, max_allowed_position)
        
        # Adjust for time of day if enabled
        if self.risk_increases_near_close:
            hour = datetime.now().hour
            minute = datetime.now().minute
            
            # Reduce position size in the last hour of trading (assuming market closes at 4 PM)
            if hour >= 15:
                # Calculate time remaining in percentage
                time_remaining = (16 - hour) * 60 - minute
                time_factor = time_remaining / 60  # Scale from 0 to 1
                
                # Adjust position size (reduce more as close approaches)
                position_size *= max(0.25, time_factor)
        
        # Ensure minimum sensible position size
        min_position_size = 100  # Minimum $100 position
        position_size = max(position_size, min_position_size)
        
        # Log position sizing
        self.logger.info(
            f"Calculated position size for {symbol}: ${position_size:.2f} "
            f"(max risk: ${max_risk_amount:.2f})"
        )
        
        return position_size
    
    def check_overnight_risk(self):
        """
        Check for overnight risk exposure and make recommendations.
        
        Returns:
            dict: Overnight risk assessment and recommendations
        """
        # Skip if overnight positions are allowed
        if self.allow_overnight_positions:
            return {
                "allow_overnight": True,
                "recommendation": "No action needed, overnight positions allowed",
                "positions_to_close": []
            }
        
        # Get current time
        now = datetime.now()
        
        # Check if close to market close (assuming 4 PM close time)
        if now.hour >= 15 and now.minute >= 30:
            # Close all positions
            positions_to_close = list(self.current_positions.keys())
            
            return {
                "allow_overnight": False,
                "recommendation": "Close all positions before market close",
                "positions_to_close": positions_to_close
            }
        
        # Not near close yet
        return {
            "allow_overnight": False,
            "recommendation": "No action needed yet, monitor for market close",
            "positions_to_close": []
        }
    
    def check_correlated_positions(self):
        """
        Check for correlated positions in the portfolio.
        
        Returns:
            list: Correlated position groups
        """
        # Skip if no positions
        if len(self.current_positions) < 2:
            return []
        
        # Group positions by sector
        sector_positions = {}
        for position_id, position in self.current_positions.items():
            symbol = position.get('symbol', '')
            sector = self.sector_mappings.get(symbol, 'Unknown')
            
            if sector not in sector_positions:
                sector_positions[sector] = []
            
            sector_positions[sector].append(position_id)
        
        # Find sectors with multiple positions
        correlated_groups = []
        
        for sector, positions in sector_positions.items():
            if len(positions) > 1:
                # Calculate total exposure
                total_exposure = sum(
                    self.current_positions[pos_id].get('position_size', 0)
                    for pos_id in positions
                )
                
                # Check if exposure exceeds limit
                if total_exposure > self.portfolio_value * self.max_correlation_exposure:
                    correlated_groups.append({
                        "sector": sector,
                        "positions": positions,
                        "total_exposure": total_exposure,
                        "max_allowed": self.portfolio_value * self.max_correlation_exposure,
                        "excess": total_exposure - (self.portfolio_value * self.max_correlation_exposure)
                    })
        
        return correlated_groups
    
    def check_drawdown_protection(self):
        """
        Check if drawdown protection measures should be activated.
        
        Returns:
            dict: Drawdown status and recommendations
        """
        # Calculate current drawdown
        self.update_portfolio_value(self.portfolio_value)
        
        # Check against maximum allowed drawdown
        if self.current_drawdown > self.max_drawdown:
            return {
                "status": "critical",
                "drawdown": self.current_drawdown,
                "max_allowed": self.max_drawdown,
                "recommendation": "Close all positions and pause trading",
                "reduce_risk": 0.0  # Reduce to zero (stop trading)
            }
        elif self.current_drawdown > self.max_drawdown * 0.7:
            # Approaching max drawdown, reduce risk significantly
            reduce_factor = 1.0 - (self.current_drawdown / self.max_drawdown)
            return {
                "status": "warning",
                "drawdown": self.current_drawdown,
                "max_allowed": self.max_drawdown,
                "recommendation": "Close most positions and reduce risk significantly",
                "reduce_risk": max(0.2, reduce_factor)  # Reduce to 20% or less
            }
        elif self.current_drawdown > self.max_drawdown * 0.5:
            # In significant drawdown, reduce risk moderately
            reduce_factor = 1.0 - (self.current_drawdown / self.max_drawdown) * 0.7
            return {
                "status": "caution",
                "drawdown": self.current_drawdown,
                "max_allowed": self.max_drawdown,
                "recommendation": "Consider closing some positions and reducing position sizes",
                "reduce_risk": reduce_factor  # Linear reduction based on drawdown
            }
        else:
            # Normal operation
            return {
                "status": "normal",
                "drawdown": self.current_drawdown,
                "max_allowed": self.max_drawdown,
                "recommendation": "No action needed",
                "reduce_risk": 1.0  # No reduction
            }
    
    def check_total_exposure(self):
        """
        Check total portfolio exposure against risk limits.
        
        Returns:
            dict: Exposure status and recommendations
        """
        # Calculate total exposure
        total_exposure = sum(
            position.get('position_size', 0)
            for position in self.current_positions.values()
        )
        
        # Calculate exposure as percentage of portfolio
        exposure_percent = total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Get maximum allowed exposure
        max_exposure = self.risk_params.get("max_portfolio_exposure_percent", 100.0) / 100
        
        # Check exposure
        if exposure_percent > max_exposure:
            return {
                "status": "exceeded",
                "exposure": exposure_percent,
                "max_allowed": max_exposure,
                "recommendation": "Reduce overall exposure by closing some positions",
                "excess_amount": total_exposure - (self.portfolio_value * max_exposure)
            }
        elif exposure_percent > max_exposure * 0.9:
            return {
                "status": "approaching",
                "exposure": exposure_percent,
                "max_allowed": max_exposure,
                "recommendation": "Approaching maximum exposure, be cautious with new positions",
                "remaining_capacity": (self.portfolio_value * max_exposure) - total_exposure
            }
        else:
            return {
                "status": "normal",
                "exposure": exposure_percent,
                "max_allowed": max_exposure,
                "recommendation": "Exposure within limits",
                "remaining_capacity": (self.portfolio_value * max_exposure) - total_exposure
            }
    
    def adjust_risk_for_market_conditions(self, market_regime=None, volatility_regime=None):
        """
        Adjust risk parameters based on market conditions.
        
        Args:
            market_regime (str, optional): Current market regime
            volatility_regime (str, optional): Current volatility regime
            
        Returns:
            dict: Adjusted risk parameters
        """
        # Start with default parameters
        adjusted_params = {
            "max_risk_per_trade": self.risk_params.get("max_risk_per_trade_percent", 1.0),
            "max_position_size": self.max_position_size * 100,  # Convert to percentage
            "position_sizing_atr_multiplier": self.position_sizing_atr_multiplier
        }
        
        # Adjust for market regime
        if market_regime:
            if market_regime == "bullish":
                # More aggressive in bullish markets
                adjusted_params["max_risk_per_trade"] *= 1.2
                adjusted_params["max_position_size"] *= 1.2
                adjusted_params["position_sizing_atr_multiplier"] *= 1.2
            elif market_regime == "bearish":
                # More conservative in bearish markets
                adjusted_params["max_risk_per_trade"] *= 0.8
                adjusted_params["max_position_size"] *= 0.8
                adjusted_params["position_sizing_atr_multiplier"] *= 0.8
            elif market_regime == "volatile":
                # Much more conservative in volatile markets
                adjusted_params["max_risk_per_trade"] *= 0.6
                adjusted_params["max_position_size"] *= 0.6
                adjusted_params["position_sizing_atr_multiplier"] *= 0.7
        
        # Further adjust for volatility regime
        if volatility_regime:
            if volatility_regime == "high":
                # More conservative in high volatility
                adjusted_params["max_risk_per_trade"] *= 0.8
                adjusted_params["max_position_size"] *= 0.8
            elif volatility_regime == "low":
                # More aggressive in low volatility
                adjusted_params["max_risk_per_trade"] *= 1.2
                adjusted_params["max_position_size"] *= 1.2
        
        # Apply drawdown adjustments
        drawdown_protection = self.check_drawdown_protection()
        reduce_factor = drawdown_protection.get("reduce_risk", 1.0)
        
        adjusted_params["max_risk_per_trade"] *= reduce_factor
        adjusted_params["max_position_size"] *= reduce_factor
        
        # Log adjustments
        self.logger.info(
            f"Adjusted risk parameters for {market_regime or 'unknown'} market, "
            f"{volatility_regime or 'unknown'} volatility"
        )
        
        return adjusted_params
    
    def log_risk_metrics(self):
        """
        Log current risk metrics for monitoring and analysis.
        
        Returns:
            dict: Current risk metrics
        """
        # Calculate current metrics
        total_exposure = sum(
            position.get('position_size', 0)
            for position in self.current_positions.values()
        )
        
        exposure_percent = total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Calculate exposure by option type
        call_exposure = sum(
            position.get('position_size', 0)
            for position in self.current_positions.values()
            if position.get('option_type') == 'call'
        )
        
        put_exposure = sum(
            position.get('position_size', 0)
            for position in self.current_positions.values()
            if position.get('option_type') == 'put'
        )
        
        # Prepare metrics
        metrics = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "portfolio_value": self.portfolio_value,
            "total_exposure": total_exposure,
            "exposure_percent": exposure_percent,
            "call_exposure": call_exposure,
            "put_exposure": put_exposure,
            "call_put_ratio": call_exposure / put_exposure if put_exposure > 0 else float('inf'),
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "active_positions": len(self.current_positions),
            "sector_exposure": self.sector_exposure
        }
        
        # Log metrics
        log_file = os.path.join(self.logs_dir, f"risk_metrics_{datetime.now().strftime('%Y%m%d')}.json")
        
        try:
            # Load existing logs if available
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = []
            
            # Add new metrics
            log_data.append(metrics)
            
            # Save log file
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging risk metrics: {str(e)}")
        
        return metrics
    
    def get_risk_summary(self):
        """
        Get a summary of current risk status.
        
        Returns:
            dict: Risk summary
        """
        # Check various risk conditions
        exposure_status = self.check_total_exposure()
        drawdown_status = self.check_drawdown_protection()
        correlated_positions = self.check_correlated_positions()
        overnight_risk = self.check_overnight_risk()
        
        # Determine overall risk status
        if drawdown_status["status"] == "critical" or exposure_status["status"] == "exceeded":
            overall_status = "critical"
        elif drawdown_status["status"] == "warning" or exposure_status["status"] == "approaching":
            overall_status = "warning"
        elif drawdown_status["status"] == "caution" or len(correlated_positions) > 0:
            overall_status = "caution"
        else:
            overall_status = "normal"
        
        # Create summary
        summary = {
            "overall_status": overall_status,
            "exposure": exposure_status,
            "drawdown": drawdown_status,
            "correlated_positions": correlated_positions,
            "overnight_risk": overnight_risk,
            "active_positions": len(self.current_positions),
            "portfolio_value": self.portfolio_value,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Log summary if not normal
        if overall_status != "normal":
            self.logger.warning(f"Risk status: {overall_status} - {summary}")
        
        return summary
