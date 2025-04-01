"""
Hypothesis Generator Module

This module generates trading strategy hypotheses for the Option Hunter system.
It uses data mining, pattern recognition, and evolutionary algorithms to discover
potentially profitable trading patterns and rules.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import itertools
from collections import defaultdict, deque
import time
import multiprocessing
from functools import partial
import pickle
import copy
import re

class TradingRule:
    """
    Representation of a trading rule hypothesis.
    """
    
    def __init__(self, rule_type=None, condition=None, action=None, params=None, rule_id=None):
        """
        Initialize a trading rule.
        
        Args:
            rule_type (str): Type of rule (e.g., 'indicator', 'pattern', 'time')
            condition (str): Rule condition
            action (str): Action to take ('buy', 'sell', 'hold')
            params (dict): Rule parameters
            rule_id (str): Unique rule identifier
        """
        self.rule_id = rule_id or f"rule_{int(time.time())}_{random.randint(1000, 9999)}"
        self.rule_type = rule_type or random.choice(['indicator', 'pattern', 'time', 'fundamental'])
        self.condition = condition
        self.action = action or random.choice(['buy', 'sell', 'hold'])
        self.params = params or {}
        
        # Performance metrics
        self.performance = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_return': 0.0,
            'sharpe_ratio': 0.0,
            'trades': 0,
            'score': 0.0,
            'tested': False
        }
    
    def __str__(self):
        """String representation of the rule."""
        condition_str = self.condition or "Undefined"
        return f"{self.rule_type.capitalize()} Rule {self.rule_id}: IF {condition_str} THEN {self.action.upper()}"
    
    def to_dict(self):
        """Convert rule to dictionary."""
        return {
            'rule_id': self.rule_id,
            'rule_type': self.rule_type,
            'condition': self.condition,
            'action': self.action,
            'params': self.params,
            'performance': self.performance
        }
    
    @classmethod
    def from_dict(cls, rule_dict):
        """Create rule from dictionary."""
        rule = cls(
            rule_type=rule_dict.get('rule_type'),
            condition=rule_dict.get('condition'),
            action=rule_dict.get('action'),
            params=rule_dict.get('params'),
            rule_id=rule_dict.get('rule_id')
        )
        
        # Set performance if available
        if 'performance' in rule_dict:
            rule.performance = rule_dict.get('performance')
        
        return rule


class TradingStrategy:
    """
    Collection of trading rules that form a hypothesis.
    """
    
    def __init__(self, rules=None, strategy_id=None, name=None, description=None):
        """
        Initialize a trading strategy.
        
        Args:
            rules (list): List of TradingRule objects
            strategy_id (str): Unique strategy identifier
            name (str): Strategy name
            description (str): Strategy description
        """
        self.strategy_id = strategy_id or f"strategy_{int(time.time())}_{random.randint(1000, 9999)}"
        self.name = name or f"Strategy {self.strategy_id}"
        self.description = description or f"Generated strategy {self.strategy_id}"
        self.rules = rules or []
        self.creation_date = datetime.now().isoformat()
        
        # Performance metrics
        self.performance = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'trades': 0,
            'score': 0.0,
            'tested': False
        }
    
    def __str__(self):
        """String representation of the strategy."""
        return f"{self.name}: {len(self.rules)} rules, Score: {self.performance.get('score', 0.0):.2f}"
    
    def add_rule(self, rule):
        """Add a rule to the strategy."""
        self.rules.append(rule)
    
    def to_dict(self):
        """Convert strategy to dictionary."""
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'description': self.description,
            'rules': [rule.to_dict() for rule in self.rules],
            'creation_date': self.creation_date,
            'performance': self.performance
        }
    
    @classmethod
    def from_dict(cls, strategy_dict):
        """Create strategy from dictionary."""
        strategy = cls(
            rules=[TradingRule.from_dict(rule) for rule in strategy_dict.get('rules', [])],
            strategy_id=strategy_dict.get('strategy_id'),
            name=strategy_dict.get('name'),
            description=strategy_dict.get('description')
        )
        
        # Set creation date if available
        if 'creation_date' in strategy_dict:
            strategy.creation_date = strategy_dict.get('creation_date')
        
        # Set performance if available
        if 'performance' in strategy_dict:
            strategy.performance = strategy_dict.get('performance')
        
        return strategy
    
    def evaluate(self, data, metrics=None):
        """
        Evaluate the strategy on historical data.
        
        Args:
            data (pd.DataFrame): Historical market data
            metrics (list): Metrics to calculate
            
        Returns:
            dict: Performance metrics
        """
        # Implement strategy evaluation logic
        # This would be implemented by the hypothesis tester component
        pass


class HypothesisGenerator:
    """
    Generator for trading strategy hypotheses based on historical data.
    
    Features:
    - Pattern discovery in market data
    - Rule generation for trading signals
    - Strategy composition from rules
    - Pruning of low-quality hypotheses
    - Diversity maintenance in the hypothesis population
    """
    
    def __init__(self, config=None):
        """
        Initialize the HypothesisGenerator.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Extract configuration
        self.hypothesis_params = self.config.get("hypothesis_generator", {})
        
        # Default parameters
        self.max_hypotheses = self.hypothesis_params.get("max_hypotheses", 100)
        self.min_rule_confidence = self.hypothesis_params.get("min_rule_confidence", 0.6)
        self.indicator_weight = self.hypothesis_params.get("indicator_weight", 0.4)
        self.pattern_weight = self.hypothesis_params.get("pattern_weight", 0.3)
        self.time_weight = self.hypothesis_params.get("time_weight", 0.2)
        self.fundamental_weight = self.hypothesis_params.get("fundamental_weight", 0.1)
        
        # Storage for generated hypotheses
        self.hypotheses = []
        self.rules_library = []
        
        # Data metrics
        self.market_statistics = {}
        
        # Create logs and models directories
        self.logs_dir = "logs/hypotheses"
        self.models_dir = "models/hypotheses"
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.logger.info(f"HypothesisGenerator initialized")
    
    def analyze_data(self, data, symbol=None):
        """
        Analyze historical data to extract patterns and statistics.
        
        Args:
            data (pd.DataFrame): Historical market data
            symbol (str, optional): Symbol identifier
            
        Returns:
            dict: Analysis results
        """
        if data is None or len(data) == 0:
            self.logger.error("No data provided for analysis")
            return None
        
        try:
            self.logger.info(f"Analyzing data for pattern discovery ({len(data)} data points)")
            
            # Make a copy to avoid modifying original data
            df = data.copy()
            analysis_results = {}
            
            # Calculate basic statistics
            if 'option_price' in df.columns:
                price_mean = df['option_price'].mean()
                price_std = df['option_price'].std()
                price_min = df['option_price'].min()
                price_max = df['option_price'].max()
                
                analysis_results['price_statistics'] = {
                    'mean': price_mean,
                    'std': price_std,
                    'min': price_min,
                    'max': price_max,
                    'range': price_max - price_min
                }
                
                # Calculate returns
                df['option_return'] = df['option_price'].pct_change()
                
                return_mean = df['option_return'].mean()
                return_std = df['option_return'].std()
                
                analysis_results['return_statistics'] = {
                    'mean': return_mean,
                    'std': return_std,
                    'sharpe': return_mean / return_std if return_std > 0 else 0,
                    'positive_days': (df['option_return'] > 0).mean()
                }
            
            # Analyze volatility regimes
            if 'option_price' in df.columns:
                # Calculate historical volatility
                for window in [10, 20, 30]:
                    df[f'volatility_{window}d'] = df['option_price'].pct_change().rolling(window).std() * np.sqrt(252)
                
                # Identify volatility regimes
                vol_mean = df['volatility_20d'].mean()
                vol_std = df['volatility_20d'].std()
                
                df['vol_regime'] = 'normal'
                df.loc[df['volatility_20d'] > vol_mean + vol_std, 'vol_regime'] = 'high'
                df.loc[df['volatility_20d'] < vol_mean - vol_std, 'vol_regime'] = 'low'
                
                # Calculate regime statistics
                vol_regimes = df['vol_regime'].value_counts(normalize=True).to_dict()
                
                regime_returns = {}
                for regime in ['high', 'normal', 'low']:
                    if regime in df['vol_regime'].values:
                        regime_returns[regime] = df.loc[df['vol_regime'] == regime, 'option_return'].mean()
                
                analysis_results['volatility_regimes'] = {
                    'distribution': vol_regimes,
                    'returns': regime_returns
                }
            
            # Identify significant price patterns
            if 'option_price' in df.columns:
                # Implement pattern detection logic
                patterns = self._detect_price_patterns(df)
                analysis_results['price_patterns'] = patterns
            
            # Store results
            symbol_key = symbol or 'default'
            self.market_statistics[symbol_key] = analysis_results
            
            self.logger.info(f"Data analysis completed with {len(analysis_results)} metrics")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing data: {str(e)}")
            return None
    
    def _detect_price_patterns(self, df):
        """
        Detect common price patterns in the data.
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            
        Returns:
            dict: Detected patterns
        """
        patterns = {}
        
        try:
            # Check for available indicators
            price_col = 'option_price'
            if price_col not in df.columns:
                return patterns
            
            # Calculate required indicators if not present
            if 'sma_20' not in df.columns:
                df['sma_20'] = df[price_col].rolling(20).mean()
            
            if 'sma_50' not in df.columns:
                df['sma_50'] = df[price_col].rolling(50).mean()
            
            # Detect trend patterns
            uptrend = (df[price_col] > df['sma_20']) & (df['sma_20'] > df['sma_50'])
            downtrend = (df[price_col] < df['sma_20']) & (df['sma_20'] < df['sma_50'])
            
            patterns['uptrend_days'] = uptrend.sum()
            patterns['downtrend_days'] = downtrend.sum()
            
            # Calculate returns following patterns
            if 'option_return' in df.columns:
                uptrend_returns = df.loc[uptrend, 'option_return'].mean()
                downtrend_returns = df.loc[downtrend, 'option_return'].mean()
                
                patterns['uptrend_avg_return'] = uptrend_returns
                patterns['downtrend_avg_return'] = downtrend_returns
            
            # Detect support and resistance levels
            if len(df) > 50:
                # Simple method: look for price levels that have been tested multiple times
                price_bins = pd.cut(df[price_col], bins=20)
                price_levels = price_bins.value_counts()
                
                # Find local maxima in the histogram (resistance)
                resistance_levels = []
                for i in range(1, len(price_levels) - 1):
                    if price_levels.iloc[i] > price_levels.iloc[i-1] and price_levels.iloc[i] > price_levels.iloc[i+1]:
                        resistance_levels.append(price_levels.index[i].mid)
                
                # Find local minima in the histogram (support)
                support_levels = []
                for i in range(1, len(price_levels) - 1):
                    if price_levels.iloc[i] < price_levels.iloc[i-1] and price_levels.iloc[i] < price_levels.iloc[i+1]:
                        support_levels.append(price_levels.index[i].mid)
                
                patterns['resistance_levels'] = resistance_levels
                patterns['support_levels'] = support_levels
            
            # Detect breakouts and breakdowns
            if 'option_return' in df.columns:
                # Define breakout as 2% move above previous high
                df['prev_high_5d'] = df[price_col].rolling(5).max().shift(1)
                breakouts = df[price_col] > df['prev_high_5d'] * 1.02
                
                # Define breakdown as 2% move below previous low
                df['prev_low_5d'] = df[price_col].rolling(5).min().shift(1)
                breakdowns = df[price_col] < df['prev_low_5d'] * 0.98
                
                patterns['breakout_days'] = breakouts.sum()
                patterns['breakdown_days'] = breakdowns.sum()
                
                # Calculate returns following patterns
                breakout_returns = df.loc[breakouts, 'option_return'].mean()
                breakdown_returns = df.loc[breakdowns, 'option_return'].mean()
                
                patterns['breakout_avg_return'] = breakout_returns
                patterns['breakdown_avg_return'] = breakdown_returns
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting price patterns: {str(e)}")
            return patterns
    
    def _generate_indicator_rules(self, data, num_rules=10):
        """
        Generate rules based on technical indicators.
        
        Args:
            data (pd.DataFrame): Historical market data
            num_rules (int): Number of rules to generate
            
        Returns:
            list: Generated trading rules
        """
        rules = []
        
        try:
            # Make sure we have price data
            if 'option_price' not in data.columns:
                return rules
            
            # Calculate common indicators if not already present
            df = data.copy()
            
            # Simple Moving Averages
            for window in [5, 10, 20, 50]:
                if f'sma_{window}' not in df.columns:
                    df[f'sma_{window}'] = df['option_price'].rolling(window).mean()
            
            # Exponential Moving Averages
            for window in [9, 21]:
                if f'ema_{window}' not in df.columns:
                    df[f'ema_{window}'] = df['option_price'].ewm(span=window, adjust=False).mean()
            
            # Relative Strength Index
            if 'rsi_14' not in df.columns:
                delta = df['option_price'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss.replace(0, 1e-9)  # Avoid division by zero
                df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            if 'macd' not in df.columns:
                ema_12 = df['option_price'].ewm(span=12, adjust=False).mean()
                ema_26 = df['option_price'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema_12 - ema_26
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Available rule templates
            rule_templates = [
                # SMA crossovers
                {'condition': '{price} > {sma}', 'action': 'buy', 'params': {'sma': [f'sma_{w}' for w in [5, 10, 20, 50]]}},
                {'condition': '{price} < {sma}', 'action': 'sell', 'params': {'sma': [f'sma_{w}' for w in [5, 10, 20, 50]]}},
                
                # EMA crossovers
                {'condition': '{ema_fast} > {ema_slow}', 'action': 'buy', 'params': {'ema_fast': ['ema_9'], 'ema_slow': ['ema_21']}},
                {'condition': '{ema_fast} < {ema_slow}', 'action': 'sell', 'params': {'ema_fast': ['ema_9'], 'ema_slow': ['ema_21']}},
                
                # RSI conditions
                {'condition': '{rsi} < 30', 'action': 'buy', 'params': {'rsi': ['rsi_14']}},
                {'condition': '{rsi} > 70', 'action': 'sell', 'params': {'rsi': ['rsi_14']}},
                
                # MACD conditions
                {'condition': '{macd} > {macd_signal}', 'action': 'buy', 'params': {'macd': ['macd'], 'macd_signal': ['macd_signal']}},
                {'condition': '{macd} < {macd_signal}', 'action': 'sell', 'params': {'macd': ['macd'], 'macd_signal': ['macd_signal']}},
                
                # Price vs Moving Average
                {'condition': '{price} > {sma} * 1.05', 'action': 'sell', 'params': {'sma': [f'sma_{w}' for w in [20, 50]]}},
                {'condition': '{price} < {sma} * 0.95', 'action': 'buy', 'params': {'sma': [f'sma_{w}' for w in [20, 50]]}}
            ]
            
            # Generate rules by instantiating templates
            for i in range(num_rules):
                # Randomly select a rule template
                template = random.choice(rule_templates)
                
                # Instantiate parameters
                params = {}
                condition = template['condition']
                
                # Replace placeholders with actual indicators
                for param_name, param_options in template['params'].items():
                    param_value = random.choice(param_options)
                    params[param_name] = param_value
                    condition = condition.replace(f'{{{param_name}}}', param_value)
                
                # Replace {price} with 'option_price'
                condition = condition.replace('{price}', 'option_price')
                
                # Create rule
                rule = TradingRule(
                    rule_type='indicator',
                    condition=condition,
                    action=template['action'],
                    params=params
                )
                
                rules.append(rule)
            
            self.logger.info(f"Generated {len(rules)} indicator-based rules")
            
            return rules
            
        except Exception as e:
            self.logger.error(f"Error generating indicator rules: {str(e)}")
            return rules
    
    def _generate_pattern_rules(self, data, num_rules=10):
        """
        Generate rules based on price patterns.
        
        Args:
            data (pd.DataFrame): Historical market data
            num_rules (int): Number of rules to generate
            
        Returns:
            list: Generated trading rules
        """
        rules = []
        
        try:
            # Make sure we have price data
            if 'option_price' not in data.columns:
                return rules
            
            # Calculate pattern indicators if not present
            df = data.copy()
            
            # Previous highs and lows
            for window in [5, 10, 20]:
                if f'prev_high_{window}d' not in df.columns:
                    df[f'prev_high_{window}d'] = df['option_price'].rolling(window).max().shift(1)
                
                if f'prev_low_{window}d' not in df.columns:
                    df[f'prev_low_{window}d'] = df['option_price'].rolling(window).min().shift(1)
            
            # Consecutive days in same direction
            if 'consec_up' not in df.columns:
                df['price_change'] = df['option_price'].diff()
                df['consec_up'] = 0
                df['consec_down'] = 0
                
                # Calculate consecutive up/down days
                for i in range(1, len(df)):
                    if df['price_change'].iloc[i] > 0:
                        df['consec_up'].iloc[i] = df['consec_up'].iloc[i-1] + 1
                        df['consec_down'].iloc[i] = 0
                    elif df['price_change'].iloc[i] < 0:
                        df['consec_down'].iloc[i] = df['consec_down'].iloc[i-1] + 1
                        df['consec_up'].iloc[i] = 0
                    else:
                        df['consec_up'].iloc[i] = 0
                        df['consec_down'].iloc[i] = 0
            
            # Available rule templates
            rule_templates = [
                # Breakouts
                {'condition': 'option_price > {prev_high} * 1.02', 'action': 'buy', 'params': {'prev_high': [f'prev_high_{w}d' for w in [5, 10, 20]]}},
                {'condition': 'option_price < {prev_low} * 0.98', 'action': 'sell', 'params': {'prev_low': [f'prev_low_{w}d' for w in [5, 10, 20]]}},
                
                # Consecutive days
                {'condition': 'consec_up >= {num_days}', 'action': 'sell', 'params': {'num_days': [3, 4, 5]}},
                {'condition': 'consec_down >= {num_days}', 'action': 'buy', 'params': {'num_days': [3, 4, 5]}},
                
                # Price channels
                {'condition': 'option_price > prev_high_20d AND option_price < prev_high_20d * 1.05', 'action': 'buy', 'params': {}},
                {'condition': 'option_price < prev_low_20d AND option_price > prev_low_20d * 0.95', 'action': 'sell', 'params': {}},
                
                # Gap ups/downs
                {'condition': 'option_price > option_price.shift(1) * 1.03', 'action': 'buy', 'params': {}},
                {'condition': 'option_price < option_price.shift(1) * 0.97', 'action': 'sell', 'params': {}},
                
                # Mean reversion
                {'condition': '(option_price / sma_20) < 0.9', 'action': 'buy', 'params': {}},
                {'condition': '(option_price / sma_20) > 1.1', 'action': 'sell', 'params': {}}
            ]
            
            # Generate rules by instantiating templates
            for i in range(num_rules):
                # Randomly select a rule template
                template = random.choice(rule_templates)
                
                # Instantiate parameters
                params = {}
                condition = template['condition']
                
                # Replace placeholders with actual values
                for param_name, param_options in template['params'].items():
                    if param_options:  # If there are options to choose from
                        param_value = random.choice(param_options)
                        params[param_name] = param_value
                        condition = condition.replace(f'{{{param_name}}}', str(param_value))
                
                # Create rule
                rule = TradingRule(
                    rule_type='pattern',
                    condition=condition,
                    action=template['action'],
                    params=params
                )
                
                rules.append(rule)
            
            self.logger.info(f"Generated {len(rules)} pattern-based rules")
            
            return rules
            
        except Exception as e:
            self.logger.error(f"Error generating pattern rules: {str(e)}")
            return rules
    
    def _generate_time_rules(self, data, num_rules=5):
        """
        Generate rules based on time patterns.
        
        Args:
            data (pd.DataFrame): Historical market data
            num_rules (int): Number of rules to generate
            
        Returns:
            list: Generated trading rules
        """
        rules = []
        
        try:
            # Make sure we have date data
            if 'date' not in data.columns:
                return rules
            
            # Extract time features if not present
            df = data.copy()
            df['date'] = pd.to_datetime(df['date'])
            
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df['date'].dt.dayofweek
            
            if 'month' not in df.columns:
                df['month'] = df['date'].dt.month
            
            if 'day_of_month' not in df.columns:
                df['day_of_month'] = df['date'].dt.day
            
            # Calculate returns by time period if not present
            if 'option_return' not in df.columns and 'option_price' in df.columns:
                df['option_return'] = df['option_price'].pct_change()
            
            # Available rule templates
            rule_templates = [
                # Day of week
                {'condition': 'day_of_week == {day}', 'action': '{action}', 'params': {'day': [0, 1, 2, 3, 4], 'action': ['buy', 'sell']}},
                
                # Month
                {'condition': 'month == {month}', 'action': '{action}', 'params': {'month': list(range(1, 13)), 'action': ['buy', 'sell']}},
                
                # Start/end of month
                {'condition': 'day_of_month <= 3', 'action': 'buy', 'params': {}},
                {'condition': 'day_of_month >= 28', 'action': 'sell', 'params': {}},
                
                # Days to expiration (if available)
                {'condition': 'days_to_expiration <= 5', 'action': 'sell', 'params': {}}
            ]
            
            # Analyze returns by time period to improve rule generation
            if 'option_return' in df.columns:
                # Returns by day of week
                day_returns = df.groupby('day_of_week')['option_return'].mean()
                best_day = day_returns.idxmax()
                worst_day = day_returns.idxmin()
                
                # Returns by month
                month_returns = df.groupby('month')['option_return'].mean()
                best_month = month_returns.idxmax()
                worst_month = month_returns.idxmin()
                
                # Add informed rules
                rule_templates.extend([
                    {'condition': f'day_of_week == {best_day}', 'action': 'buy', 'params': {}},
                    {'condition': f'day_of_week == {worst_day}', 'action': 'sell', 'params': {}},
                    {'condition': f'month == {best_month}', 'action': 'buy', 'params': {}},
                    {'condition': f'month == {worst_month}', 'action': 'sell', 'params': {}}
                ])
            
            # Generate rules by instantiating templates
            for i in range(num_rules):
                # Randomly select a rule template
                template = random.choice(rule_templates)
                
                # Instantiate parameters
                params = {}
                condition = template['condition']
                action = template['action']
                
                # Replace placeholders with actual values
                for param_name, param_options in template['params'].items():
                    if param_options:  # If there are options to choose from
                        param_value = random.choice(param_options)
                        params[param_name] = param_value
                        
                        if param_name == 'action':
                            action = param_value
                        else:
                            condition = condition.replace(f'{{{param_name}}}', str(param_value))
                
                # Create rule
                rule = TradingRule(
                    rule_type='time',
                    condition=condition,
                    action=action,
                    params=params
                )
                
                rules.append(rule)
            
            self.logger.info(f"Generated {len(rules)} time-based rules")
            
            return rules
            
        except Exception as e:
            self.logger.error(f"Error generating time rules: {str(e)}")
            return rules
    
    def _generate_fundamental_rules(self, data, num_rules=5):
        """
        Generate rules based on fundamental data.
        
        Args:
            data (pd.DataFrame): Historical market data
            num_rules (int): Number of rules to generate
            
        Returns:
            list: Generated trading rules
        """
        rules = []
        
        try:
            # Check if we have fundamental data
            fundamental_columns = [
                'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 
                'open_interest', 'volume', 'bid_ask_spread'
            ]
            
            available_columns = [col for col in fundamental_columns if col in data.columns]
            
            if not available_columns:
                return rules
            
            # Available rule templates
            rule_templates = []
            
            # Add templates for each available column
            if 'implied_volatility' in available_columns:
                rule_templates.extend([
                    {'condition': 'implied_volatility > {iv_threshold}', 'action': 'sell', 'params': {'iv_threshold': [0.3, 0.4, 0.5]}},
                    {'condition': 'implied_volatility < {iv_threshold}', 'action': 'buy', 'params': {'iv_threshold': [0.15, 0.2, 0.25]}}
                ])
            
            if 'delta' in available_columns:
                rule_templates.extend([
                    {'condition': 'delta > {delta_threshold}', 'action': 'buy', 'params': {'delta_threshold': [0.6, 0.7, 0.8]}},
                    {'condition': 'delta < {delta_threshold}', 'action': 'sell', 'params': {'delta_threshold': [0.3, 0.4, 0.5]}}
                ])
            
            if 'open_interest' in available_columns and 'volume' in available_columns:
                rule_templates.extend([
                    {'condition': 'volume / open_interest > {oi_ratio}', 'action': 'buy', 'params': {'oi_ratio': [0.3, 0.5, 0.7]}},
                    {'condition': 'volume > {vol_threshold} * volume.rolling(5).mean()', 'action': 'buy', 'params': {'vol_threshold': [2, 3, 4]}}
                ])
            
            if 'bid_ask_spread' in available_columns:
                rule_templates.extend([
                    {'condition': 'bid_ask_spread < {spread_threshold}', 'action': 'buy', 'params': {'spread_threshold': [0.05, 0.1, 0.15]}}
                ])
            
            # Generate rules by instantiating templates
            for i in range(min(num_rules, len(rule_templates))):
                # Randomly select a rule template
                template = random.choice(rule_templates)
                
                # Instantiate parameters
                params = {}
                condition = template['condition']
                
                # Replace placeholders with actual values
                for param_name, param_options in template['params'].items():
                    if param_options:  # If there are options to choose from
                        param_value = random.choice(param_options)
                        params[param_name] = param_value
                        condition = condition.replace(f'{{{param_name}}}', str(param_value))
                
                # Create rule
                rule = TradingRule(
                    rule_type='fundamental',
                    condition=condition,
                    action=template['action'],
                    params=params
                )
                
                rules.append(rule)
            
            self.logger.info(f"Generated {len(rules)} fundamental-based rules")
            
            return rules
            
        except Exception as e:
            self.logger.error(f"Error generating fundamental rules: {str(e)}")
            return rules
