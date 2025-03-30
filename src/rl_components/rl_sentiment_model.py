"""
RL Sentiment Model Module

This module implements reinforcement learning models for analyzing market sentiment data
to predict market movements and identify trading opportunities based on sentiment patterns.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import gym
from gym import spaces
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Try to import SHAP - an optional dependency
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Import for RL algorithms
try:
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO, A2C, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


class SentimentEnv(gym.Env):
    """
    Custom gym environment for market sentiment data.
    
    This environment allows the agent to learn from sentiment data
    to predict market movements and make trading decisions.
    """
    
    def __init__(self, sentiment_data, price_data, lookback=5, window_size=60):
        """
        Initialize the environment.
        
        Args:
            sentiment_data (pd.DataFrame): Market sentiment data
            price_data (pd.DataFrame): Price data for the same time period
            lookback (int): Number of time periods to include in the state
            window_size (int): Size of the rolling window for feature calculation
        """
        super(SentimentEnv, self).__init__()
        
        # Store data
        self.sentiment_data = sentiment_data
        self.price_data = price_data
        self.lookback = lookback
        self.window_size = window_size
        
        # Ensure data is aligned and sorted
        self._align_data()
        
        # Define action space: 0 = no position, 1 = long, 2 = short
        self.action_space = spaces.Discrete(3)
        
        # Define observation space (state)
        # Features include sentiment metrics and price data
        self.n_features = self._get_feature_count()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.lookback, self.n_features),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def _align_data(self):
        """
        Align sentiment and price data by date/time.
        """
        # Ensure we have datetime index
        if not isinstance(self.sentiment_data.index, pd.DatetimeIndex):
            if 'date' in self.sentiment_data.columns:
                self.sentiment_data['date'] = pd.to_datetime(self.sentiment_data['date'])
                self.sentiment_data.set_index('date', inplace=True)
            elif 'timestamp' in self.sentiment_data.columns:
                self.sentiment_data['timestamp'] = pd.to_datetime(self.sentiment_data['timestamp'])
                self.sentiment_data.set_index('timestamp', inplace=True)
        
        if not isinstance(self.price_data.index, pd.DatetimeIndex):
            if 'date' in self.price_data.columns:
                self.price_data['date'] = pd.to_datetime(self.price_data['date'])
                self.price_data.set_index('date', inplace=True)
            elif 'timestamp' in self.price_data.columns:
                self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])
                self.price_data.set_index('timestamp', inplace=True)
        
        # Sort by index
        self.sentiment_data = self.sentiment_data.sort_index()
        self.price_data = self.price_data.sort_index()
        
        # Align indices by finding common dates
        common_dates = self.sentiment_data.index.intersection(self.price_data.index)
        self.sentiment_data = self.sentiment_data.loc[common_dates]
        self.price_data = self.price_data.loc[common_dates]
    
    def _get_feature_count(self):
        """
        Determine the number of features in the processed data.
        
        Returns:
            int: Number of features
        """
        # Process a single window to determine feature count
        sample_features = self._process_window(0)
        return sample_features.shape[1]
    
    def _process_window(self, start_idx):
        """
        Process a window of sentiment and price data to extract features.
        
        Args:
            start_idx (int): Starting index of the window
            
        Returns:
            np.ndarray: Extracted features
        """
        # Get window of data
        end_idx = min(start_idx + self.window_size, len(self.sentiment_data))
        sentiment_window = self.sentiment_data.iloc[start_idx:end_idx]
        price_window = self.price_data.iloc[start_idx:end_idx]
        
        # Extract sentiment features
        sentiment_features = self._extract_sentiment_features(sentiment_window)
        
        # Extract price features
        price_features = self._extract_price_features(price_window)
        
        # Combine features
        features = np.concatenate([sentiment_features, price_features], axis=1)
        
        return features
    
    def _extract_sentiment_features(self, sentiment_window):
        """
        Extract features from sentiment data.
        
        Args:
            sentiment_window (pd.DataFrame): Window of sentiment data
            
        Returns:
            np.ndarray: Sentiment features
        """
        features = []
        
        # Check which sentiment columns are available
        sentiment_columns = ['sentiment_score', 'positive_score', 'negative_score', 'neutral_score',
                           'bullish_ratio', 'bearish_ratio', 'sentiment_volume']
        
        available_columns = [col for col in sentiment_columns if col in sentiment_window.columns]
        
        if not available_columns:
            # If no direct sentiment columns, look for a text column to analyze
            text_columns = ['text', 'content', 'message']
            text_column = next((col for col in text_columns if col in sentiment_window.columns), None)
            
            if text_column:
                # Extract sentiment from text
                sentiment_scores = self._analyze_text_sentiment(sentiment_window[text_column])
                
                # Add computed sentiment metrics
                features.append(sentiment_scores['sentiment_score'].mean())
                features.append(sentiment_scores['positive_score'].mean())
                features.append(sentiment_scores['negative_score'].mean())
                features.append(sentiment_scores['neutral_score'].mean())
                
                # Calculate bullish/bearish ratio
                positive_count = (sentiment_scores['sentiment_score'] > 0.2).sum()
                negative_count = (sentiment_scores['sentiment_score'] < -0.2).sum()
                total_count = len(sentiment_scores)
                
                if total_count > 0:
                    bullish_ratio = positive_count / total_count
                    bearish_ratio = negative_count / total_count
                else:
                    bullish_ratio = 0.5
                    bearish_ratio = 0.5
                
                features.append(bullish_ratio)
                features.append(bearish_ratio)
                features.append(total_count)
                features.append(bullish_ratio)
                features.append(bearish_ratio)
                features.append(total_count)
            else:
                # No text available, use default values
                features.extend([0, 0, 0, 0, 0.5, 0.5, 0])
        else:
            # Use available sentiment metrics
            for col in sentiment_columns:
                if col in sentiment_window.columns:
                    features.append(sentiment_window[col].mean())
                else:
                    # Use default value for missing columns
                    if col in ['bullish_ratio', 'bearish_ratio']:
                        features.append(0.5)  # Default to balanced
                    else:
                        features.append(0)  # Default to neutral/zero
        
        # Calculate additional metrics
        if len(available_columns) > 0:
            # Sentiment trend (increasing or decreasing)
            if len(sentiment_window) > 1 and 'sentiment_score' in sentiment_window.columns:
                trend = sentiment_window['sentiment_score'].iloc[-1] - sentiment_window['sentiment_score'].iloc[0]
                features.append(trend)
            else:
                features.append(0)
            
            # Sentiment volatility
            if 'sentiment_score' in sentiment_window.columns:
                volatility = sentiment_window['sentiment_score'].std()
                features.append(volatility)
            else:
                features.append(0)
        else:
            features.extend([0, 0])  # No trend or volatility
        
        return np.array(features).reshape(1, -1)
    
    def _analyze_text_sentiment(self, texts):
        """
        Analyze sentiment from text data.
        
        Args:
            texts (pd.Series): Series of text messages
            
        Returns:
            pd.DataFrame: Sentiment scores
        """
        try:
            # Use NLTK for basic sentiment analysis
            # Initialize NLTK resources if not already done
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
        except:
            # If NLTK not available, return default values
            return pd.DataFrame({
                'sentiment_score': [0] * len(texts),
                'positive_score': [0] * len(texts),
                'negative_score': [0] * len(texts),
                'neutral_score': [1] * len(texts)
            })
        
        # Define sentiment words
        positive_words = set([
            'bullish', 'buy', 'uptrend', 'growth', 'positive', 'gain', 'profit', 'strong',
            'optimistic', 'confident', 'good', 'rally', 'boom', 'recovery', 'rebound',
            'upside', 'up', 'rising', 'higher', 'increase', 'outperform', 'beat', 'winning'
        ])
        
        negative_words = set([
            'bearish', 'sell', 'downtrend', 'decline', 'negative', 'loss', 'weak', 'pessimistic',
            'worried', 'bad', 'crash', 'bust', 'recession', 'downside', 'down', 'falling',
            'lower', 'decrease', 'underperform', 'miss', 'losing', 'bear', 'short'
        ])
        
        # Process each text
        sentiment_scores = []
        stop_words = set(stopwords.words('english'))
        
        for text in texts:
            if not isinstance(text, str):
                sentiment_scores.append({
                    'sentiment_score': 0,
                    'positive_score': 0,
                    'negative_score': 0,
                    'neutral_score': 1
                })
                continue
            
            # Tokenize and clean
            tokens = word_tokenize(text.lower())
            tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
            
            # Count sentiment words
            positive_count = sum(1 for word in tokens if word in positive_words)
            negative_count = sum(1 for word in tokens if word in negative_words)
            
            # Calculate scores
            total_words = max(1, len(tokens))  # Avoid division by zero
            positive_score = positive_count / total_words
            negative_score = negative_count / total_words
            neutral_score = 1 - positive_score - negative_score
            sentiment_score = positive_score - negative_score
            
            sentiment_scores.append({
                'sentiment_score': sentiment_score,
                'positive_score': positive_score,
                'negative_score': negative_score,
                'neutral_score': neutral_score
            })
        
        return pd.DataFrame(sentiment_scores)
    
    def _extract_price_features(self, price_window):
        """
        Extract features from price data.
        
        Args:
            price_window (pd.DataFrame): Window of price data
            
        Returns:
            np.ndarray: Price features
        """
        features = []
        
        # Check if we have OHLC data or just close
        price_columns = ['close', 'open', 'high', 'low', 'volume']
        available_columns = [col for col in price_columns if col in price_window.columns]
        
        if not available_columns:
            # No price data, use zeros
            features.extend([0, 0, 0, 0, 0])
            return np.array(features).reshape(1, -1)
        
        # Get close prices (or equivalent)
        if 'close' in price_window.columns:
            prices = price_window['close']
        elif 'price' in price_window.columns:
            prices = price_window['price']
        elif len(price_window.columns) > 0:
            prices = price_window.iloc[:, 0]  # Use first column
        else:
            # No usable price data
            features.extend([0, 0, 0, 0, 0])
            return np.array(features).reshape(1, -1)
        
        # Calculate price metrics
        if len(prices) > 0:
            # Price change
            if len(prices) > 1:
                price_change = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            else:
                price_change = 0
            features.append(price_change)
            
            # Price volatility
            price_volatility = prices.pct_change().std()
            features.append(price_volatility)
            
            # Normalized price
            if prices.iloc[0] != 0:  # Avoid division by zero
                normalized_price = prices.iloc[-1] / prices.iloc[0]
            else:
                normalized_price = 1
            features.append(normalized_price)
        else:
            features.extend([0, 0, 1])  # No change, volatility or normalization
        
        # Volume features
        if 'volume' in price_window.columns:
            # Normalized volume
            mean_volume = price_window['volume'].mean()
            last_volume = price_window['volume'].iloc[-1]
            if mean_volume != 0:  # Avoid division by zero
                volume_ratio = last_volume / mean_volume
            else:
                volume_ratio = 1
            features.append(volume_ratio)
            
            # Volume trend
            if len(price_window) > 1:
                volume_trend = price_window['volume'].iloc[-1] - price_window['volume'].iloc[0]
                if price_window['volume'].iloc[0] != 0:  # Avoid division by zero
                    volume_trend = volume_trend / price_window['volume'].iloc[0]
            else:
                volume_trend = 0
            features.append(volume_trend)
        else:
            features.extend([1, 0])  # No volume data
        
        return np.array(features).reshape(1, -1)
    
    def _process_data(self):
        """
        Process all sentiment and price data to create features.
        
        Returns:
            np.ndarray: Processed features
        """
        # Calculate features for each window
        feature_list = []
        
        for i in range(len(self.sentiment_data) - self.window_size + 1):
            features = self._process_window(i)
            feature_list.append(features)
        
        # Concatenate all features
        return np.vstack(feature_list)
    
    def reset(self):
        """
        Reset the environment to start a new episode.
        
        Returns:
            np.ndarray: Initial state
        """
        # Process data
        self.features = self._process_data()
        
        # Initialize position
        self.current_step = 0
        self.current_position = 0  # 0 = no position, 1 = long, 2 = short
        self.entry_price = None
        self.rewards = []
        
        # Create initial state (lookback window of features)
        self.state = self.features[0:self.lookback]
        
        return self.state
    
    def step(self, action):
        """
        Take one step in the environment.
        
        Args:
            action (int): Action to take (0 = no position, 1 = long, 2 = short)
            
        Returns:
            tuple: Next state, reward, done, info
        """
        # Get current price from price data
        current_idx = self.current_step + self.window_size
        current_price = self._get_price(current_idx)
        
        # Update position
        prev_position = self.current_position
        self.current_position = action
        
        # Calculate reward
        reward = self._calculate_reward(prev_position, action, current_idx)
        self.rewards.append(reward)
        
        # Update step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.features) - self.lookback
        
        # Get new state
        if not done:
            state_idx = min(self.current_step, len(self.features) - self.lookback)
            self.state = self.features[state_idx:state_idx+self.lookback]
        
        # Additional info
        info = {
            'current_step': self.current_step,
            'current_position': self.current_position,
            'cumulative_reward': sum(self.rewards)
        }
        
        return self.state, reward, done, info
    
    def _get_price(self, idx):
        """
        Get price at a specific index.
        
        Args:
            idx (int): Index in the data
            
        Returns:
            float: Price at the index
        """
        if idx >= len(self.price_data):
            # Use last available price
            price_idx = len(self.price_data) - 1
        else:
            price_idx = idx
        
        # Get price from close or equivalent column
        if 'close' in self.price_data.columns:
            return self.price_data['close'].iloc[price_idx]
        elif 'price' in self.price_data.columns:
            return self.price_data['price'].iloc[price_idx]
        elif len(self.price_data.columns) > 0:
            return self.price_data.iloc[price_idx, 0]  # Use first column
        else:
            return 1.0  # Default
    
    def _calculate_reward(self, prev_position, action, current_idx):
        """
        Calculate reward based on position and price changes.
        
        Args:
            prev_position (int): Previous position
            action (int): Current action
            current_idx (int): Current index in the data
            
        Returns:
            float: Reward
        """
        # Get current and next prices
        current_price = self._get_price(current_idx)
        next_idx = min(current_idx + 1, len(self.price_data) - 1)
        next_price = self._get_price(next_idx)
        
        # Calculate price change
        price_change = (next_price - current_price) / current_price
        
        # Calculate reward based on position and price change
        position_reward = 0
        
        if action == 1:  # Long position
            position_reward = price_change * 100  # Scale for more meaningful rewards
        elif action == 2:  # Short position
            position_reward = -price_change * 100
        
        # Add transaction cost for changing positions
        transaction_cost = 0
        if prev_position != action:
            transaction_cost = 0.1  # Small penalty for changing positions
        
        # Final reward
        reward = position_reward - transaction_cost
        
        return reward
    
    def render(self, mode='human'):
        """
        Render the environment (not implemented).
        
        Args:
            mode (str): Rendering mode
            
        Returns:
            None
