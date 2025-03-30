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
        Render the environment.
        
        Args:
            mode (str): Rendering mode
            
        Returns:
            None
        """
        # This method is not fully implemented as it's primarily intended
        # for visualization purposes during development and debugging
        if mode == 'human':
            print(f"Step: {self.current_step}, Position: {self.current_position}, "
                  f"Reward: {self.rewards[-1] if self.rewards else 0}, "
                  f"Total Reward: {sum(self.rewards)}")
        return None


class RLSentimentModel:
    """
    Reinforcement learning model for market sentiment analysis and trading.
    
    This model uses the SentimentEnv environment to train various RL algorithms
    to make trading decisions based on sentiment data.
    """
    
    def __init__(self, config=None):
        """
        Initialize the RLSentimentModel.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.models_dir = "models/rl_sentiment"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Default parameters
        self.algorithm = self.config.get('rl_algorithm', 'ppo')
        self.lookback = self.config.get('lookback_window', 5)
        self.window_size = self.config.get('feature_window_size', 60)
        self.total_timesteps = self.config.get('total_timesteps', 100000)
        
        # Initialize model
        self.model = None
        self.env = None
        
        self.logger.info(f"RLSentimentModel initialized with {self.algorithm} algorithm")
    
    def create_environment(self, sentiment_data, price_data):
        """
        Create the reinforcement learning environment.
        
        Args:
            sentiment_data (pd.DataFrame): Market sentiment data
            price_data (pd.DataFrame): Price data for the same time period
            
        Returns:
            gym.Env: Sentiment trading environment
        """
        # Create environment
        env = SentimentEnv(
            sentiment_data=sentiment_data, 
            price_data=price_data,
            lookback=self.lookback,
            window_size=self.window_size
        )
        
        # Wrap in DummyVecEnv for Stable Baselines compatibility
        if HAS_SB3:
            env = DummyVecEnv([lambda: env])
        
        return env
    
    def build_model(self, env, algorithm=None):
        """
        Build the reinforcement learning model.
        
        Args:
            env (gym.Env): Trading environment
            algorithm (str, optional): RL algorithm to use
            
        Returns:
            object: RL model
        """
        if algorithm is None:
            algorithm = self.algorithm
            
        if not HAS_SB3:
            self.logger.error("Stable Baselines 3 is required but not installed")
            return None
        
        if algorithm == 'ppo':
            # Create PPO model with MLP policy
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                gamma=0.99,
                tensorboard_log=os.path.join(self.models_dir, "tb_logs")
            )
            
        elif algorithm == 'a2c':
            # Create A2C model
            model = A2C(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=0.0007,
                gamma=0.99,
                tensorboard_log=os.path.join(self.models_dir, "tb_logs")
            )
            
        elif algorithm == 'sac':
            # Create SAC model for continuous actions (not suitable for this env)
            self.logger.warning("SAC is designed for continuous action spaces, defaulting to PPO")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=0.0003,
                tensorboard_log=os.path.join(self.models_dir, "tb_logs")
            )
            
        else:
            self.logger.error(f"Unsupported algorithm: {algorithm}, defaulting to PPO")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=os.path.join(self.models_dir, "tb_logs")
            )
        
        return model
    
    def train(self, sentiment_data, price_data, total_timesteps=None, model_name=None):
        """
        Train the reinforcement learning model.
        
        Args:
            sentiment_data (pd.DataFrame): Market sentiment data
            price_data (pd.DataFrame): Price data for the same time period
            total_timesteps (int, optional): Total training timesteps
            model_name (str, optional): Name for the saved model
            
        Returns:
            dict: Training metrics
        """
        self.logger.info(f"Training {self.algorithm} model on sentiment data")
        
        # Set total timesteps
        if total_timesteps is None:
            total_timesteps = self.total_timesteps
        
        # Create environment
        self.env = self.create_environment(sentiment_data, price_data)
        
        # Build model
        self.model = self.build_model(self.env)
        
        if self.model is None:
            self.logger.error("Failed to build model, cannot train")
            return None
        
        # Set up evaluation callback
        eval_env = self.create_environment(sentiment_data, price_data)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.models_dir, "best_model"),
            log_path=os.path.join(self.models_dir, "eval_logs"),
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        # Train model
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=eval_callback
            )
            
            # Save model
            if model_name:
                model_path = os.path.join(self.models_dir, f"{model_name}.zip")
                self.model.save(model_path)
                self.logger.info(f"Model saved to {model_path}")
                
                # Save environment configuration
                env_config = {
                    'lookback': self.lookback,
                    'window_size': self.window_size,
                    'algorithm': self.algorithm
                }
                config_path = os.path.join(self.models_dir, f"{model_name}_config.json")
                with open(config_path, 'w') as f:
                    json.dump(env_config, f, indent=4)
            
            # Evaluate model
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                eval_env,
                n_eval_episodes=10,
                deterministic=True
            )
            
            metrics = {
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'total_timesteps': total_timesteps,
                'algorithm': self.algorithm
            }
            
            self.logger.info(f"Model training completed. Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            return None
    
    def load_model(self, model_path, config_path=None):
        """
        Load a trained model from file.
        
        Args:
            model_path (str): Path to saved model
            config_path (str, optional): Path to environment configuration
            
        Returns:
            bool: True if model loaded successfully
        """
        if not HAS_SB3:
            self.logger.error("Stable Baselines 3 is required but not installed")
            return False
        
        try:
            # Load environment configuration if provided
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    env_config = json.load(f)
                
                self.lookback = env_config.get('lookback', self.lookback)
                self.window_size = env_config.get('window_size', self.window_size)
                self.algorithm = env_config.get('algorithm', self.algorithm)
                
            # Determine algorithm from path if not specified in config
            if 'ppo' in model_path.lower():
                self.algorithm = 'ppo'
                model_class = PPO
            elif 'a2c' in model_path.lower():
                self.algorithm = 'a2c'
                model_class = A2C
            elif 'sac' in model_path.lower():
                self.algorithm = 'sac'
                model_class = SAC
            else:
                # Default to PPO
                model_class = PPO
            
            # Load model
            self.model = model_class.load(model_path)
            
            self.logger.info(f"Loaded {self.algorithm} model from {model_path}")
            return True
                
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            return False
    
    def predict(self, sentiment_data, price_data, deterministic=True):
        """
        Make predictions using the trained model.
        
        Args:
            sentiment_data (pd.DataFrame): Market sentiment data
            price_data (pd.DataFrame): Price data for the same time period
            deterministic (bool): Whether to use deterministic actions
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            self.logger.error("Model not trained or loaded")
            return None
        
        self.logger.info("Making predictions on sentiment data")
        
        try:
            # Create environment for prediction
            predict_env = self.create_environment(sentiment_data, price_data)
            
            # Get initial observation
            obs = predict_env.reset()
            
            # Run prediction
            actions = []
            rewards = []
            positions = []
            done = False
            
            while not done:
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=deterministic)
                
                # Execute action in environment
                obs, reward, done_array, info = predict_env.step(action)
                
                # For vectorized environments, extract first element
                done = done_array[0] if isinstance(done_array, (list, np.ndarray)) else done_array
                
                # Store action and reward
                actions.append(int(action[0]) if isinstance(action, (list, np.ndarray)) else int(action))
                rewards.append(float(reward[0]) if isinstance(reward, (list, np.ndarray)) else float(reward))
                
                # Map action to position
                position = "no_position" if actions[-1] == 0 else "long" if actions[-1] == 1 else "short"
                positions.append(position)
            
            # Calculate cumulative reward
            cumulative_reward = sum(rewards)
            
            # Get final position distribution
            position_counts = {
                'no_position': positions.count('no_position'),
                'long': positions.count('long'),
                'short': positions.count('short')
            }
            
            # Create timestamp-based predictions
            predictions = []
            
            # Create dataframe copy with datetime index
            sentiment_df = sentiment_data.copy()
            if not isinstance(sentiment_df.index, pd.DatetimeIndex):
                # Try to convert date column to index
                if 'date' in sentiment_df.columns:
                    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                    sentiment_df = sentiment_df.set_index('date')
                elif 'timestamp' in sentiment_df.columns:
                    sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
                    sentiment_df = sentiment_df.set_index('timestamp')
            
            # Map positions to timestamps
            dates = sentiment_df.index[self.window_size:]
            
            for i, (date, position, reward) in enumerate(zip(dates, positions, rewards)):
                if i >= len(dates):
                    break
                
                predictions.append({
                    'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
                    'position': position,
                    'action': actions[i],
                    'reward': reward,
                    'cumulative_reward': sum(rewards[:i+1])
                })
            
            # Calculate strategy performance
            performance = {
                'total_reward': cumulative_reward,
                'mean_reward': np.mean(rewards),
                'position_distribution': position_counts,
                'win_rate': sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0,
                'reward_volatility': np.std(rewards) if len(rewards) > 1 else 0
            }
            
            self.logger.info(f"Generated predictions with cumulative reward: {cumulative_reward:.2f}")
            
            return {
                'predictions': predictions,
                'performance': performance,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def evaluate_strategy(self, sentiment_data, price_data, initial_balance=10000.0):
        """
        Evaluate the trading strategy using the trained model.
        
        Args:
            sentiment_data (pd.DataFrame): Market sentiment data
            price_data (pd.DataFrame): Price data for the same time period
            initial_balance (float): Initial balance for backtesting
            
        Returns:
            dict: Evaluation metrics
        """
        self.logger.info("Evaluating strategy performance")
        
        try:
            # Run predictions
            prediction_results = self.predict(sentiment_data, price_data, deterministic=True)
            
            if prediction_results is None:
                self.logger.error("Failed to generate predictions")
                return None
            
            predictions = prediction_results['predictions']
            
            # Initialize backtest variables
            balance = initial_balance
            position = "no_position"
            entry_price = None
            trades = []
            equity_curve = []
            
            # Extract prices at each timestamp
            price_df = price_data.copy()
            if not isinstance(price_df.index, pd.DatetimeIndex):
                # Try to convert date column to index
                if 'date' in price_df.columns:
                    price_df['date'] = pd.to_datetime(price_df['date'])
                    price_df = price_df.set_index('date')
                elif 'timestamp' in price_df.columns:
                    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
                    price_df = price_df.set_index('timestamp')
            
            # Ensure we have close prices
            if 'close' in price_df.columns:
                price_column = 'close'
            elif 'price' in price_df.columns:
                price_column = 'price'
            else:
                price_column = price_df.columns[0]  # Use first column
            
            # Simulate trading
            for i, pred in enumerate(predictions):
                timestamp = pd.to_datetime(pred['timestamp'])
                new_position = pred['position']
                
                # Get current price
                try:
                    current_price = price_df.loc[timestamp, price_column]
                    if isinstance(current_price, pd.Series):
                        current_price = current_price.iloc[0]
                except (KeyError, IndexError):
                    # Find closest timestamp if exact match not found
                    closest_idx = price_df.index.get_indexer([timestamp], method='nearest')[0]
                    if closest_idx >= 0 and closest_idx < len(price_df):
                        current_price = price_df.iloc[closest_idx][price_column]
                    else:
                        continue  # Skip if no price available
                
                # Record equity at this point
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': balance,
                    'position': position
                })
                
                # Check for position change
                if new_position != position:
                    # Close existing position
                    if position != "no_position" and entry_price is not None:
                        pnl = 0
                        if position == "long":
                            pnl = (current_price - entry_price) / entry_price * balance
                        elif position == "short":
                            pnl = (entry_price - current_price) / entry_price * balance
                        
                        # Update balance
                        balance += pnl
                        
                        # Record trade
                        trades.append({
                            'entry_time': entry_timestamp,
                            'exit_time': timestamp,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position': position,
                            'pnl': pnl,
                            'pnl_percent': pnl / (balance - pnl) * 100 if balance != pnl else 0
                        })
                    
                    # Open new position
                    if new_position != "no_position":
                        entry_price = current_price
                        entry_timestamp = timestamp
                    else:
                        entry_price = None
                    
                    position = new_position
            
            # Close any remaining position at the end
            if position != "no_position" and entry_price is not None:
                # Get last price
                last_price = price_df[price_column].iloc[-1]
                
                # Calculate PNL
                pnl = 0
                if position == "long":
                    pnl = (last_price - entry_price) / entry_price * balance
                elif position == "short":
                    pnl = (entry_price - last_price) / entry_price * balance
                
                # Update balance
                balance += pnl
                
                # Record final trade
                trades.append({
                    'entry_time': entry_timestamp,
                    'exit_time': price_df.index[-1],
                    'entry_price': entry_price,
                    'exit_price': last_price,
                    'position': position,
                    'pnl': pnl,
                    'pnl_percent': pnl / (balance - pnl) * 100 if balance != pnl else 0
                })
            
            # Add final equity point
            equity_curve.append({
                'timestamp': price_df.index[-1] if len(price_df) > 0 else pd.Timestamp.now(),
                'equity': balance,
                'position': 'no_position'
            })
            
            # Calculate performance metrics
            total_return = (balance - initial_balance) / initial_balance * 100
            
            # Calculate win rate
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] <= 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            # Calculate average profit and loss
            avg_profit = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            
            # Calculate profit factor
            gross_profit = sum(t['pnl'] for t in winning_trades)
            gross_loss = sum(t['pnl'] for t in losing_trades)
            profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
            
            # Calculate Sharpe ratio (simplified)
            if len(equity_curve) > 1:
                equity_values = [e['equity'] for e in equity_curve]
                returns = np.diff(equity_values) / equity_values[:-1]
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Compile results
            evaluation = {
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_return_percent': total_return,
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'trades': trades,
                'equity_curve': equity_curve
            }
            
            self.logger.info(f"Strategy evaluation completed. Total return: {total_return:.2f}%, Win rate: {win_rate:.2f}")
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating strategy: {str(e)}")
            return None
    
    def visualize_performance(self, evaluation_results):
        """
        Visualize strategy performance.
        
        Args:
            evaluation_results (dict): Results from evaluate_strategy
            
        Returns:
            tuple: Figure and axes
        """
        if not evaluation_results:
            self.logger.error("No evaluation results to visualize")
            return None, None
        
        try:
            # Create figure with subplots
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot equity curve
            equity_data = evaluation_results['equity_curve']
            if equity_data:
                timestamps = [pd.to_datetime(e['timestamp']) for e in equity_data]
                equity = [e['equity'] for e in equity_data]
                
                axs[0, 0].plot(timestamps, equity, linewidth=2)
                axs[0, 0].set_title('Equity Curve', fontsize=14)
                axs[0, 0].set_xlabel('Date')
                axs[0, 0].set_ylabel('Equity')
                axs[0, 0].grid(True)
                
                # Highlight different positions
                position_changes = []
                for i in range(1, len(equity_data)):
                    if equity_data[i]['position'] != equity_data[i-1]['position']:
                        position_changes.append((timestamps[i], equity_data[i]['position']))
                
                # Plot position changes
                for timestamp, position in position_changes:
                    color = 'green' if position == 'long' else 'red' if position == 'short' else 'gray'
                    axs[0, 0].axvline(x=timestamp, color=color, linestyle='--', alpha=0.5)
                
                # Format y-axis as currency
                axs[0, 0].yaxis.set_major_formatter('${x:,.2f}')
            else:
                axs[0, 0].text(0.5, 0.5, "No equity data available", 
                             ha='center', va='center', transform=axs[0, 0].transAxes)
            
            # Plot trade outcomes
            trades = evaluation_results['trades']
            if trades:
                # Calculate trade PnLs
                pnls = [trade['pnl'] for trade in trades]
                
                # Create histogram of trade PnLs
                axs[0, 1].hist(pnls, bins=20, color=['green' if pnl > 0 else 'red' for pnl in pnls])
                axs[0, 1].set_title('Trade PnL Distribution', fontsize=14)
                axs[0, 1].set_xlabel('PnL ($)')
                axs[0, 1].set_ylabel('Frequency')
                axs[0, 1].grid(True)
                
                # Add mean and median lines
                mean_pnl = np.mean(pnls)
                median_pnl = np.median(pnls)
                axs[0, 1].axvline(x=mean_pnl, color='blue', linestyle='-', label=f'Mean: ${mean_pnl:.2f}')
                axs[0, 1].axvline(x=median_pnl, color='orange', linestyle='--', label=f'Median: ${median_pnl:.2f}')
                axs[0, 1].axvline(x=0, color='black', linestyle='-')
                axs[0, 1].legend()
            else:
                axs[0, 1].text(0.5, 0.5, "No trade data available", 
                             ha='center', va='center', transform=axs[0, 1].transAxes)
            
            # Plot key metrics
            metrics = [
                ('Win Rate', evaluation_results['win_rate']),
                ('Profit Factor', evaluation_results['profit_factor']),
                ('Sharpe Ratio', evaluation_results['sharpe_ratio']),
                ('Total Return', evaluation_results['total_return_percent'] / 100)
            ]
            
            metric_names, metric_values = zip(*metrics)
            bars = axs[1, 0].bar(metric_names, metric_values)
            
            # Color bars based on value
            for i, bar in enumerate(bars):
                if metric_values[i] > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            axs[1, 0].set_title('Performance Metrics', fontsize=14)
            axs[1, 0].grid(True)
            
            # Add value labels
            for i, v in enumerate(metric_values):
                axs[1, 0].text(i, v + 0.02, f"{v:.2f}", ha='center')
            
            # Plot trade counts
            counts = [
                evaluation_results['total_trades'],
                evaluation_results['winning_trades'],
                evaluation_results['losing_trades']
            ]
            
            labels = ['Total Trades', 'Winning Trades', 'Losing Trades']
            colors = ['blue', 'green', 'red']
            
            axs[1, 1].pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            axs[1, 1].set_title('Trade Distribution', fontsize=14)
            axs[1, 1].axis('equal')
            
            # Add text with trade statistics
            win_rate = evaluation_results['win_rate'] * 100
            total_return = evaluation_results['total_return_percent']
            
            fig.text(0.5, 0.01, 
                   f"Total Return: {total_return:.2f}% | Win Rate: {win_rate:.2f}% | "
                   f"Trades: {evaluation_results['total_trades']} | "
                   f"Profit Factor: {evaluation_results['profit_factor']:.2f}",
                   ha='center', fontsize=12)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"{self.models_dir}/strategy_performance_{timestamp}.png"
            plt.savefig(plot_file)
            
            self.logger.info(f"Performance visualization saved to {plot_file}")
            
            return fig, axs
            
        except Exception as e:
            self.logger.error(f"Error visualizing performance: {str(e)}")
            return None, None
    
    def analyze_feature_importance(self, sentiment_data, price_data):
        """
        Analyze feature importance using SHAP values.
        
        Args:
            sentiment_data (pd.DataFrame): Market sentiment data
            price_data (pd.DataFrame): Price data for the same time period
            
        Returns:
            dict: Feature importance analysis
        """
        if not HAS_SHAP:
            self.logger.error("SHAP package is required for feature importance analysis")
            return None
        
        if self.model is None:
            self.logger.error("Model not trained or loaded")
            return None
        
        self.logger.info("Analyzing feature importance")
        
        try:
            # Create environment for prediction
            env = SentimentEnv(sentiment_data, price_data, lookback=self.lookback, window_size=self.window_size)
            
            # Extract features
            features = env._process_data()
            
            # Create a sampling of states
            n_samples = min(100, len(features))
            sample_indices = np.random.choice(len(features) - env.lookback, n_samples, replace=False)
            
            states = []
            for idx in sample_indices:
                state = features[idx:idx+env.lookback]
                states.append(state)
            
            # Convert to numpy array
            states = np.array(states)
            
            # Get action probabilities
            if hasattr(self.model, 'predict_proba'):
                action_probs = self.model.predict_proba(states)
            else:
                # If no predict_proba, use deterministic actions
                actions = []
                for state in states:
                    action, _ = self.model.predict(state.reshape(1, -1, env.n_features), deterministic=True)
                    actions.append(action)
                action_probs = np.eye(3)[np.array(actions).flatten()]
            
            # Initialize SHAP explainer
            explainer = shap.KernelExplainer(
                lambda x: self.model.predict_proba(x) if hasattr(self.model, 'predict_proba') else None,
                shap.sample(states, 10)
            )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(states[:10])
            
            # Process feature importance
            feature_names = []
            for i in range(env.n_features):
                feature_names.append(f"Feature_{i}")
            
            importance_values = []
            for i, feature in enumerate(feature_names):
                importance_values.append({
                    'feature': feature,
                    'importance': np.mean([abs(shap_values[j][:, :, i]).mean() for j in range(len(shap_values))])
                })
            
            # Sort by importance
            importance_values.sort(key=lambda x: x['importance'], reverse=True)
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values[0], states, feature_names=feature_names)
            
            # Save visualization
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            shap_file = f"{self.models_dir}/feature_importance_{timestamp}.png"
            plt.savefig(shap_file)
            
            self.logger.info(f"Feature importance analysis saved to {shap_file}")
            
            return {
                'feature_importance': importance_values,
                'top_features': [x['feature'] for x in importance_values[:5]],
                'visualization_path': shap_file
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature importance: {str(e)}")
            return None
