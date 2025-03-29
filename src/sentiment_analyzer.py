"""
Sentiment Analyzer Module

This module analyzes social media and news sentiment for stocks
to determine market sentiment and inform trading decisions.
"""

import logging
import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import re
import os
import math

class SentimentAnalyzer:
    """
    Analyzes Twitter/X and news sentiment for trading signals.
    
    Features:
    - Social media sentiment analysis
    - News sentiment analysis
    - Sentiment scoring and classification
    - Sentiment trend detection
    - Sentiment-based trading signals
    """
    
    # Sentiment classification constants
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"
    
    def __init__(self, twitter_credentials):
        """
        Initialize the SentimentAnalyzer.
        
        Args:
            twitter_credentials (dict): Twitter API credentials
        """
        self.logger = logging.getLogger(__name__)
        
        # Twitter API credentials
        self.twitter_credentials = twitter_credentials
        
        # Cache for sentiment data
        self.sentiment_cache = {}  # {symbol: {timestamp: sentiment_data}}
        
        # Ensure cache directory exists
        os.makedirs("data/sentiment", exist_ok=True)
        
        self.logger.info("SentimentAnalyzer initialized")
    
    def analyze_sentiment(self, symbol, include_news=True, include_social=True):
        """
        Analyze sentiment for a symbol from multiple sources.
        
        Args:
            symbol (str): Stock symbol
            include_news (bool): Include news sentiment
            include_social (bool): Include social media sentiment
            
        Returns:
            str: Overall sentiment classification
        """
        self.logger.debug(f"Analyzing sentiment for {symbol}")
        
        try:
            # Get current timestamp
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Check if we have recent data in cache (within last hour)
            if symbol in self.sentiment_cache:
                latest_time = max(self.sentiment_cache[symbol].keys())
                latest_dt = datetime.strptime(latest_time, '%Y-%m-%d %H:%M:%S')
                
                if (datetime.now() - latest_dt).total_seconds() < 3600:
                    # Use cached data
                    return self.sentiment_cache[symbol][latest_time]['classification']
            
            # Initialize sentiment scores
            sentiment_scores = []
            
            # Get social media sentiment if requested
            if include_social:
                social_sentiment = self._analyze_twitter_sentiment(symbol)
                if social_sentiment:
                    sentiment_scores.append(social_sentiment)
            
            # Get news sentiment if requested
            if include_news:
                news_sentiment = self._analyze_news_sentiment(symbol)
                if news_sentiment:
                    sentiment_scores.append(news_sentiment)
            
            # If no data available, return neutral
            if not sentiment_scores:
                self.logger.warning(f"No sentiment data available for {symbol}")
                return self.NEUTRAL
            
            # Average sentiment scores
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            # Classify sentiment
            classification = self._classify_sentiment(avg_sentiment)
            
            # Store in cache
            if symbol not in self.sentiment_cache:
                self.sentiment_cache[symbol] = {}
                
            self.sentiment_cache[symbol][now] = {
                'score': avg_sentiment,
                'classification': classification,
                'social_score': social_sentiment if include_social else None,
                'news_score': news_sentiment if include_news else None
            }
            
            # Save to file periodically
            self._save_sentiment_data(symbol)
            
            return classification
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            return self.NEUTRAL
    
    def _analyze_twitter_sentiment(self, symbol):
        """
        Analyze Twitter/X sentiment for a symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            float: Sentiment score (-1.0 to 1.0) or None if unavailable
        """
        try:
            # Check if we have valid Twitter credentials
            if not self.twitter_credentials or not self.twitter_credentials.get("api_key"):
                self.logger.warning("Twitter credentials not configured")
                return None
            
            # For now, we'll simulate Twitter sentiment analysis
            # In a real implementation, you would call the Twitter API
            
            # Simulate a sentiment score based on symbol
            # This is just a placeholder - replace with actual API call
            char_sum = sum(ord(c) for c in symbol)
            simulated_sentiment = (char_sum % 20 - 10) / 10.0  # Between -1.0 and 1.0
            
            # Add some randomness
            import random
            simulated_sentiment += random.uniform(-0.2, 0.2)
            simulated_sentiment = max(-1.0, min(1.0, simulated_sentiment))  # Clamp to range
            
            return simulated_sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing Twitter sentiment for {symbol}: {str(e)}")
            return None
    
    def _analyze_news_sentiment(self, symbol):
        """
        Analyze news sentiment for a symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            float: Sentiment score (-1.0 to 1.0) or None if unavailable
        """
        try:
            # In a real implementation, you would call a news API or service
            # For now, we'll simulate news sentiment
            
            # Simulate sentiment based on symbol and current date
            today = datetime.now().day
            simulated_sentiment = (today % 20 - 10) / 10.0  # Between -1.0 and 1.0
            
            # Adjust by symbol
            char_sum = sum(ord(c) for c in symbol)
            symbol_factor = (char_sum % 10) / 20.0  # Small adjustment
            
            simulated_sentiment += symbol_factor
            simulated_sentiment = max(-1.0, min(1.0, simulated_sentiment))  # Clamp to range
            
            return simulated_sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment for {symbol}: {str(e)}")
            return None
    
    def _classify_sentiment(self, sentiment_score):
        """
        Classify a sentiment score into a category.
        
        Args:
            sentiment_score (float): Sentiment score (-1.0 to 1.0)
            
        Returns:
            str: Sentiment classification
        """
        if sentiment_score <= -0.6:
            return self.VERY_BEARISH
        elif sentiment_score <= -0.2:
            return self.BEARISH
        elif sentiment_score <= 0.2:
            return self.NEUTRAL
        elif sentiment_score <= 0.6:
            return self.BULLISH
        else:
            return self.VERY_BULLISH
    
    def _save_sentiment_data(self, symbol):
        """
        Save sentiment data to file.
        
        Args:
            symbol (str): Stock symbol
        """
        try:
            if symbol not in self.sentiment_cache:
                return
                
            # Convert to DataFrame
            data = []
            for timestamp, values in self.sentiment_cache[symbol].items():
                entry = {
                    'timestamp': timestamp,
                    'score': values['score'],
                    'classification': values['classification'],
                    'social_score': values.get('social_score'),
                    'news_score': values.get('news_score')
                }
                data.append(entry)
                
            df = pd.DataFrame(data)
            
            # Save to CSV
            file_path = f"data/sentiment/{symbol}_sentiment.csv"
            df.to_csv(file_path, index=False)
            
        except Exception as e:
            self.logger.error(f"Error saving sentiment data for {symbol}: {str(e)}")
    
    def get_sentiment(self, symbol, lookback_hours=24):
        """
        Get current sentiment for a symbol.
        
        Args:
            symbol (str): Stock symbol
            lookback_hours (int): Hours to look back for cached data
            
        Returns:
            str: Sentiment classification
        """
        try:
            # Check if we have data in cache
            if symbol in self.sentiment_cache:
                # Find the most recent entry within lookback period
                now = datetime.now()
                lookback_threshold = now - timedelta(hours=lookback_hours)
                
                valid_entries = []
                for timestamp, data in self.sentiment_cache[symbol].items():
                    entry_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    if entry_time >= lookback_threshold:
                        valid_entries.append((entry_time, data))
                
                if valid_entries:
                    # Sort by time (most recent first)
                    valid_entries.sort(reverse=True)
                    return valid_entries[0][1]['classification']
            
            # If no cached data, analyze sentiment now
            return self.analyze_sentiment(symbol)
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
            return self.NEUTRAL
    
    def get_sentiment_trend(self, symbol, days=7):
        """
        Get sentiment trend over time.
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days to analyze
            
        Returns:
            dict: Sentiment trend data
        """
        try:
            # Try to load from file first
            file_path = f"data/sentiment/{symbol}_sentiment.csv"
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Filter for requested time period
                start_date = datetime.now() - timedelta(days=days)
                df = df[df['timestamp'] >= start_date]
                
                if df.empty:
                    return {'trend': 'neutral', 'data': []}
                
                # Calculate daily average sentiment
                df['date'] = df['timestamp'].dt.date
                daily_sentiment = df.groupby('date')['score'].mean().reset_index()
                
                # Determine trend
                if len(daily_sentiment) < 2:
                    trend = 'insufficient_data'
                else:
                    first_half = daily_sentiment.iloc[:len(daily_sentiment)//2]
                    second_half = daily_sentiment.iloc[len(daily_sentiment)//2:]
                    
                    first_avg = first_half['score'].mean()
                    second_avg = second_half['score'].mean()
                    
                    if second_avg > first_avg + 0.2:
                        trend = 'strongly_improving'
                    elif second_avg > first_avg + 0.05:
                        trend = 'improving'
                    elif second_avg < first_avg - 0.2:
                        trend = 'strongly_deteriorating'
                    elif second_avg < first_avg - 0.05:
                        trend = 'deteriorating'
                    else:
                        trend = 'stable'
                
                # Format data for return
                trend_data = []
                for _, row in daily_sentiment.iterrows():
                    trend_data.append({
                        'date': row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], datetime.date) else row['date'],
                        'score': row['score'],
                        'classification': self._classify_sentiment(row['score'])
                    })
                
                return {
                    'trend': trend,
                    'data': trend_data,
                    'current': self._classify_sentiment(daily_sentiment.iloc[-1]['score']) if not daily_sentiment.empty else 'neutral'
                }
            
            # If no file exists, return empty data
            return {'trend': 'insufficient_data', 'data': []}
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment trend for {symbol}: {str(e)}")
            return {'trend': 'error', 'data': []}
    
    def analyze_market_sentiment(self, market_symbols=None):
        """
        Analyze overall market sentiment using major indices.
        
        Args:
            market_symbols (list, optional): List of symbols to use for market sentiment
            
        Returns:
            dict: Market sentiment analysis
        """
        try:
            # Use default market symbols if none provided
            if market_symbols is None:
                market_symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
            
            # Get sentiment for each symbol
            sentiments = {}
            sentiment_scores = []
            
            for symbol in market_symbols:
                result = self.analyze_sentiment(symbol)
                sentiments[symbol] = result
                
                # Convert classification to score for averaging
                if result == self.VERY_BEARISH:
                    score = -1.0
                elif result == self.BEARISH:
                    score = -0.5
                elif result == self.NEUTRAL:
                    score = 0.0
                elif result == self.BULLISH:
                    score = 0.5
                else:  # VERY_BULLISH
                    score = 1.0
                
                sentiment_scores.append(score)
            
            # Calculate average sentiment
            if sentiment_scores:
                avg_score = sum(sentiment_scores) / len(sentiment_scores)
                market_sentiment = self._classify_sentiment(avg_score)
            else:
                market_sentiment = self.NEUTRAL
            
            return {
                'market_sentiment': market_sentiment,
                'symbol_sentiments': sentiments,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment: {str(e)}")
            return {'market_sentiment': self.NEUTRAL, 'symbol_sentiments': {}}
    
    def sentiment_to_signal(self, sentiment, confidence_threshold=0.7):
        """
        Convert sentiment classification to a trading signal.
        
        Args:
            sentiment (str): Sentiment classification
            confidence_threshold (float): Minimum confidence for generating signal
            
        Returns:
            dict: Trading signal information
        """
        # Map sentiment to trade direction and confidence
        if sentiment == self.VERY_BULLISH:
            direction = 'buy'
            confidence = 0.9
        elif sentiment == self.BULLISH:
            direction = 'buy'
            confidence = 0.7
        elif sentiment == self.VERY_BEARISH:
            direction = 'sell'
            confidence = 0.9
        elif sentiment == self.BEARISH:
            direction = 'sell'
            confidence = 0.7
        else:  # NEUTRAL
            direction = 'hold'
            confidence = 0.5
        
        # Determine if signal is actionable
        actionable = confidence >= confidence_threshold
        
        return {
            'signal': direction,
            'confidence': confidence,
            'actionable': actionable,
            'sentiment': sentiment
        }
    
    def load_historical_data(self):
        """
        Load historical sentiment data from files.
        
        Returns:
            dict: Symbol to sentiment data mapping
        """
        try:
            data_dir = "data/sentiment"
            
            if not os.path.exists(data_dir):
                return {}
            
            # Find all sentiment CSV files
            sentiment_files = [f for f in os.listdir(data_dir) if f.endswith('_sentiment.csv')]
            
            # Load each file
            for file in sentiment_files:
                symbol = file.split('_')[0]
                file_path = os.path.join(data_dir, file)
                
                try:
                    df = pd.read_csv(file_path)
                    
                    # Convert to cache format
                    if symbol not in self.sentiment_cache:
                        self.sentiment_cache[symbol] = {}
                    
                    for _, row in df.iterrows():
                        timestamp = row['timestamp']
                        self.sentiment_cache[symbol][timestamp] = {
                            'score': row['score'],
                            'classification': row['classification'],
                            'social_score': row.get('social_score'),
                            'news_score': row.get('news_score')
                        }
                        
                    self.logger.debug(f"Loaded historical sentiment data for {symbol}: {len(df)} entries")
                    
                except Exception as e:
                    self.logger.error(f"Error loading sentiment file {file}: {str(e)}")
            
            return self.sentiment_cache
            
        except Exception as e:
            self.logger.error(f"Error loading historical sentiment data: {str(e)}")
            return {}
