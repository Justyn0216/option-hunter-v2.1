"""
Sentiment Model Module

This module implements machine learning models for analyzing market sentiment
from social media, news articles, and other text data to predict market movements.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from textblob import TextBlob
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentModel:
    """
    Machine learning model for analyzing market sentiment from text data.
    
    Features:
    - Social media sentiment analysis
    - News article sentiment extraction
    - Topic modeling for market themes
    - Sentiment-based market prediction
    """
    
    def __init__(self, config=None):
        """
        Initialize the SentimentModel.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.models_dir = "models/sentiment"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Default parameters
        self.model_type = self.config.get('sentiment_model_type', 'xgboost')
        self.use_pretrained = self.config.get('use_pretrained_sentiment', True)
        self.min_token_length = self.config.get('min_token_length', 3)
        self.max_features = self.config.get('max_features', 5000)
        self.n_topics = self.config.get('n_topics', 10)
        
        # Initialize models
        self.model = None
        self.vectorizer = None
        self.topic_model = None
        
        # Initialize NLP resources
        self._initialize_nlp_resources()
        
        self.logger.info(f"SentimentModel initialized with {self.model_type} model")
    
    def _initialize_nlp_resources(self):
        """Initialize NLP resources and models."""
        try:
            # Download NLTK resources if not already downloaded
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
                
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet')
            
            # Initialize stop words
            self.stop_words = set(stopwords.words('english'))
            
            # Extend stop words with financial terms that don't convey sentiment
            financial_stop_words = {'stock', 'stocks', 'market', 'markets', 'trade', 'trading',
                                  'investment', 'invest', 'price', 'prices', 'share', 'shares',
                                  'company', 'companies', 'corp', 'inc', 'nasdaq', 'nyse',
                                  'dow', 'sp500', 's&p', 'etf', 'option', 'options', 'call', 'put'}
            
            self.stop_words.update(financial_stop_words)
            
            # Initialize lemmatizer
            self.lemmatizer = WordNetLemmatizer()
            
            # Initialize sentiment analyzers
            self.vader = SentimentIntensityAnalyzer()
            
            # Initialize spaCy (if available)
            try:
                self.nlp = spacy.load('en_core_web_sm')
                self.has_spacy = True
            except:
                self.logger.warning("SpaCy model not available. Using fallback NLP processing.")
                self.has_spacy = False
            
            # Initialize pretrained sentiment model if configured
            if self.use_pretrained:
                pretrained_path = os.path.join(self.models_dir, "pretrained_sentiment.pkl")
                if os.path.exists(pretrained_path):
                    with open(pretrained_path, 'rb') as f:
                        self.pretrained = pickle.load(f)
                    self.logger.info("Loaded pretrained sentiment model")
                else:
                    self.pretrained = None
                    self.logger.info("No pretrained sentiment model found, will train from scratch")
            else:
                self.pretrained = None
            
            self.logger.info("NLP resources initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing NLP resources: {str(e)}")
            raise
    
    def preprocess_text(self, texts, lemmatize=True):
        """
        Preprocess text data for sentiment analysis.
        
        Args:
            texts (list): List of text strings
            lemmatize (bool): Whether to lemmatize tokens
            
        Returns:
            list: List of preprocessed text strings
        """
        self.logger.info(f"Preprocessing {len(texts)} text documents")
        
        processed_texts = []
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                processed_texts.append("")
                continue
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            
            # Remove mentions and hashtags for social media
            text = re.sub(r'@\w+|#\w+', '', text)
            
            # Remove punctuation and numbers
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stop words and short tokens
            tokens = [token for token in tokens if token not in self.stop_words and len(token) >= self.min_token_length]
            
            # Lemmatize if requested
            if lemmatize:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            # Rejoin
            processed_text = ' '.join(tokens)
            processed_texts.append(processed_text)
        
        self.logger.info(f"Text preprocessing completed")
        return processed_texts
    
    def extract_features(self, texts, feature_type='tfidf'):
        """
        Extract features from preprocessed text.
        
        Args:
            texts (list): List of preprocessed text strings
            feature_type (str): Type of features to extract ('tfidf', 'count', or 'both')
            
        Returns:
            scipy.sparse.csr.csr_matrix: Feature matrix
        """
        self.logger.info(f"Extracting {feature_type} features from text")
        
        # Create vectorizer if not already created
        if self.vectorizer is None:
            if feature_type == 'tfidf':
                self.vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    min_df=5,
                    max_df=0.8,
                    ngram_range=(1, 2),
                    stop_words=self.stop_words
                )
            elif feature_type == 'count':
                self.vectorizer = CountVectorizer(
                    max_features=self.max_features,
                    min_df=5,
                    max_df=0.8,
                    ngram_range=(1, 2),
                    stop_words=self.stop_words
                )
            else:  # 'both' - use TF-IDF as default
                self.vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    min_df=5,
                    max_df=0.8,
                    ngram_range=(1, 2),
                    stop_words=self.stop_words
                )
        
        # Transform texts to feature matrix
        X = self.vectorizer.fit_transform(texts)
        
        self.logger.info(f"Extracted {X.shape[1]} features from text")
        return X
    
    def extract_sentiment_features(self, texts):
        """
        Extract sentiment features using rule-based analyzers.
        
        Args:
            texts (list): List of text strings (raw, not preprocessed)
            
        Returns:
            numpy.ndarray: Sentiment feature matrix
        """
        self.logger.info(f"Extracting sentiment features from {len(texts)} texts")
        
        features = []
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                features.append([0, 0, 0, 0, 0, 0])
                continue
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # VADER sentiment
            vader_scores = self.vader.polarity_scores(text)
            vader_compound = vader_scores['compound']
            vader_pos = vader_scores['pos']
            vader_neg = vader_scores['neg']
            vader_neu = vader_scores['neu']
            
            # Combine features
            features.append([
                textblob_polarity,
                textblob_subjectivity,
                vader_compound,
                vader_pos,
                vader_neg,
                vader_neu
            ])
        
        return np.array(features)
    
    def combine_features(self, text_features, sentiment_features):
        """
        Combine text-based and sentiment-based features.
        
        Args:
            text_features (scipy.sparse.csr.csr_matrix): Text features
            sentiment_features (numpy.ndarray): Sentiment features
            
        Returns:
            scipy.sparse.csr.csr_matrix: Combined feature matrix
        """
        # Convert sparse matrix to dense if needed
        from scipy.sparse import hstack, csr_matrix
        
        # Convert sentiment features to sparse matrix
        sentiment_sparse = csr_matrix(sentiment_features)
        
        # Combine horizontally
        combined = hstack([text_features, sentiment_sparse])
        
        return combined
    
    def build_topic_model(self, texts, n_topics=None, method='lda'):
        """
        Build a topic model from text data.
        
        Args:
            texts (list): List of preprocessed text strings
            n_topics (int, optional): Number of topics to extract
            method (str): Topic modeling method ('lda' or 'nmf')
            
        Returns:
            object: Trained topic model
        """
        if n_topics is None:
            n_topics = self.n_topics
        
        self.logger.info(f"Building {method} topic model with {n_topics} topics")
        
        # Create document-term matrix
        vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=5,
            max_df=0.8,
            stop_words=self.stop_words
        )
        
        X = vectorizer.fit_transform(texts)
        
        # Build topic model
        if method == 'lda':
            model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method='online',
                random_state=42,
                n_jobs=-1
            )
        else:  # 'nmf'
            model = NMF(
                n_components=n_topics,
                random_state=42,
                alpha=0.1,
                l1_ratio=0.5
            )
        
        model.fit(X)
        
        # Store vectorizer with the model
        self.topic_vectorizer = vectorizer
        self.topic_model = model
        
        # Extract top words for each topic
        feature_names = vectorizer.get_feature_names_out()
        
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[:-11:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'id': topic_idx,
                'words': top_words,
                'weights': topic[top_words_idx].tolist()
            })
        
        self.topics = topics
        
        self.logger.info(f"Topic model built successfully")
        return model
    
    def get_document_topics(self, texts):
        """
        Get topic distributions for documents.
        
        Args:
            texts (list): List of preprocessed text strings
            
        Returns:
            numpy.ndarray: Document-topic matrix
        """
        if self.topic_model is None or self.topic_vectorizer is None:
            self.logger.error("Topic model not built. Call build_topic_model first.")
            return None
        
        # Transform texts to document-term matrix
        X = self.topic_vectorizer.transform(texts)
        
        # Get topic distributions
        doc_topic_matrix = self.topic_model.transform(X)
        
        return doc_topic_matrix
    
    def train(self, text_data, labels, model_name=None, test_size=0.2):
        """
        Train the sentiment analysis model.
        
        Args:
            text_data (list): List of text strings
            labels (list): List of sentiment labels
            model_name (str, optional): Name for the saved model
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Training metrics
        """
        self.logger.info(f"Training {self.model_type} model on {len(text_data)} text documents")
        
        # Preprocess texts
        processed_texts = self.preprocess_text(text_data)
        
        # Extract features
        X_text = self.extract_features(processed_texts)
        X_sentiment = self.extract_sentiment_features(text_data)
        
        # Combine features
        X = self.combine_features(X_text, X_sentiment)
        y = np.array(labels)
        
        # Build topic model on a subset of data for efficiency
        sample_size = min(5000, len(processed_texts))
        sample_indices = np.random.choice(len(processed_texts), sample_size, replace=False)
        sample_texts = [processed_texts[i] for i in sample_indices]
        
        self.build_topic_model(sample_texts)
        
        # Add topic features
        doc_topics = self.get_document_topics(processed_texts)
        X_combined = self.combine_features(X, doc_topics)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=test_size, random_state=42, stratify=y)
        
        # Build and train model
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                scale_pos_weight=1,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Get class probabilities if available
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_test)
            class_probs = True
        else:
            class_probs = False
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'class_probabilities': class_probs
        }
        
        # Save model
        if model_name:
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'vectorizer': self.vectorizer,
                    'topic_model': self.topic_model,
                    'topic_vectorizer': self.topic_vectorizer,
                    'topics': self.topics,
                    'config': {
                        'model_type': self.model_type,
                        'max_features': self.max_features,
                        'n_topics': self.n_topics
                    }
                }, f)
            
            self.logger.info(f"Model saved to {model_path}")
        
        # Save as pretrained model
        pretrained_path = os.path.join(self.models_dir, "pretrained_sentiment.pkl")
        with open(pretrained_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'topic_model': self.topic_model,
                'topic_vectorizer': self.topic_vectorizer,
                'topics': self.topics
            }, f)
        
        self.logger.info(f"Model training completed. Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def load_model(self, model_path):
        """
        Load a trained model from file.
        
        Args:
            model_path (str): Path to saved model
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.model = saved_data['model']
            self.vectorizer = saved_data['vectorizer']
            self.topic_model = saved_data.get('topic_model')
            self.topic_vectorizer = saved_data.get('topic_vectorizer')
            self.topics = saved_data.get('topics', [])
            
            # Load configuration
            if 'config' in saved_data:
                config = saved_data['config']
                self.model_type = config.get('model_type', self.model_type)
                self.max_features = config.get('max_features', self.max_features)
                self.n_topics = config.get('n_topics', self.n_topics)
            
            self.logger.info(f"Loaded {self.model_type} model from {model_path}")
            return True
                
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            return False
    
    def analyze_sentiment(self, texts, use_pretrained=True):
        """
        Analyze sentiment of text data.
        
        Args:
            texts (list or str): Text data to analyze
            use_pretrained (bool): Whether to use the pretrained model
            
        Returns:
            list or dict: Sentiment analysis results
        """
        self.logger.info("Analyzing sentiment of text data")
        
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Use pretrained analyzer if no model is trained
        if use_pretrained and self.model is None:
            return self._analyze_with_pretrained_model(texts, single_text)
        
        # Use fully trained model if available
        if self.model is not None and self.vectorizer is not None:
            try:
                # Preprocess texts
                processed_texts = self.preprocess_text(texts)
                
                # Extract features
                X_text = self.vectorizer.transform(processed_texts)
                X_sentiment = self.extract_sentiment_features(texts)
                
                # Combine features
                X = self.combine_features(X_text, X_sentiment)
                
                # Add topic features if available
                if self.topic_model is not None and self.topic_vectorizer is not None:
                    doc_topics = self.get_document_topics(processed_texts)
                    X = self.combine_features(X, doc_topics)
                
                # Make predictions
                predictions = self.model.predict(X)
                
                # Get probabilities if available
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(X)
                else:
                    probabilities = None
                
                # Return results
                results = []
                
                for i, pred in enumerate(predictions):
                    result = {
                        'text': texts[i],
                        'sentiment': pred,
                        'label': self._sentiment_to_label(pred)
                    }
                    
                    # Add probabilities if available
                    if probabilities is not None:
                        result['probability'] = probabilities[i].tolist()
                        result['confidence'] = probabilities[i].max()
                    
                    # Extract topics if available
                    if self.topic_model is not None and hasattr(doc_topics, '__len__') and i < len(doc_topics):
                        top_topic_idx = doc_topics[i].argmax()
                        if top_topic_idx < len(self.topics):
                            result['top_topic'] = self.topics[top_topic_idx]['words'][:5]
                    
                    results.append(result)
                
                self.logger.info(f"Sentiment analysis completed for {len(texts)} texts")
                
                return results[0] if single_text else results
                
            except Exception as e:
                self.logger.error(f"Error analyzing sentiment with trained model: {str(e)}")
                # Fall back to pretrained model
                return self._analyze_with_pretrained_model(texts, single_text)
        
        # Fall back to rule-based analysis
        return self._analyze_with_rule_based(texts, single_text)
    
    def _analyze_with_pretrained_model(self, texts, single_text):
        """
        Analyze sentiment using pretrained model.
        
        Args:
            texts (list): List of text strings
            single_text (bool): Whether a single text was provided
            
        Returns:
            list or dict: Sentiment analysis results
        """
        if self.pretrained is not None:
            try:
                # Extract saved components
                model = self.pretrained['model']
                vectorizer = self.pretrained['vectorizer']
                topic_model = self.pretrained.get('topic_model')
                topic_vectorizer = self.pretrained.get('topic_vectorizer')
                
                # Preprocess texts
                processed_texts = self.preprocess_text(texts)
                
                # Extract features
                X_text = vectorizer.transform(processed_texts)
                X_sentiment = self.extract_sentiment_features(texts)
                
                # Combine features
                X = self.combine_features(X_text, X_sentiment)
                
                # Add topic features if available
                if topic_model is not None and topic_vectorizer is not None:
                    doc_terms = topic_vectorizer.transform(processed_texts)
                    doc_topics = topic_model.transform(doc_terms)
                    X = self.combine_features(X, doc_topics)
                
                # Make predictions
                predictions = model.predict(X)
                
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)
                else:
                    probabilities = None
                
                # Return results
                results = []
                
                for i, pred in enumerate(predictions):
                    result = {
                        'text': texts[i],
                        'sentiment': pred,
                        'label': self._sentiment_to_label(pred)
                    }
                    
                    # Add probabilities if available
                    if probabilities is not None:
                        result['probability'] = probabilities[i].tolist()
                        result['confidence'] = probabilities[i].max()
                    
                    results.append(result)
                
                return results[0] if single_text else results
                
            except Exception as e:
                self.logger.error(f"Error analyzing sentiment with pretrained model: {str(e)}")
                # Fall back to rule-based
                return self._analyze_with_rule_based(texts, single_text)
        
        # Fall back to rule-based analysis
        return self._analyze_with_rule_based(texts, single_text)
    
    def _analyze_with_rule_based(self, texts, single_text):
        """
        Analyze sentiment using rule-based approaches.
        
        Args:
            texts (list): List of text strings
            single_text (bool): Whether a single text was provided
            
        Returns:
            list or dict: Sentiment analysis results
        """
        results = []
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                results.append({
                    'text': text,
                    'sentiment': 0,
                    'label': 'neutral',
                    'compound_score': 0,
                    'rule_based': True
                })
                continue
            
            # Use VADER for sentiment analysis
            vader_scores = self.vader.polarity_scores(text)
            compound_score = vader_scores['compound']
            
            # Determine sentiment category
            if compound_score >= 0.05:
                sentiment = 1  # positive
                label = 'positive'
            elif compound_score <= -0.05:
                sentiment = -1  # negative
                label = 'negative'
            else:
                sentiment = 0  # neutral
                label = 'neutral'
            
            # Create result
            result = {
                'text': text,
                'sentiment': sentiment,
                'label': label,
                'compound_score': compound_score,
                'rule_based': True,
                'detailed_scores': {
                    'positive': vader_scores['pos'],
                    'negative': vader_scores['neg'],
                    'neutral': vader_scores['neu']
                }
            }
            
            # Add TextBlob analysis
            blob = TextBlob(text)
            result['textblob_polarity'] = blob.sentiment.polarity
            result['textblob_subjectivity'] = blob.sentiment.subjectivity
            
            results.append(result)
        
        self.logger.info(f"Rule-based sentiment analysis completed for {len(texts)} texts")
        
        return results[0] if single_text else results
    
    def _sentiment_to_label(self, sentiment):
        """
        Convert sentiment value to human-readable label.
        
        Args:
            sentiment: Sentiment value
            
        Returns:
            str: Sentiment label
        """
        if isinstance(sentiment, np.ndarray) and sentiment.size > 0:
            sentiment = sentiment[0]
            
        if sentiment == 1 or sentiment == 'positive':
            return 'positive'
        elif sentiment == -1 or sentiment == 'negative':
            return 'negative'
        else:
            return 'neutral'
    
    def extract_market_sentiment(self, texts, symbol=None, date=None):
        """
        Extract market sentiment specifically for financial texts.
        
        Args:
            texts (list): List of text strings
            symbol (str, optional): Stock symbol to focus on
            date (datetime, optional): Date of the analysis
            
        Returns:
            dict: Market sentiment analysis
        """
        self.logger.info(f"Extracting market sentiment from {len(texts)} texts")
        
        # Analyze sentiment of each text
        sentiment_results = self.analyze_sentiment(texts)
        
        # Calculate overall sentiment metrics
        sentiment_counts = Counter([result['label'] for result in sentiment_results])
        total_texts = len(sentiment_results)
        
        sentiment_distribution = {
            'positive': sentiment_counts['positive'] / total_texts if total_texts > 0 else 0,
            'neutral': sentiment_counts['neutral'] / total_texts if total_texts > 0 else 0,
            'negative': sentiment_counts['negative'] / total_texts if total_texts > 0 else 0
        }
        
        # Calculate sentiment score (-1 to 1)
        sentiment_score = (sentiment_distribution['positive'] - sentiment_distribution['negative'])
        
        # Determine sentiment category
        if sentiment_score >= 0.3:
            sentiment_category = 'very_bullish'
        elif sentiment_score >= 0.1:
            sentiment_category = 'bullish'
        elif sentiment_score > -0.1:
            sentiment_category = 'neutral'
        elif sentiment_score > -0.3:
            sentiment_category = 'bearish'
        else:
            sentiment_category = 'very_bearish'
        
        # Extract key topics if available
        topics = self._extract_key_topics(texts) if self.topic_model is not None else []
        
        # Create result
        result = {
            'symbol': symbol,
            'date': date.strftime('%Y-%m-%d') if date else datetime.now().strftime('%Y-%m-%d'),
            'sentiment_category': sentiment_category,
            'sentiment_score': sentiment_score,
            'sentiment_distribution': sentiment_distribution,
            'topics': topics,
            'text_count': total_texts,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.logger.info(f"Market sentiment: {sentiment_category} with score {sentiment_score:.2f}")
        
        return result
    
    def _extract_key_topics(self, texts, top_n=3):
        """
        Extract key topics from texts.
        
        Args:
            texts (list): List of text strings
            top_n (int): Number of top topics to return
            
        Returns:
            list: Top topics
        """
        try:
            # Preprocess texts
            processed_texts = self.preprocess_text(texts)
            
            # Get document topics
            doc_topics = self.get_document_topics(processed_texts)
            
            if doc_topics is None:
                return []
            
            # Aggregate topic scores across documents
            topic_scores = doc_topics.sum(axis=0)
            
            # Get top topics
            top_topic_indices = topic_scores.argsort()[-top_n:][::-1]
            
            top_topics = []
            for idx in top_topic_indices:
                if idx < len(self.topics):
                    topic = self.topics[idx]
                    top_topics.append({
                        'id': topic['id'],
                        'words': topic['words'][:5],
                        'score': float(topic_scores[idx])
                    })
            
            return top_topics
            
        except Exception as e:
            self.logger.error(f"Error extracting key topics: {str(e)}")
            return []
    
    def visualize_sentiment(self, sentiment_data, symbol=None):
        """
        Visualize sentiment analysis results.
        
        Args:
            sentiment_data (dict or list): Sentiment analysis results
            symbol (str, optional): Stock symbol for labeling
            
        Returns:
            tuple: Figure and axes
        """
        try:
            # Handle both single result and time series
            if isinstance(sentiment_data, dict):
                # Single sentiment result
                is_time_series = False
                sentiment_score = sentiment_data.get('sentiment_score', 0)
                sentiment_distribution = sentiment_data.get('sentiment_distribution', {})
                topics = sentiment_data.get('topics', [])
                
            elif isinstance(sentiment_data, list) and len(sentiment_data) > 0:
                # Time series of sentiment
                is_time_series = True
                
                # Ensure we have dates
                for item in sentiment_data:
                    if 'date' not in item:
                        item['date'] = datetime.now().strftime('%Y-%m-%d')
                
                # Sort by date
                sentiment_data = sorted(sentiment_data, key=lambda x: x['date'])
                
                # Get latest distribution and topics
                sentiment_distribution = sentiment_data[-1].get('sentiment_distribution', {})
                topics = sentiment_data[-1].get('topics', [])
                
            else:
                self.logger.error("Invalid sentiment data for visualization")
                return None, None
            
            # Create figure with subplots
            fig, axs = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot sentiment distribution
            if sentiment_distribution:
                labels = list(sentiment_distribution.keys())
                values = list(sentiment_distribution.values())
                
                colors = ['green', 'gray', 'red']
                axs[0, 0].bar(labels, values, color=colors)
                axs[0, 0].set_title('Sentiment Distribution', fontsize=14)
                axs[0, 0].set_ylim(0, 1)
                
                # Add value labels
                for i, v in enumerate(values):
                    axs[0, 0].text(i, v + 0.02, f"{v:.2f}", ha='center')
            else:
                axs[0, 0].text(0.5, 0.5, "No sentiment distribution data", 
                             ha='center', va='center', transform=axs[0, 0].transAxes)
            
            # Plot sentiment gauge (only for single result)
            if not is_time_series:
                self._plot_sentiment_gauge(axs[0, 1], sentiment_score, symbol)
            else:
                # For time series, plot sentiment trend
                dates = [pd.to_datetime(item['date']) for item in sentiment_data]
                scores = [item.get('sentiment_score', 0) for item in sentiment_data]
                
                axs[0, 1].plot(dates, scores, 'o-', linewidth=2)
                axs[0, 1].set_title('Sentiment Score Trend', fontsize=14)
                axs[0, 1].set_xlabel('Date')
                axs[0, 1].set_ylabel('Sentiment Score')
                axs[0, 1].axhline(y=0, color='gray', linestyle='--')
                axs[0, 1].grid(True)
                
                # Add bands for sentiment categories
                axs[0, 1].axhspan(0.3, 1, alpha=0.2, color='green', label='Very Bullish')
                axs[0, 1].axhspan(0.1, 0.3, alpha=0.1, color='green', label='Bullish')
                axs[0, 1].axhspan(-0.1, 0.1, alpha=0.1, color='gray', label='Neutral')
                axs[0, 1].axhspan(-0.3, -0.1, alpha=0.1, color='red', label='Bearish')
                axs[0, 1].axhspan(-1, -0.3, alpha=0.2, color='red', label='Very Bearish')
                
                # Set y-axis limits
                axs[0, 1].set_ylim(-1, 1)
            
            # Plot top topics word cloud
            if topics:
                self._plot_topic_wordcloud(axs[1, 0], topics)
            else:
                axs[1, 0].text(0.5, 0.5, "No topic data available", 
                             ha='center', va='center', transform=axs[1, 0].transAxes)
            
            # Plot time distribution if time series
            if is_time_series:
                # Convert categorical sentiment to numeric for stacked area chart
                sentiment_categories = []
                for item in sentiment_data:
                    category = item.get('sentiment_category', 'neutral')
                    sentiment_categories.append(category)
                
                # Count categories by date
                dates = [pd.to_datetime(item['date']) for item in sentiment_data]
                df = pd.DataFrame({
                    'date': dates,
                    'category': sentiment_categories
                })
                
                # Group by date and count categories
                pivot_df = pd.crosstab(df['date'], df['category'])
                
                # Fill missing categories
                all_categories = ['very_bearish', 'bearish', 'neutral', 'bullish', 'very_bullish']
                for cat in all_categories:
                    if cat not in pivot_df.columns:
                        pivot_df[cat] = 0
                
                # Plot stacked area
                pivot_df.plot.area(
                    ax=axs[1, 1],
                    stacked=True,
                    color=['darkred', 'red', 'gray', 'green', 'darkgreen']
                )
                
                axs[1, 1].set_title('Sentiment Categories Over Time', fontsize=14)
                axs[1, 1].set_xlabel('Date')
                axs[1, 1].set_ylabel('Count')
                axs[1, 1].legend(title='Sentiment')
                
            else:
                # For single result, show key terms
                self._plot_key_terms(axs[1, 1], sentiment_data)
            
            plt.tight_layout()
            
            # Add title
            if symbol:
                fig.suptitle(f"Market Sentiment Analysis for {symbol}", fontsize=16, y=1.02)
            else:
                fig.suptitle("Market Sentiment Analysis", fontsize=16, y=1.02)
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol_str = f"_{symbol}" if symbol else ""
            plot_file = f"{self.models_dir}/sentiment_analysis{symbol_str}_{timestamp}.png"
            plt.savefig(plot_file)
            
            self.logger.info(f"Sentiment visualization saved to {plot_file}")
            
            return fig, axs
            
        except Exception as e:
            self.logger.error(f"Error visualizing sentiment: {str(e)}")
            return None, None
    
    def _plot_sentiment_gauge(self, ax, sentiment_score, symbol=None):
        """
        Plot a sentiment gauge visualization.
        
        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes
            sentiment_score (float): Sentiment score (-1 to 1)
            symbol (str, optional): Stock symbol for title
        """
        import matplotlib.patches as mpatches
        
        # Define the angle range for the gauge
        theta = np.linspace(-np.pi, np.pi, 100)
        
        # Define the radius
        r = 1.0
        
        # Create the circle for the gauge
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Clear the axis
        ax.clear()
        
        # Draw the gauge background
        ax.fill_between(x, 0, y, where=(theta <= 0), color='red', alpha=0.3)
        ax.fill_between(x, 0, y, where=(theta > 0), color='green', alpha=0.3)
        
        # Add center line
        ax.plot([0, 0], [-r, r], 'k--', alpha=0.3)
        
        # Convert sentiment score to angle
        angle = sentiment_score * np.pi
        
        # Plot the needle
        needle_length = 0.9 * r
        ax.plot([0, needle_length * np.cos(angle)], [0, needle_length * np.sin(angle)], 'k-', linewidth=3)
        
        # Add a center circle
        circle = plt.Circle((0, 0), 0.05, color='black')
        ax.add_artist(circle)
        
        # Add labels
        ax.text(-r*0.8, -r*0.15, "Bearish", fontsize=12)
        ax.text(r*0.8, -r*0.15, "Bullish", ha='right', fontsize=12)
        
        # Add score
        ax.text(0, -r*0.5, f"Score: {sentiment_score:.2f}", 
               ha='center', fontsize=14, fontweight='bold',
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Set title
        if symbol:
            ax.set_title(f"Sentiment Gauge for {symbol}", fontsize=14)
        else:
            ax.set_title("Sentiment Gauge", fontsize=14)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Remove ticks and frame
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    def _plot_topic_wordcloud(self, ax, topics):
        """
        Plot a simple word cloud of topics.
        
        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes
            topics (list): List of topic dictionaries
        """
        try:
            # Extract all words and their weights
            all_words = []
            all_weights = []
            
            for topic in topics:
                words = topic.get('words', [])
                
                # Get weights or assign default weights
                if 'weights' in topic:
                    weights = topic['weights']
                    if len(weights) < len(words):
                        weights.extend([1.0] * (len(words) - len(weights)))
                else:
                    weights = [1.0] * len(words)
                
                all_words.extend(words)
                all_weights.extend(weights)
            
            # Normalize weights
            if all_weights:
                max_weight = max(all_weights)
                all_weights = [w / max_weight * 100 for w in all_weights]
            
            # Create text positions
            np.random.seed(42)
            x = np.random.rand(len(all_words)) * 100
            y = np.random.rand(len(all_words)) * 100
            
            # Plot words
            for i, (word, weight) in enumerate(zip(all_words, all_weights)):
                ax.text(x[i], y[i], word, fontsize=10 + weight / 5, 
                       ha='center', va='center', color=plt.cm.viridis(weight / 100))
            
            ax.set_title('Top Topics', fontsize=14)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            
            # Remove ticks and frame
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
        except Exception as e:
            self.logger.error(f"Error plotting topic wordcloud: {str(e)}")
            ax.text(0.5, 0.5, "Error plotting topics", 
                  ha='center', va='center', transform=ax.transAxes)
    
    def _plot_key_terms(self, ax, sentiment_data):
        """
        Plot key terms from the text.
        
        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes
            sentiment_data (dict): Sentiment data
        """
        try:
            # Try to extract sample texts or most influential words
            if 'texts' in sentiment_data and isinstance(sentiment_data['texts'], list):
                sample_texts = sentiment_data['texts'][:5]  # Take up to 5 sample texts
                
                # Extract most common words
                all_text = ' '.join(sample_texts)
                words = word_tokenize(all_text.lower())
                words = [word for word in words if word not in self.stop_words and len(word) >= 3]
                
                word_counts = Counter(words).most_common(20)
                
                # Plot as horizontal bar chart
                words, counts = zip(*word_counts) if word_counts else ([], [])
                
                y_pos = np.arange(len(words))
                ax.barh(y_pos, counts, align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(words)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_title('Most Common Terms', fontsize=14)
                ax.set_xlabel('Count')
                
            else:
                ax.text(0.5, 0.5, "No key terms data available", 
                      ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            self.logger.error(f"Error plotting key terms: {str(e)}")
            ax.text(0.5, 0.5, "Error plotting key terms", 
                  ha='center', va='center', transform=ax.transAxes)
    
    def get_market_mood(self, texts, symbol=None):
        """
        Get a simple market mood indicator.
        
        Args:
            texts (list): List of text strings
            symbol (str, optional): Stock symbol
            
        Returns:
            str: Market mood indicator
        """
        sentiment_data = self.extract_market_sentiment(texts, symbol)
        sentiment_category = sentiment_data.get('sentiment_category', 'neutral')
        
        return sentiment_category
