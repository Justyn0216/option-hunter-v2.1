"""
Novelty Filter Module

This module provides tools for optimizing trade diversity and avoiding
excessive similarity in trading decisions. It helps reduce correlation
between trades and improve portfolio diversity.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

class NoveltyFilter:
    """
    Filters and optimizes for trade diversity to improve portfolio robustness.
    
    Features:
    - Trade similarity calculation
    - Diversification optimization
    - Clustering of trade opportunities
    - Correlation reduction
    """
    
    def __init__(self, config=None):
        """
        Initialize the NoveltyFilter.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Create necessary directories
        self.results_dir = "data/novelty_filter"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Default parameters
        self.similarity_threshold = self.config.get('similarity_threshold', 0.80)
        self.min_diversity_score = self.config.get('min_diversity_score', 0.3)
        self.cluster_method = self.config.get('cluster_method', 'kmeans')
        
        # Keep track of historical trades
        self.historical_trades = []
        
        self.logger.info("NoveltyFilter initialized")
    
    def extract_trade_features(self, trade):
        """
        Extract numeric features from a trade for similarity analysis.
        
        Args:
            trade (dict): Trade data
            
        Returns:
            dict: Dictionary of numeric features
        """
        features = {}
        
        # Extract basic trade data
        for key in ['delta', 'gamma', 'theta', 'vega']:
            if key in trade.get('greeks', {}):
                features[key] = trade['greeks'][key]
            elif key in trade:
                features[key] = trade[key]
            else:
                features[key] = 0.0
        
        # Extract option data if available
        if 'option_data' in trade:
            option_data = trade['option_data']
            
            # Extract days to expiration
            if 'expiration' in option_data:
                expiration = option_data['expiration']
                if isinstance(expiration, str):
                    exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
                    today = datetime.now().date()
                    days_to_exp = (exp_date - today).days
                    features['days_to_expiration'] = max(0, days_to_exp)
                else:
                    features['days_to_expiration'] = 30  # Default value
            
            # Extract moneyness (relation of strike to underlying price)
            if 'strike' in option_data and 'underlying_price' in trade:
                strike = option_data['strike']
                price = trade['underlying_price']
                features['moneyness'] = strike / price - 1.0
            
            # Option type (call=1, put=-1)
            if 'option_type' in option_data:
                features['option_type'] = 1.0 if option_data['option_type'].lower() == 'call' else -1.0
        
        # Extract additional metrics
        if 'implied_volatility' in trade:
            features['implied_volatility'] = trade['implied_volatility']
        
        if 'model_price' in trade and 'mid_price' in trade:
            features['price_difference'] = (trade['model_price'] - trade['mid_price']) / trade['mid_price']
        
        if 'entry_score' in trade:
            features['entry_score'] = trade['entry_score']
        
        return features
    
    def calculate_trade_similarity(self, trade1, trade2):
        """
        Calculate similarity between two trades.
        
        Args:
            trade1 (dict): First trade
            trade2 (dict): Second trade
            
        Returns:
            float: Similarity score (0-1)
        """
        # Extract features from both trades
        features1 = self.extract_trade_features(trade1)
        features2 = self.extract_trade_features(trade2)
        
        # Find common features
        common_keys = set(features1.keys()) & set(features2.keys())
        
        if not common_keys:
            return 0.0  # No common features
        
        # Convert to vectors
        vector1 = np.array([features1[key] for key in common_keys])
        vector2 = np.array([features2[key] for key in common_keys])
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(vector1 - vector2)
        
        # Convert distance to similarity (0-1)
        # Using exponential decay: similarity = exp(-distance)
        similarity = np.exp(-distance)
        
        return similarity
    
    def calculate_trade_diversity_score(self, candidate_trade, existing_trades):
        """
        Calculate how diverse a candidate trade is compared to existing trades.
        
        Args:
            candidate_trade (dict): Candidate trade data
            existing_trades (list): List of existing trades
            
        Returns:
            float: Diversity score (0-1, higher is more diverse)
        """
        if not existing_trades:
            return 1.0  # First trade is always diverse
        
        # Calculate similarity to each existing trade
        similarities = [
            self.calculate_trade_similarity(candidate_trade, trade)
            for trade in existing_trades
        ]
        
        # Overall diversity is inverse of maximum similarity
        max_similarity = max(similarities)
        diversity_score = 1.0 - max_similarity
        
        return diversity_score
    
    def is_trade_novel(self, candidate_trade, threshold=None):
        """
        Determine if a trade is sufficiently novel compared to historical trades.
        
        Args:
            candidate_trade (dict): Candidate trade data
            threshold (float, optional): Similarity threshold override
            
        Returns:
            bool: True if trade is novel
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        # Calculate diversity score
        diversity_score = self.calculate_trade_diversity_score(
            candidate_trade, self.historical_trades
        )
        
        # Trade is novel if diversity score is above threshold
        is_novel = diversity_score >= (1.0 - threshold)
        
        self.logger.debug(f"Trade diversity score: {diversity_score:.2f}, threshold: {1.0 - threshold:.2f}, is_novel: {is_novel}")
        
        return is_novel
    
    def add_historical_trade(self, trade):
        """
        Add a trade to the historical trade database.
        
        Args:
            trade (dict): Trade data
        """
        self.historical_trades.append(trade)
        self.logger.debug(f"Added trade to history, now {len(self.historical_trades)} historical trades")
    
    def filter_similar_opportunities(self, opportunities, max_similar=1):
        """
        Filter out similar opportunities to ensure diversity.
        
        Args:
            opportunities (list): List of trade opportunities
            max_similar (int): Maximum number of similar trades to allow
            
        Returns:
            list: Filtered list of diverse opportunities
        """
        if not opportunities:
            return []
        
        self.logger.info(f"Filtering {len(opportunities)} opportunities for diversity")
        
        # Extract features for all opportunities
        feature_list = []
        
        for opp in opportunities:
            features = self.extract_trade_features(opp)
            feature_list.append(features)
        
        # Check if we have valid features
        if not feature_list or not feature_list[0]:
            self.logger.warning("No valid features for filtering opportunities")
            return opportunities[:max_similar]  # Return a limited number if no features
        
        # Convert to DataFrame
        feature_keys = feature_list[0].keys()
        feature_matrix = []
        
        for features in feature_list:
            # Extract values, handle missing keys
            values = [features.get(key, 0.0) for key in feature_keys]
            feature_matrix.append(values)
        
        # Convert to numpy array
        feature_matrix = np.array(feature_matrix)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(opportunities), len(opportunities)))
        
        for i in range(len(opportunities)):
            for j in range(i, len(opportunities)):
                sim = np.exp(-np.linalg.norm(scaled_features[i] - scaled_features[j]))
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        # Cluster opportunities if there are enough
        if len(opportunities) > max_similar:
            # Determine optimal number of clusters
            n_clusters = min(len(opportunities), max(2, int(len(opportunities) / max_similar)))
            
            # Use k-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            # Select representative from each cluster
            selected_indices = []
            
            for cluster_id in range(n_clusters):
                # Find indices in this cluster
                cluster_indices = np.where(clusters == cluster_id)[0]
                
                if len(cluster_indices) == 0:
                    continue
                
                # Pick the opportunity closest to cluster center
                cluster_centers = kmeans.cluster_centers_
                distances = cdist(scaled_features[cluster_indices], 
                                 cluster_centers[cluster_id].reshape(1, -1))
                
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)
            
            # Filter opportunities
            filtered_opportunities = [opportunities[i] for i in selected_indices]
        else:
            filtered_opportunities = opportunities
        
        self.logger.info(f"Filtered to {len(filtered_opportunities)} diverse opportunities")
        
        # Visualize clustering if more than 10 opportunities
        if len(opportunities) > 10:
            self._visualize_opportunity_diversity(scaled_features, clusters, selected_indices)
        
        return filtered_opportunities
    
    def _visualize_opportunity_diversity(self, features, clusters, selected_indices):
        """
        Visualize opportunity diversity through clustering.
        
        Args:
            features (numpy.ndarray): Scaled feature matrix
            clusters (numpy.ndarray): Cluster assignments
            selected_indices (list): Indices of selected opportunities
            
        Returns:
            None
        """
        try:
            # Set plotting style
            plt.style.use('seaborn-darkgrid')
            
            # Create figure with subplots
            fig, axs = plt.subplots(1, 2, figsize=(14, 7))
            
            # Use PCA for visualization
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            
            # Plot clusters
            scatter = axs[0].scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='viridis', s=100, alpha=0.8)
            axs[0].set_title('Opportunity Clusters', fontsize=14)
            axs[0].set_xlabel('Principal Component 1')
            axs[0].set_ylabel('Principal Component 2')
            
            # Add legend
            legend1 = axs[0].legend(*scatter.legend_elements(),
                                  title="Clusters")
            axs[0].add_artist(legend1)
            
            # Highlight selected opportunities
            for idx in selected_indices:
                axs[0].scatter(features_2d[idx, 0], features_2d[idx, 1], s=200, facecolors='none', edgecolors='red', linewidth=2)
            
            # Plot similarity matrix
            similarity_matrix = np.zeros((len(features), len(features)))
            
            for i in range(len(features)):
                for j in range(i, len(features)):
                    sim = np.exp(-np.linalg.norm(features[i] - features[j]))
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
            
            im = axs[1].imshow(similarity_matrix, cmap='Blues')
            axs[1].set_title('Similarity Matrix', fontsize=14)
            axs[1].set_xlabel('Opportunity Index')
            axs[1].set_ylabel('Opportunity Index')
            
            # Add colorbar
            plt.colorbar(im, ax=axs[1])
            
            # Highlight selected opportunities
            for idx in selected_indices:
                axs[1].plot(idx, idx, 'rx', markersize=10)
            
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"{self.results_dir}/opportunity_diversity_{timestamp}.png"
            plt.savefig(plot_file)
            
            plt.close(fig)
            
            self.logger.info(f"Opportunity diversity visualization saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing opportunity diversity: {str(e)}")
    
    def optimize_portfolio_diversity(self, current_positions, candidate_trades, target_positions=None):
        """
        Optimize portfolio diversity by selecting the best subset of candidate trades.
        
        Args:
            current_positions (list): List of current portfolio positions
            candidate_trades (list): List of candidate trades
            target_positions (int, optional): Target number of positions
            
        Returns:
            list: Optimized subset of candidate trades
        """
        if not candidate_trades:
            return []
        
        self.logger.info(f"Optimizing diversity for {len(candidate_trades)} candidate trades")
        
        # Set default target positions if not specified
        if target_positions is None:
            if self.config and 'max_open_positions' in self.config.get('trade_parameters', {}):
                target_positions = self.config['trade_parameters']['max_open_positions']
            else:
                target_positions = 10  # Default value
        
        # Calculate how many new positions we can add
        current_count = len(current_positions)
        positions_to_add = max(0, target_positions - current_count)
        
        if positions_to_add == 0:
            self.logger.info("Portfolio already at target capacity, no trades selected")
            return []
        
        if positions_to_add >= len(candidate_trades):
            self.logger.info(f"All {len(candidate_trades)} candidate trades can be added")
            return candidate_trades
        
        # Calculate diversity scores for each candidate
        diversity_scores = []
        
        for candidate in candidate_trades:
            # Calculate diversity relative to current positions
            score = self.calculate_trade_diversity_score(candidate, current_positions)
            diversity_scores.append(score)
        
        # Calculate trade scores (e.g., expected profit) if available
        trade_scores = []
        
        for candidate in candidate_trades:
            if 'entry_score' in candidate:
                score = candidate['entry_score']
            elif 'diff_percent' in candidate:
                score = abs(candidate['diff_percent'])
            else:
                score = 0.5  # Default score
            
            trade_scores.append(score)
        
        # Normalize trade scores to 0-1
        if trade_scores:
            max_score = max(trade_scores)
            min_score = min(trade_scores)
            
            if max_score > min_score:
                trade_scores = [(s - min_score) / (max_score - min_score) for s in trade_scores]
        
        # Calculate combined scores (weighted average of diversity and trade scores)
        diversity_weight = self.config.get('diversity_weight', 0.7)
        combined_scores = [
            diversity_weight * d + (1 - diversity_weight) * t
            for d, t in zip(diversity_scores, trade_scores)
        ]
        
        # Create DataFrame for selection
        selection_df = pd.DataFrame({
            'index': range(len(candidate_trades)),
            'diversity_score': diversity_scores,
            'trade_score': trade_scores,
            'combined_score': combined_scores
        })
        
        # Sort by combined score (descending)
        selection_df = selection_df.sort_values('combined_score', ascending=False)
        
        # Select top positions_to_add trades
        selected_indices = selection_df['index'].values[:positions_to_add]
        
        # Sort back to original order
        selected_indices = sorted(selected_indices)
        
        # Select the corresponding trades
        selected_trades = [candidate_trades[i] for i in selected_indices]
        
        self.logger.info(f"Selected {len(selected_trades)} diverse trades from {len(candidate_trades)} candidates")
        
        # Visualize selection
        self._visualize_trade_selection(diversity_scores, trade_scores, combined_scores, selected_indices)
        
        return selected_trades
    
    def _visualize_trade_selection(self, diversity_scores, trade_scores, combined_scores, selected_indices):
        """
        Visualize trade selection based on diversity and trade scores.
        
        Args:
            diversity_scores (list): List of diversity scores
            trade_scores (list): List of trade scores
            combined_scores (list): List of combined scores
            selected_indices (list): Indices of selected trades
            
        Returns:
            None
        """
        try:
            # Set plotting style
            plt.style.use('seaborn-darkgrid')
            
            # Create figure with subplots
            fig, axs = plt.subplots(2, 1, figsize=(10, 12))
            
            # Plot scores
            indices = range(len(diversity_scores))
            
            axs[0].plot(indices, diversity_scores, 'bo-', label='Diversity Score')
            axs[0].plot(indices, trade_scores, 'go-', label='Trade Score')
            axs[0].plot(indices, combined_scores, 'ro-', label='Combined Score')
            
            # Highlight selected trades
            for idx in selected_indices:
                axs[0].axvline(x=idx, color='gray', linestyle='--', alpha=0.5)
                axs[0].plot(idx, diversity_scores[idx], 'bo', markersize=10)
                axs[0].plot(idx, trade_scores[idx], 'go', markersize=10)
                axs[0].plot(idx, combined_scores[idx], 'ro', markersize=10)
            
            axs[0].set_title('Trade Selection Scores', fontsize=14)
            axs[0].set_xlabel('Trade Index')
            axs[0].set_ylabel('Score')
            axs[0].legend()
            axs[0].set_ylim(0, 1.1)
            
            # Plot scatter of diversity vs trade score
            scatter = axs[1].scatter(diversity_scores, trade_scores, c=combined_scores, s=100, cmap='viridis')
            
            # Highlight selected trades
            for idx in selected_indices:
                axs[1].scatter(diversity_scores[idx], trade_scores[idx], s=200, facecolors='none', 
                             edgecolors='red', linewidth=2)
            
            axs[1].set_title('Diversity vs Trade Score', fontsize=14)
            axs[1].set_xlabel('Diversity Score')
            axs[1].set_ylabel('Trade Score')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=axs[1])
            cbar.set_label('Combined Score')
            
            plt.tight_layout()
            
            # Save figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = f"{self.results_dir}/trade_selection_{timestamp}.png"
            plt.savefig(plot_file)
            
            plt.close(fig)
            
            self.logger.info(f"Trade selection visualization saved to {plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing trade selection: {str(e)}")
    
    def calculate_portfolio_diversity(self, positions):
        """
        Calculate overall portfolio diversity score.
        
        Args:
            positions (list): List of current portfolio positions
            
        Returns:
            float: Portfolio diversity score (0-1)
        """
        if len(positions) <= 1:
            return 1.0  # Single position is diverse by definition
        
        # Extract features for all positions
        feature_list = []
        
        for position in positions:
            features = self.extract_trade_features(position)
            feature_list.append(features)
        
        # Check if we have valid features
        if not feature_list or not feature_list[0]:
            self.logger.warning("No valid features for portfolio diversity calculation")
            return 0.5  # Default value
        
        # Convert to DataFrame
        feature_keys = set()
        for features in feature_list:
            feature_keys.update(features.keys())
        
        feature_matrix = []
        for features in feature_list:
            # Extract values, handle missing keys
            values = [features.get(key, 0.0) for key in feature_keys]
            feature_matrix.append(values)
        
        # Convert to numpy array
        feature_matrix = np.array(feature_matrix)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        # Calculate average pairwise distance
        n = len(scaled_features)
        total_distance = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                distance = np.linalg.norm(scaled_features[i] - scaled_features[j])
                total_distance += distance
                count += 1
        
        average_distance = total_distance / count if count > 0 else 0.0
        
        # Convert to diversity score (0-1)
        # Using 1 - exp(-distance) which gives 0 for identical positions and approaches 1 as distance increases
        diversity_score = 1.0 - np.exp(-average_distance)
        
        return diversity_score
    
    def is_portfolio_diversified(self, positions, threshold=None):
        """
        Check if the portfolio is sufficiently diversified.
        
        Args:
            positions (list): List of current portfolio positions
            threshold (float, optional): Diversity threshold override
            
        Returns:
            bool: True if portfolio is diversified
        """
        if threshold is None:
            threshold = self.min_diversity_score
        
        diversity_score = self.calculate_portfolio_diversity(positions)
        
        is_diversified = diversity_score >= threshold
        
        self.logger.debug(f"Portfolio diversity score: {diversity_score:.2f}, " +
                         f"threshold: {threshold:.2f}, is_diversified: {is_diversified}")
        
        return is_diversified
    
    def suggest_diversity_improvements(self, positions, candidate_trades=None):
        """
        Suggest improvements to increase portfolio diversity.
        
        Args:
            positions (list): List of current portfolio positions
            candidate_trades (list, optional): List of candidate replacement trades
            
        Returns:
            dict: Suggested improvements
        """
        if len(positions) <= 1:
            return {"status": "insufficient_positions", "message": "Need more positions for diversity analysis"}
        
        self.logger.info(f"Analyzing diversity improvements for {len(positions)} positions")
        
        # Calculate current diversity
        current_diversity = self.calculate_portfolio_diversity(positions)
        
        # Check if diverse enough
        if current_diversity >= self.min_diversity_score:
            return {
                "status": "sufficiently_diverse",
                "diversity_score": current_diversity,
                "message": f"Portfolio is already sufficiently diverse (score: {current_diversity:.2f})"
            }
        
        # Calculate similarity matrix
        n = len(positions)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                sim = self.calculate_trade_similarity(positions[i], positions[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        # Find most similar pair
        most_similar_pair = None
        highest_similarity = 0.0
        
        for i in range(n):
            for j in range(i+1, n):
                if similarity_matrix[i, j] > highest_similarity:
                    highest_similarity = similarity_matrix[i, j]
                    most_similar_pair = (i, j)
        
        if most_similar_pair is None:
            return {
                "status": "error",
                "message": "Could not identify similar positions"
            }
        
        # Determine which position to potentially replace
        i, j = most_similar_pair
        
        # Choose position with worse metrics (if available)
        replace_idx = i  # Default
        
        if 'entry_score' in positions[i] and 'entry_score' in positions[j]:
            if positions[i]['entry_score'] < positions[j]['entry_score']:
                replace_idx = i
            else:
                replace_idx = j
        elif 'unrealized_pnl' in positions[i] and 'unrealized_pnl' in positions[j]:
            if positions[i]['unrealized_pnl'] < positions[j]['unrealized_pnl']:
                replace_idx = i
            else:
                replace_idx = j
        
        # Prepare result
        result = {
            "status": "improvement_suggested",
            "diversity_score": current_diversity,
            "most_similar_pair": most_similar_pair,
            "similarity": highest_similarity,
            "suggested_replace_index": replace_idx,
            "position_to_replace": positions[replace_idx]
        }
        
        # If candidate trades are provided, suggest replacement
        if candidate_trades:
            # Calculate diversity score for each candidate
            replacement_scores = []
            
            for candidate in candidate_trades:
                # Create a copy of positions with the candidate replacing the chosen position
                new_positions = positions.copy()
                new_positions[replace_idx] = candidate
                
                # Calculate new diversity
                new_diversity = self.calculate_portfolio_diversity(new_positions)
                improvement = new_diversity - current_diversity
                
                replacement_scores.append({
                    "candidate": candidate,
                    "new_diversity": new_diversity,
                    "improvement": improvement
                })
            
            # Sort by improvement (descending)
            replacement_scores.sort(key=lambda x: x["improvement"], reverse=True)
            
            # Add top suggestions to result
            result["replacement_suggestions"] = replacement_scores[:3]
        
        return result
