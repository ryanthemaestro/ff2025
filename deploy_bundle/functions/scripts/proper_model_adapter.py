#!/usr/bin/env python3
"""
Proper Model Adapter
Converts ADP player data into historical features for our leak-free AI model
"""

import pandas as pd
import numpy as np
import joblib
import os

class ProperModelAdapter:
    """Adapter to use proper AI model with ADP data"""
    
    def __init__(self):
        self.model = None
        self.model_features = [
            'hist_games_played',
            'hist_avg_passing_yards', 'hist_avg_passing_tds', 'hist_avg_interceptions',
            'hist_avg_rushing_yards', 'hist_avg_rushing_tds',
            'hist_avg_receiving_yards', 'hist_avg_receiving_tds', 
            'hist_avg_receptions', 'hist_avg_targets', 'hist_avg_carries',
            'hist_std_fantasy_points', 'hist_max_fantasy_points', 'hist_min_fantasy_points',
            'recent_avg_fantasy_points', 'recent_vs_season_trend',
            'is_qb', 'is_rb', 'is_wr', 'is_te',
            'season_2022', 'season_2023', 'season_2024',
            'week', 'early_season', 'mid_season', 'late_season'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the proper CatBoost model"""
        try:
            model_path = '../models/proper_fantasy_model.pkl' if not os.path.exists('models/proper_fantasy_model.pkl') else 'models/proper_fantasy_model.pkl'
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print("✅ Loaded proper AI model (no data leakage)")
                return True
            else:
                print(f"❌ Proper model not found at {model_path}")
                return False
        except Exception as e:
            print(f"❌ Error loading proper model: {e}")
            self.model = None
            return False
    
    def create_mock_historical_features(self, players_df):
        """
        Create mock historical features from ADP data
        Since we don't have real historical data for 2025 ADP players,
        we'll create reasonable estimates based on position and ADP ranking
        """
        features_df = players_df.copy()
        
        # Position encoding
        features_df['is_qb'] = (features_df['position'] == 'QB').astype(int)
        features_df['is_rb'] = (features_df['position'] == 'RB').astype(int)
        features_df['is_wr'] = (features_df['position'] == 'WR').astype(int)
        features_df['is_te'] = (features_df['position'] == 'TE').astype(int)
        
        # Season encoding (assume current season 2024)
        features_df['season_2022'] = 0
        features_df['season_2023'] = 0
        features_df['season_2024'] = 1
        
        # Week features (assume mid-season for draft)
        features_df['week'] = 8
        features_df['early_season'] = 0
        features_df['mid_season'] = 1
        features_df['late_season'] = 0
        
        # Mock historical games played (veteran vs rookie estimates)
        features_df['hist_games_played'] = features_df.apply(
            lambda row: 12 if 'rookie' not in str(row.get('name', '')).lower() else 0, axis=1
        )
        
        # Create position-based historical stat estimates from ADP ranking
        for _, row in features_df.iterrows():
            adp = row.get('adp_rank', 100)
            position = row['position']
            
            # Better ADP = higher estimated historical performance
            # Scale inversely with ADP (lower ADP rank = higher stats)
            performance_factor = max(0.1, (200 - adp) / 200)
            
            if position == 'QB':
                features_df.loc[_, 'hist_avg_passing_yards'] = 220 * performance_factor
                features_df.loc[_, 'hist_avg_passing_tds'] = 1.8 * performance_factor
                features_df.loc[_, 'hist_avg_interceptions'] = 0.8 * (1 - performance_factor * 0.5)
                features_df.loc[_, 'hist_avg_rushing_yards'] = 15 * performance_factor
                features_df.loc[_, 'hist_avg_rushing_tds'] = 0.3 * performance_factor
                features_df.loc[_, 'recent_avg_fantasy_points'] = 18 * performance_factor
                
            elif position == 'RB':
                features_df.loc[_, 'hist_avg_rushing_yards'] = 75 * performance_factor
                features_df.loc[_, 'hist_avg_rushing_tds'] = 0.6 * performance_factor
                features_df.loc[_, 'hist_avg_carries'] = 14 * performance_factor
                features_df.loc[_, 'hist_avg_receptions'] = 3 * performance_factor
                features_df.loc[_, 'hist_avg_receiving_yards'] = 25 * performance_factor
                features_df.loc[_, 'hist_avg_targets'] = 4 * performance_factor
                features_df.loc[_, 'recent_avg_fantasy_points'] = 12 * performance_factor
                
            elif position == 'WR':
                features_df.loc[_, 'hist_avg_receptions'] = 4.5 * performance_factor
                features_df.loc[_, 'hist_avg_receiving_yards'] = 65 * performance_factor
                features_df.loc[_, 'hist_avg_receiving_tds'] = 0.5 * performance_factor
                features_df.loc[_, 'hist_avg_targets'] = 7 * performance_factor
                features_df.loc[_, 'recent_avg_fantasy_points'] = 11 * performance_factor
                
            elif position == 'TE':
                features_df.loc[_, 'hist_avg_receptions'] = 3.2 * performance_factor
                features_df.loc[_, 'hist_avg_receiving_yards'] = 40 * performance_factor
                features_df.loc[_, 'hist_avg_receiving_tds'] = 0.4 * performance_factor
                features_df.loc[_, 'hist_avg_targets'] = 5 * performance_factor
                features_df.loc[_, 'recent_avg_fantasy_points'] = 8 * performance_factor
        
        # Fill in remaining features with defaults
        stat_columns = [
            'hist_avg_passing_yards', 'hist_avg_passing_tds', 'hist_avg_interceptions',
            'hist_avg_rushing_yards', 'hist_avg_rushing_tds',
            'hist_avg_receiving_yards', 'hist_avg_receiving_tds', 
            'hist_avg_receptions', 'hist_avg_targets', 'hist_avg_carries'
        ]
        
        for col in stat_columns:
            if col not in features_df.columns:
                features_df[col] = 0
            features_df[col] = features_df[col].fillna(0)
        
        # Historical consistency features (mock based on ADP)
        features_df['hist_std_fantasy_points'] = features_df['recent_avg_fantasy_points'] * 0.4
        features_df['hist_max_fantasy_points'] = features_df['recent_avg_fantasy_points'] * 1.8
        features_df['hist_min_fantasy_points'] = features_df['recent_avg_fantasy_points'] * 0.2
        features_df['recent_vs_season_trend'] = 0  # No trend data available
        
        return features_df
    
    def get_ai_predictions(self, players_df):
        """Get AI predictions for a list of players"""
        if self.model is None:
            print("❌ Model not loaded, cannot make predictions")
            return None
        
        try:
            # Create features
            features_df = self.create_mock_historical_features(players_df)
            
            # Ensure all required features are present
            for feature in self.model_features:
                if feature not in features_df.columns:
                    features_df[feature] = 0
            
            # Make predictions
            X = features_df[self.model_features]
            predictions = self.model.predict(X)
            
            # Add predictions to the dataframe
            result_df = players_df.copy()
            result_df['ai_prediction'] = predictions
            
            print(f"✅ Generated AI predictions for {len(result_df)} players")
            return result_df
            
        except Exception as e:
            print(f"❌ Error generating AI predictions: {e}")
            return None
    
    def is_available(self):
        """Check if the model is available for predictions"""
        return self.model is not None

# Global instance
_adapter = None

def get_adapter():
    """Get the global adapter instance"""
    global _adapter
    if _adapter is None:
        _adapter = ProperModelAdapter()
    return _adapter

def predict_players(players_df):
    """Simple function to get predictions for players"""
    adapter = get_adapter()
    return adapter.get_ai_predictions(players_df)

def is_model_available():
    """Check if the AI model is available"""
    adapter = get_adapter()
    return adapter.is_available() 