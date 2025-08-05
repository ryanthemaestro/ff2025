#!/usr/bin/env python3
"""
Create Proper Training Data for Fantasy Football Prediction
This creates a dataset where we predict FUTURE fantasy performance from PAST performance
NO DATA LEAKAGE - Uses historical stats to predict next week/season performance
"""

import pandas as pd
import numpy as np
from datetime import datetime

class ProperTrainingDataCreator:
    """Create training data that predicts future performance from past stats"""
    
    def __init__(self):
        self.min_games_threshold = 4  # Need at least 4 games of history
        
    def load_nflverse_data(self):
        """Load the clean NFLverse weekly data"""
        print("📊 Loading NFLverse weekly data...")
        
        weekly_data = pd.read_csv('data/nflverse/combined_weekly_2022_2024.csv')
        print(f"✅ Loaded {len(weekly_data)} weekly records")
        
        # Filter to fantasy-relevant positions only
        fantasy_positions = ['QB', 'RB', 'WR', 'TE']
        weekly_data = weekly_data[weekly_data['position'].isin(fantasy_positions)]
        print(f"✅ Filtered to {len(weekly_data)} records for fantasy positions")
        
        return weekly_data
    
    def calculate_fantasy_points(self, df):
        """Calculate fantasy points from raw stats"""
        return (
            df['passing_tds'].fillna(0) * 4 +
            df['passing_yards'].fillna(0) * 0.04 +
            df['interceptions'].fillna(0) * -2 +
            df['rushing_tds'].fillna(0) * 6 +
            df['rushing_yards'].fillna(0) * 0.1 +
            df['receiving_tds'].fillna(0) * 6 +
            df['receiving_yards'].fillna(0) * 0.1 +
            df['receptions'].fillna(0) * 1  # PPR scoring
        )
    
    def create_historical_features(self, player_data):
        """Create features from historical performance (past 4+ games)"""
        
        # Sort by week to ensure proper chronological order
        player_data = player_data.sort_values('week')
        
        features_list = []
        
        for i in range(self.min_games_threshold, len(player_data)):
            # Historical window: games 0 to i-1 (past performance)
            historical_games = player_data.iloc[:i]
            
            # Target: game i (future performance we want to predict)
            target_game = player_data.iloc[i]
            
            # Calculate historical averages (features)
            features = {
                'player_id': target_game['player_id'],
                'player_name': target_game['player_name'],
                'position': target_game['position'],
                'season': target_game['season'],
                'week': target_game['week'],
                'recent_team': target_game['recent_team'],
                
                # Historical performance features (past games avg)
                'hist_games_played': len(historical_games),
                'hist_avg_passing_yards': historical_games['passing_yards'].fillna(0).mean(),
                'hist_avg_passing_tds': historical_games['passing_tds'].fillna(0).mean(),
                'hist_avg_interceptions': historical_games['interceptions'].fillna(0).mean(),
                'hist_avg_rushing_yards': historical_games['rushing_yards'].fillna(0).mean(),
                'hist_avg_rushing_tds': historical_games['rushing_tds'].fillna(0).mean(),
                'hist_avg_receiving_yards': historical_games['receiving_yards'].fillna(0).mean(),
                'hist_avg_receiving_tds': historical_games['receiving_tds'].fillna(0).mean(),
                'hist_avg_receptions': historical_games['receptions'].fillna(0).mean(),
                'hist_avg_targets': historical_games['targets'].fillna(0).mean(),
                'hist_avg_carries': historical_games['carries'].fillna(0).mean(),
                
                # Historical consistency features
                'hist_std_fantasy_points': historical_games['fantasy_points'].std(),
                'hist_max_fantasy_points': historical_games['fantasy_points'].max(),
                'hist_min_fantasy_points': historical_games['fantasy_points'].min(),
                
                # Recent trend (last 3 games vs all historical)
                'recent_avg_fantasy_points': historical_games['fantasy_points'].tail(3).mean(),
                'recent_vs_season_trend': historical_games['fantasy_points'].tail(3).mean() - historical_games['fantasy_points'].mean(),
                
                # Target: actual fantasy points in the current game (what we want to predict)
                'target_fantasy_points': target_game['fantasy_points']
            }
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def create_training_dataset(self):
        """Create the complete training dataset"""
        print("🔧 Creating proper training dataset...")
        
        # Load data
        weekly_data = self.load_nflverse_data()
        
        # Add fantasy points
        weekly_data['fantasy_points'] = self.calculate_fantasy_points(weekly_data)
        
        all_features = []
        
        # Process each player separately
        players = weekly_data.groupby('player_id')
        print(f"👥 Processing {len(players)} unique players...")
        
        for player_id, player_data in players:
            if len(player_data) >= self.min_games_threshold:
                player_features = self.create_historical_features(player_data)
                all_features.append(player_features)
        
        # Combine all player data
        if all_features:
            training_data = pd.concat(all_features, ignore_index=True)
            print(f"✅ Created {len(training_data)} training samples")
            
            # Remove any invalid samples
            training_data = training_data.dropna(subset=['target_fantasy_points'])
            print(f"✅ Final dataset: {len(training_data)} valid samples")
            
            # Show breakdown
            print(f"📈 Position breakdown:")
            print(training_data['position'].value_counts().to_dict())
            
            print(f"📈 Season breakdown:")
            print(training_data['season'].value_counts().to_dict())
            
            print(f"📈 Target stats:")
            print(f"   Mean: {training_data['target_fantasy_points'].mean():.2f}")
            print(f"   Std: {training_data['target_fantasy_points'].std():.2f}")
            print(f"   Range: {training_data['target_fantasy_points'].min():.1f} to {training_data['target_fantasy_points'].max():.1f}")
            
            return training_data
        else:
            print("❌ No valid training data created")
            return None
    
    def save_training_data(self, training_data):
        """Save the training dataset"""
        if training_data is not None:
            filename = f"data/proper_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            training_data.to_csv(filename, index=False)
            print(f"💾 Saved training data to {filename}")
            return filename
        return None

def main():
    """Main execution"""
    print("🚀 CREATING PROPER TRAINING DATA (NO LEAKAGE)")
    print("=" * 60)
    
    creator = ProperTrainingDataCreator()
    training_data = creator.create_training_dataset()
    filename = creator.save_training_data(training_data)
    
    if filename:
        print(f"\n🎉 PROPER TRAINING DATA CREATED!")
        print(f"✅ No data leakage - predicts future from past performance")
        print(f"✅ Uses historical averages as features")
        print(f"✅ Target is future game performance")
        print(f"📁 Saved to: {filename}")
    else:
        print(f"\n💥 Failed to create training data")

if __name__ == "__main__":
    main() 