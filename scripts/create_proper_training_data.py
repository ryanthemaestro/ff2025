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
        
        # Sort by season and week to ensure proper chronological order across seasons
        player_data = player_data.sort_values(['season', 'week'])
        
        features_list = []
        
        for i in range(self.min_games_threshold, len(player_data)):
            # Historical window: games 0 to i-1 (past performance)
            historical_games = player_data.iloc[:i]
            
            # Target: game i (future performance we want to predict)
            target_game = player_data.iloc[i]
            
            # Recent trend (last 3 games vs all historical)
            recent_avg_fantasy_points = historical_games['fantasy_points'].tail(3).mean()
            recent_vs_season_trend = recent_avg_fantasy_points - historical_games['fantasy_points'].mean()

            # Recent 5-game stats (robust to short windows)
            recent5 = historical_games.tail(5)
            recent5_avg_fantasy_points = recent5['fantasy_points'].mean()
            recent5_std_fantasy_points = recent5['fantasy_points'].std()

            # Recency-weighted average fantasy points across seasons
            weighted_avg = 0
            seasons = historical_games['season'].unique()
            if len(seasons) > 0:
                seasons_sorted = sorted(seasons, reverse=True)  # Most recent first
                weights = [0.5, 0.3, 0.2][:len(seasons_sorted)]  # Up to 3 seasons
                if len(seasons_sorted) > len(weights):
                    weights += [0.1] * (len(seasons_sorted) - len(weights))
                weighted_sum = 0
                total_weight = 0
                for season, weight in zip(seasons_sorted, weights):
                    season_data = historical_games[historical_games['season'] == season]
                    season_avg = season_data['fantasy_points'].mean()
                    weighted_sum += season_avg * weight * len(season_data)
                    total_weight += weight * len(season_data)
                weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0

            # Efficiency features (use sums to be robust to NaNs)
            sum_rec_yards = historical_games['receiving_yards'].fillna(0).sum()
            sum_targets = historical_games['targets'].fillna(0).sum()
            sum_carries = historical_games['carries'].fillna(0).sum()
            sum_rush_yards = historical_games['rushing_yards'].fillna(0).sum()
            sum_receptions = historical_games['receptions'].fillna(0).sum()
            sum_rec_tds = historical_games['receiving_tds'].fillna(0).sum()
            sum_rush_tds = historical_games['rushing_tds'].fillna(0).sum()
            sum_pass_tds = historical_games['passing_tds'].fillna(0).sum()
            sum_ints = historical_games['interceptions'].fillna(0).sum()

            eff_yards_per_target = sum_rec_yards / max(1.0, sum_targets)
            eff_yards_per_carry = sum_rush_yards / max(1.0, sum_carries)
            rate_receiving_td = sum_rec_tds / max(1.0, sum_receptions)
            rate_rushing_td = sum_rush_tds / max(1.0, sum_carries)
            rate_pass_td_to_int = (sum_pass_tds + 1.0) / (sum_ints + 1.0)
            
            # Target: actual fantasy points in the current game (what we want to predict)
            target_fantasy_points = target_game['fantasy_points']
            
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
                
                # Trend and recency features
                'recent_avg_fantasy_points': recent_avg_fantasy_points,
                'recent_vs_season_trend': recent_vs_season_trend,
                'recent5_avg_fantasy_points': recent5_avg_fantasy_points,
                'recent5_std_fantasy_points': recent5_std_fantasy_points,
                'weighted_avg_fantasy_points': weighted_avg,

                # Efficiency features
                'eff_yards_per_target': eff_yards_per_target,
                'eff_yards_per_carry': eff_yards_per_carry,
                'rate_receiving_td': rate_receiving_td,
                'rate_rushing_td': rate_rushing_td,
                'rate_pass_td_to_int': rate_pass_td_to_int,

                # Target
                'target_fantasy_points': target_fantasy_points
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