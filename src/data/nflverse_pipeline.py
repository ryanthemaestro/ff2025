#!/usr/bin/env python3
"""
NFLverse Data Pipeline for AI Model Training
Fetches real NFL performance data for fantasy football predictions
"""
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime
import os

class NFLDataPipeline:
    def __init__(self):
        self.current_year = datetime.now().year
        self.years = [self.current_year - 1, self.current_year - 2, self.current_year - 3]  # Last 3 years
        
    def fetch_player_stats(self):
        """Fetch player statistics from NFLverse"""
        print("📊 Fetching NFLverse player statistics...")
        
        try:
            # Fetch player stats for multiple years
            stats_df = nfl.import_seasonal_data(self.years, s_type='REG')
            
            print(f"✅ Loaded {len(stats_df)} player-season records")
            return stats_df
            
        except Exception as e:
            print(f"❌ Error fetching player stats: {e}")
            return pd.DataFrame()
    
    def fetch_schedule_data(self):
        """Fetch schedule data for strength of schedule calculations"""
        print("📅 Fetching schedule data...")
        
        try:
            schedule_df = nfl.import_schedules(self.years)
            print(f"✅ Loaded {len(schedule_df)} games")
            return schedule_df
            
        except Exception as e:
            print(f"❌ Error fetching schedule: {e}")
            return pd.DataFrame()
    
    def process_fantasy_points(self, stats_df):
        """Calculate fantasy points from NFL stats"""
        print("🏈 Calculating fantasy points...")
        
        # Standard fantasy scoring
        stats_df['fantasy_points'] = (
            stats_df.get('passing_yards', 0) * 0.04 +
            stats_df.get('passing_tds', 0) * 4 +
            stats_df.get('interceptions', 0) * -2 +
            stats_df.get('rushing_yards', 0) * 0.1 +
            stats_df.get('rushing_tds', 0) * 6 +
            stats_df.get('receiving_yards', 0) * 0.1 +
            stats_df.get('receiving_tds', 0) * 6 +
            stats_df.get('receptions', 0) * 1  # PPR scoring
        )
        
        return stats_df
    
    def create_training_features(self, stats_df):
        """Create features for machine learning model"""
        print("🔧 Creating training features...")
        
        # Sort by player and season
        stats_df = stats_df.sort_values(['player_display_name', 'season'])
        
        # Create rolling averages and trends
        feature_cols = [
            'passing_yards', 'passing_tds', 'interceptions',
            'rushing_yards', 'rushing_tds', 'receiving_yards', 
            'receiving_tds', 'receptions', 'fantasy_points'
        ]
        
        for col in feature_cols:
            if col in stats_df.columns:
                # Previous season performance
                stats_df[f'{col}_prev'] = stats_df.groupby('player_display_name')[col].shift(1)
                
                # 2-year rolling average
                stats_df[f'{col}_avg_2yr'] = stats_df.groupby('player_display_name')[col].rolling(window=2, min_periods=1).mean().reset_index(0, drop=True)
        
        # Age calculation (approximate)
        current_year = datetime.now().year
        stats_df['age_approx'] = current_year - stats_df.get('season', current_year) + 25  # Rough estimate
        
        # Games played consistency
        stats_df['games_played_prev'] = stats_df.groupby('player_display_name')['games'].shift(1)
        
        return stats_df
    
    def save_training_data(self, df, filename=None):
        """Save processed data for model training"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"../../data/nflverse_training_data_{timestamp}.csv"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"💾 Saved training data to {filename}")
        return filename
    
    def run_pipeline(self):
        """Run the complete data pipeline"""
        print("🚀 Starting NFLverse Data Pipeline...")
        
        # Fetch data
        stats_df = self.fetch_player_stats()
        if stats_df.empty:
            print("❌ No stats data available")
            return None
        
        # Process fantasy points
        stats_df = self.process_fantasy_points(stats_df)
        
        # Create features
        stats_df = self.create_training_features(stats_df)
        
        # Filter to relevant positions
        fantasy_positions = ['QB', 'RB', 'WR', 'TE']
        stats_df = stats_df[stats_df['position'].isin(fantasy_positions)]
        
        # Save processed data
        filename = self.save_training_data(stats_df)
        
        print(f"✅ Pipeline complete! {len(stats_df)} records processed")
        print(f"   Positions: {stats_df['position'].value_counts().to_dict()}")
        
        return filename

def main():
    """Main execution function"""
    pipeline = NFLDataPipeline()
    result = pipeline.run_pipeline()
    
    if result:
        print(f"\n🎉 Training data ready: {result}")
        print("\n📋 Next steps:")
        print("1. Use this data to train your AI model")
        print("2. Combine with ADP data for draft recommendations")
    else:
        print("\n❌ Pipeline failed")

if __name__ == "__main__":
    main() 