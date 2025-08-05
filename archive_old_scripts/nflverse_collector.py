#!/usr/bin/env python3
"""
Comprehensive nflverse Data Collector
Get REAL NFL statistics and fantasy points to replace synthetic data

Based on: https://github.com/nflverse & https://pypi.org/project/nfl-data-py/
REAL NFL STATISTICS ONLY - NO SYNTHETIC DATA
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
from datetime import datetime
import os

class NFLverseCollector:
    """Comprehensive NFL data collector using nflverse ecosystem"""
    
    def __init__(self):
        self.current_year = 2024
        
    def collect_all_nfl_data(self):
        """Collect comprehensive NFL data for fantasy football"""
        print("🏈 NFLVERSE COMPREHENSIVE DATA COLLECTION")
        print("=" * 60)
        
        try:
            # Step 1: Get current rosters
            print(f"\n📋 Collecting 2024 NFL Rosters...")
            rosters = nfl.import_seasonal_rosters([self.current_year])
            print(f"✅ Got {len(rosters)} roster entries")
            
            # Step 2: Get weekly fantasy data (multiple years for better stats)
            print(f"\n🏆 Collecting Weekly Fantasy Data (2022-2024)...")
            weekly_data = nfl.import_weekly_data([2022, 2023, 2024])
            print(f"✅ Got {len(weekly_data)} weekly records across 3 years")
            
            # Step 3: Get seasonal data with market shares
            print(f"\n📈 Collecting Seasonal Data with Market Shares...")
            seasonal_data = nfl.import_seasonal_data([2022, 2023, 2024])
            print(f"✅ Got {len(seasonal_data)} seasonal records")
            
            # Step 4: Process and combine data
            print(f"\n🔧 Processing and Combining Data...")
            processed_data = self.process_fantasy_data(rosters, weekly_data, seasonal_data)
            
            # Step 5: Save processed data
            print(f"\n💾 Saving Processed Data...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"nflverse_fantasy_data_{timestamp}.csv"
            processed_data.to_csv(filename, index=False)
            print(f"✅ Saved {len(processed_data)} players to {filename}")
            
            return processed_data, filename
            
        except Exception as e:
            print(f"❌ Error collecting nflverse data: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def process_fantasy_data(self, rosters, weekly_data, seasonal_data):
        """Process and combine nflverse data into fantasy-ready format"""
        print("   🔄 Processing fantasy data...")
        
        # Filter to 2024 rosters for current players
        current_rosters = rosters[rosters['season'] == self.current_year].copy()
        
        # Calculate 2024 fantasy stats from weekly data
        weekly_2024 = weekly_data[weekly_data['season'] == self.current_year].copy()
        
        # Aggregate weekly data to get season totals
        fantasy_stats = weekly_2024.groupby('player_id').agg({
            'player_name': 'first',
            'position': 'first', 
            'recent_team': 'first',
            'fantasy_points': 'sum',
            'fantasy_points_ppr': 'sum',
            'passing_yards': 'sum',
            'passing_tds': 'sum',
            'rushing_yards': 'sum', 
            'rushing_tds': 'sum',
            'receiving_yards': 'sum',
            'receiving_tds': 'sum',
            'receptions': 'sum',
            'targets': 'sum',
            'week': 'count'  # Games played
        }).reset_index()
        
        # Calculate averages and projections
        fantasy_stats['games_played'] = fantasy_stats['week']
        fantasy_stats['avg_fantasy_points'] = fantasy_stats['fantasy_points'] / fantasy_stats['games_played']
        fantasy_stats['avg_fantasy_points_ppr'] = fantasy_stats['fantasy_points_ppr'] / fantasy_stats['games_played']
        
        # Project for full season (17 games)
        fantasy_stats['projected_fantasy_points'] = fantasy_stats['avg_fantasy_points'] * 17
        fantasy_stats['projected_fantasy_points_ppr'] = fantasy_stats['avg_fantasy_points_ppr'] * 17
        
        # Calculate multi-year averages for consistency
        multi_year_stats = weekly_data.groupby('player_id').agg({
            'fantasy_points_ppr': ['mean', 'std', 'count'],
            'season': 'nunique'
        }).reset_index()
        
        # Flatten column names
        multi_year_stats.columns = ['player_id', 'avg_ppr_all_years', 'std_ppr_all_years', 
                                  'total_games', 'seasons_played']
        
        # Merge with current stats
        fantasy_stats = fantasy_stats.merge(multi_year_stats, on='player_id', how='left')
        
        # Calculate consistency score (lower std = more consistent)
        fantasy_stats['consistency_score'] = 1 / (1 + fantasy_stats['std_ppr_all_years'].fillna(0))
        
        # Add roster info
        roster_info = current_rosters[['player_id', 'team', 'age', 'years_exp', 'draft_number']].copy()
        fantasy_stats = fantasy_stats.merge(roster_info, on='player_id', how='left')
        
        # Use team from roster if available, otherwise recent_team
        fantasy_stats['team'] = fantasy_stats['team'].fillna(fantasy_stats['recent_team'])
        
        # Filter to fantasy-relevant positions
        fantasy_positions = ['QB', 'RB', 'WR', 'TE', 'K']
        fantasy_stats = fantasy_stats[fantasy_stats['position'].isin(fantasy_positions)].copy()
        
        # Filter to players with meaningful stats
        fantasy_stats = fantasy_stats[
            (fantasy_stats['games_played'] >= 1) & 
            (fantasy_stats['fantasy_points_ppr'] > 0)
        ].copy()
        
        # Rename columns to match our system
        column_mapping = {
            'player_name': 'name',
            'projected_fantasy_points_ppr': 'projected_points',
            'fantasy_points_ppr': 'fantasy_points_2024',
            'avg_fantasy_points_ppr': 'avg_points'
        }
        fantasy_stats.rename(columns=column_mapping, inplace=True)
        
        # Add required columns for our system
        fantasy_stats['injury_status'] = 'Healthy'
        fantasy_stats['adp'] = 999  # Default ADP
        
        # Sort by projected points
        fantasy_stats = fantasy_stats.sort_values('projected_points', ascending=False)
        
        print(f"   ✅ Processed {len(fantasy_stats)} fantasy-relevant players")
        print(f"   📊 Positions: {fantasy_stats['position'].value_counts().to_dict()}")
        
        return fantasy_stats
    
    def create_fantasy_dataset(self):
        """Create complete fantasy dataset and save it"""
        data, filename = self.collect_all_nfl_data()
        
        if data is not None:
            print(f"\n🎯 FANTASY DATASET SUMMARY:")
            print(f"   📊 Total Players: {len(data)}")
            print(f"   🏆 Top 5 PPR Projections:")
            top_5 = data[['name', 'position', 'team', 'projected_points']].head()
            print(top_5.to_string(index=False))
            
            # Show position breakdown
            print(f"\n   📋 Position Breakdown:")
            pos_breakdown = data['position'].value_counts()
            print(pos_breakdown.to_string())
            
            return data, filename
        else:
            print("❌ Failed to create fantasy dataset")
            return None, None

if __name__ == "__main__":
    collector = NFLverseCollector()
    data, filename = collector.create_fantasy_dataset()
    
    if data is not None:
        print(f"\n🎉 SUCCESS! Real NFL fantasy data collected!")
        print(f"📁 File: {filename}")
        print(f"🚀 Ready to replace synthetic data in fantasy system!")
    else:
        print("💥 Failed to collect nflverse data") 