#!/usr/bin/env python3
"""
Complete nflverse Data Integration with Bye Weeks and Model Training
Fix all data format issues and create a comprehensive fantasy dataset

REAL NFL STATISTICS ONLY - NO SYNTHETIC DATA
"""

import nfl_data_py as nfl
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import shutil

class CompleteDataIntegrator:
    """Complete integration of nflverse data with all required fields"""
    
    def __init__(self):
        self.current_year = 2024
        self.backup_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def backup_existing_files(self):
        """Backup all existing files"""
        print("💾 BACKING UP EXISTING FILES...")
        
        files_to_backup = [
            'data/players.json',
            'data/fantasy_metrics_2024.csv',
            'models/draft_model.pkl'
        ]
        
        for file_path in files_to_backup:
            if os.path.exists(file_path):
                backup_path = f"{file_path}.backup_{self.backup_time}"
                shutil.copy2(file_path, backup_path)
                print(f"   ✅ Backed up {file_path}")
    
    def collect_comprehensive_nfl_data(self):
        """Collect comprehensive NFL data with bye weeks"""
        print("🏈 COLLECTING COMPREHENSIVE NFL DATA WITH BYE WEEKS...")
        
        try:
            # Get rosters (includes current teams)
            print("   📋 Getting 2024 rosters...")
            rosters = nfl.import_seasonal_rosters([self.current_year])
            
            # Get weekly data (for fantasy statistics)
            print("   🏆 Getting weekly fantasy data (2022-2024)...")
            weekly_data = nfl.import_weekly_data([2022, 2023, 2024])
            
            # Get seasonal data (for market shares)
            print("   📈 Getting seasonal data...")
            seasonal_data = nfl.import_seasonal_data([2022, 2023, 2024])
            
            # Get schedules for bye weeks
            print("   📅 Getting 2024 schedules for bye weeks...")
            schedules = nfl.import_schedules([self.current_year])
            
            return rosters, weekly_data, seasonal_data, schedules
            
        except Exception as e:
            print(f"❌ Error collecting data: {e}")
            return None, None, None, None
    
    def extract_bye_weeks(self, schedules):
        """Extract bye weeks for each team"""
        print("   🔄 Extracting bye weeks from schedules...")
        
        # Find weeks where teams don't play
        all_weeks = set(range(1, 19))  # NFL weeks 1-18
        team_bye_weeks = {}
        
        # Get all teams
        teams = set(schedules['home_team'].unique()) | set(schedules['away_team'].unique())
        
        for team in teams:
            # Find weeks where team plays
            team_games = schedules[
                (schedules['home_team'] == team) | 
                (schedules['away_team'] == team)
            ]
            playing_weeks = set(team_games['week'].unique())
            
            # Bye weeks are weeks they don't play (excluding playoffs)
            bye_weeks = all_weeks - playing_weeks
            # Filter to regular season only (weeks 1-17)
            bye_weeks = [week for week in bye_weeks if 1 <= week <= 17]
            
            if bye_weeks:
                team_bye_weeks[team] = bye_weeks[0]  # Usually just one bye week
            else:
                team_bye_weeks[team] = 'Unknown'
        
        print(f"   ✅ Found bye weeks for {len(team_bye_weeks)} teams")
        return team_bye_weeks
    
    def process_comprehensive_data(self, rosters, weekly_data, seasonal_data, team_bye_weeks):
        """Process all data into comprehensive fantasy format"""
        print("   🔧 Processing comprehensive fantasy data...")
        
        # Filter to current year rosters
        current_rosters = rosters[rosters['season'] == self.current_year].copy()
        
        # Calculate 2024 fantasy stats
        weekly_2024 = weekly_data[weekly_data['season'] == self.current_year].copy()
        
        # Aggregate weekly stats to season totals
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
        fantasy_stats['avg_fantasy_points_ppr'] = fantasy_stats['fantasy_points_ppr'] / fantasy_stats['games_played']
        fantasy_stats['projected_points'] = fantasy_stats['avg_fantasy_points_ppr'] * 17  # Full season projection
        
        # Multi-year consistency analysis
        multi_year_stats = weekly_data.groupby('player_id').agg({
            'fantasy_points_ppr': ['mean', 'std', 'count'],
            'season': 'nunique'
        }).reset_index()
        
        # Flatten multi-index columns
        multi_year_stats.columns = ['player_id', 'avg_ppr_all_years', 'std_ppr_all_years', 
                                   'total_games', 'seasons_played']
        
        # Merge with fantasy stats
        fantasy_stats = fantasy_stats.merge(multi_year_stats, on='player_id', how='left')
        
        # Calculate consistency score
        fantasy_stats['consistency_score'] = 1 / (1 + fantasy_stats['std_ppr_all_years'].fillna(0))
        
        # Add roster information
        roster_info = current_rosters[['player_id', 'team', 'age', 'years_exp', 'draft_number']].copy()
        fantasy_stats = fantasy_stats.merge(roster_info, on='player_id', how='left')
        
        # Use roster team if available, otherwise recent_team
        fantasy_stats['team'] = fantasy_stats['team'].fillna(fantasy_stats['recent_team'])
        
        # Add bye weeks
        fantasy_stats['bye_week'] = fantasy_stats['team'].map(team_bye_weeks).fillna('Unknown')
        
        # Filter to fantasy-relevant positions and meaningful stats
        fantasy_positions = ['QB', 'RB', 'WR', 'TE', 'K']
        fantasy_stats = fantasy_stats[
            (fantasy_stats['position'].isin(fantasy_positions)) &
            (fantasy_stats['games_played'] >= 1) &
            (fantasy_stats['fantasy_points_ppr'] > 0)
        ].copy()
        
        # Add required columns
        fantasy_stats['injury_status'] = 'Healthy'  # Default, can be updated from injury API
        fantasy_stats['adp'] = 999  # Default ADP
        
        # Rename columns to match system expectations
        fantasy_stats.rename(columns={'player_name': 'name'}, inplace=True)
        
        # Sort by projected points
        fantasy_stats = fantasy_stats.sort_values('projected_points', ascending=False)
        
        print(f"   ✅ Processed {len(fantasy_stats)} fantasy-relevant players")
        return fantasy_stats
    
    def create_players_json_dict(self, fantasy_stats):
        """Create players.json in dictionary format expected by UI"""
        print("   📝 Creating players.json in dictionary format...")
        
        players_dict = {}
        for _, player in fantasy_stats.iterrows():
            player_id = player['player_id']
            players_dict[player_id] = {
                'name': player['name'],
                'full_name': player['name'],  # For compatibility
                'position': player['position'],
                'team': player['team'],
                'injury_status': player['injury_status'],
                'bye_week': player['bye_week'],
                'age': player.get('age', 0),
                'years_exp': player.get('years_exp', 0)
            }
        
        return players_dict
    
    def create_model_training_data(self, weekly_data):
        """Create comprehensive dataset for catboost model training"""
        print("🤖 CREATING MODEL TRAINING DATA...")
        
        # Use last 2 years for training (2022-2023)
        training_data = weekly_data[weekly_data['season'].isin([2022, 2023])].copy()
        
        # Filter to fantasy-relevant positions
        fantasy_positions = ['QB', 'RB', 'WR', 'TE', 'K']
        training_data = training_data[training_data['position'].isin(fantasy_positions)].copy()
        
        # Remove players with 0 fantasy points (didn't play meaningfully)
        training_data = training_data[training_data['fantasy_points_ppr'] > 0].copy()
        
        print(f"   ✅ Created training dataset with {len(training_data)} records")
        print(f"   📊 Training data breakdown:")
        print(f"      - Years: 2022-2023")
        print(f"      - Positions: {training_data['position'].value_counts().to_dict()}")
        
        return training_data
    
    def save_all_data(self, fantasy_stats, players_dict, training_data):
        """Save all data files"""
        print("💾 SAVING ALL DATA FILES...")
        
        # Save players.json (dictionary format)
        with open('data/players.json', 'w') as f:
            json.dump(players_dict, f, indent=2)
        print(f"   ✅ Saved data/players.json with {len(players_dict)} players (dict format)")
        
        # Save fantasy_metrics_2024.csv
        fantasy_stats.to_csv('data/fantasy_metrics_2024.csv', index=False)
        print(f"   ✅ Saved data/fantasy_metrics_2024.csv with {len(fantasy_stats)} players")
        
        # Save training data for model
        training_data.to_csv('data/model_training_data.csv', index=False)
        print(f"   ✅ Saved data/model_training_data.csv with {len(training_data)} records")
        
        # Save summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary = {
            'timestamp': timestamp,
            'total_players': len(fantasy_stats),
            'positions': fantasy_stats['position'].value_counts().to_dict(),
            'top_players': fantasy_stats[['name', 'position', 'team', 'projected_points', 'bye_week']].head(10).to_dict('records'),
            'training_records': len(training_data)
        }
        
        with open(f'data/integration_summary_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def integrate_complete_data(self):
        """Complete data integration process"""
        print("🚀 COMPLETE NFLVERSE DATA INTEGRATION")
        print("=" * 60)
        
        # Step 1: Backup existing files
        self.backup_existing_files()
        
        # Step 2: Collect comprehensive data
        rosters, weekly_data, seasonal_data, schedules = self.collect_comprehensive_nfl_data()
        
        if any(data is None for data in [rosters, weekly_data, seasonal_data, schedules]):
            print("❌ Failed to collect required data")
            return False
        
        # Step 3: Extract bye weeks
        team_bye_weeks = self.extract_bye_weeks(schedules)
        
        # Step 4: Process comprehensive data
        fantasy_stats = self.process_comprehensive_data(rosters, weekly_data, seasonal_data, team_bye_weeks)
        
        # Step 5: Create players.json dictionary
        players_dict = self.create_players_json_dict(fantasy_stats)
        
        # Step 6: Create model training data
        training_data = self.create_model_training_data(weekly_data)
        
        # Step 7: Save all data
        summary = self.save_all_data(fantasy_stats, players_dict, training_data)
        
        # Step 8: Display results
        print(f"\n🎉 INTEGRATION COMPLETE!")
        print(f"✅ {summary['total_players']} real NFL players with full data")
        print(f"✅ Bye weeks for all teams")
        print(f"✅ {summary['training_records']} training records for catboost model")
        print(f"\n🏆 Top 5 Players:")
        for i, player in enumerate(summary['top_players'][:5], 1):
            print(f"   {i}. {player['name']} ({player['position']}, {player['team']}) - {player['projected_points']:.1f} pts, Bye: {player['bye_week']}")
        
        print(f"\n📊 Position Breakdown:")
        for pos, count in summary['positions'].items():
            print(f"   {pos}: {count}")
        
        return True

if __name__ == "__main__":
    integrator = CompleteDataIntegrator()
    success = integrator.integrate_complete_data()
    
    if success:
        print(f"\n🚀 Ready to restart UI with complete real NFL data!")
        print(f"🤖 Ready to train catboost model on 2 years of real data!")
    else:
        print(f"\n💥 Integration failed, check errors above") 