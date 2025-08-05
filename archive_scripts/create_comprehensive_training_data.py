#!/usr/bin/env python3
"""
Comprehensive NFL Training Data Builder
Creates multi-year, feature-rich dataset for fantasy football AI model training
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('/home/nar/ff2025/venv/lib/python3.12/site-packages')
import nfl_data_py as nfl
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import re

def normalize_name(name):
    if pd.isna(name) or not name:
        return ""
    name = str(name).strip().upper()
    name = re.sub(r'\s+(JR\.?|SR\.?|III|IV|V)$', '', name)
    parts = name.split()
    known_expansions = {
        'D.HENRY': 'DERRICK HENRY',
        'R.WHITE': 'RACHAAD WHITE',
        'J.CHASE': 'JAMARR CHASE',
        'A.ST': 'AMONRA ST BROWN',
        'T.KELCE': 'TRAVIS KELCE',
        'C.MCCAFFREY': 'CHRISTIAN MCCAFFREY',
        'S.BARKLEY': 'SAQUON BARKLEY',
        'J.TAYLOR': 'JONATHAN TAYLOR',
        'D.MOORE': 'DJ MOORE',
        'M.EVANS': 'MIKE EVANS',
        'D.ADAMS': 'DAVANTE ADAMS',
        'T.HILL': 'TYREEK HILL',
        'J.JEFFERSON': 'JUSTIN JEFFERSON',
        'C.LAMB': 'CEEDEE LAMB',
        'A.BROWN': 'AJ BROWN',
        'G.KITTLE': 'GEORGE KITTLE',
        'S.DIGGS': 'STEFON DIGGS'
    }
    if len(parts) == 2 and len(parts[0]) <= 2:
        # Try with dot if not present
        short = parts[0] + '.' + parts[1]
        if short in known_expansions:
            return known_expansions[short]
        # Also try without dot
        short_no_dot = parts[0] + parts[1]
        if short_no_dot in known_expansions:
            return known_expansions[short_no_dot]
    name = re.sub(r"['\-\.]", "", name)
    return name

def determine_position(row):
    """Determine position based on stats"""
    initial = row.get('position', '').upper()
    
    # Preserve QB if originally QB or has significant passing stats
    if initial == 'QB' or row.get('passing_yards', 0) > 500 or row.get('passing_tds', 0) > 0:
        return 'QB'
    
    # For other positions, use existing logic with thresholds
    if row.get('carries', 0) > 50 or row.get('rushing_yards', 0) > 300:
        return 'RB'
    if row.get('targets', 0) > 50 and row.get('receptions', 0) > row.get('carries', 0):
        if row.get('receiving_yards', 0) / row.get('receptions', 1) < 8 and row.get('targets', 0) < 80:
            return 'TE'
        return 'WR'
    return initial or 'UNKNOWN'

def create_comprehensive_training_data():
    """Create comprehensive multi-year training dataset"""
    
    print("🏈 BUILDING COMPREHENSIVE NFL TRAINING DATASET")
    print("=" * 60)
    
    # 1. START WITH OUR BEST AVAILABLE DATA
    print("📊 Loading comprehensive local datasets...")
    
    # Fetch data for 2022-2024
    years = [2022, 2023, 2024]
    print('Fetching weekly data...')
    weekly_data = nfl.import_weekly_data(years)
    print('Fetching seasonal data...')
    seasonal_data = nfl.import_seasonal_data(years)
    print('Fetching rosters...')
    rosters = nfl.import_seasonal_rosters(years)
    print('Fetching schedules for bye weeks...')
    schedules = nfl.import_schedules(years)

    # Process bye weeks (example logic, adapt as needed)
    team_byes = {}
    for year in years:
        year_sched = schedules[schedules['season'] == year]
        for team in year_sched['away_team'].unique():
            bye_week = year_sched[(year_sched['away_team'] == team) | (year_sched['home_team'] == team)]['week'].max() + 1  # Simplified
            team_byes[(year, team)] = bye_week

    # Merge data (adapt your existing merge logic)
    merged_df = pd.merge(weekly_data, rosters, on=['player_id', 'season'], how='left')
    merged_df = pd.merge(merged_df, seasonal_data, on=['player_id', 'season'], how='left')

    # Add bye weeks
    merged_df['bye_week'] = merged_df.apply(lambda row: team_byes.get((row['season'], row['recent_team'])), axis=1)

    # Set player_name
    merged_df['player_name'] = merged_df['player_display_name']

    # Set position column
    if 'pos' in merged_df.columns:
        merged_df['position'] = merged_df['pos']
    elif 'position' not in merged_df.columns:
        print('Warning: Position column not found, adding default')
        merged_df['position'] = 'UNKNOWN'

    # Set position from rosters
    if 'position_y' in merged_df.columns:
        merged_df['position'] = merged_df['position_y']

    # Aggregation
    print('Merged DF columns:', list(merged_df.columns))
    seasonal_aggregated = merged_df.groupby(['player_id', 'season', 'player_name', 'position']).agg({
        'recent_team': 'last',
        'bye_week': 'first',
        'week_x': 'count',
        'fantasy_points_ppr_x': 'sum',
        'targets_x': 'sum',
        'carries_x': 'sum',
        'receiving_yards_x': 'sum',
        'rushing_yards_x': 'sum',
        'receptions_x': 'sum',
        'receiving_tds_x': 'sum',
        'rushing_tds_x': 'sum',
        'passing_yards_x': 'sum',
        'passing_tds_x': 'sum',
        'interceptions_x': 'sum'
    }).reset_index().rename(columns={'week_x': 'games'})

    # Then rename the summed columns back without _x for later use
    seasonal_aggregated = seasonal_aggregated.rename(columns={
        'fantasy_points_ppr_x': 'fantasy_points_ppr',
        'targets_x': 'targets',
        'carries_x': 'carries',
        'receiving_yards_x': 'receiving_yards',
        'rushing_yards_x': 'rushing_yards',
        'receptions_x': 'receptions',
        'receiving_tds_x': 'receiving_tds',
        'rushing_tds_x': 'rushing_tds',
        'passing_yards_x': 'passing_yards',
        'passing_tds_x': 'passing_tds',
        'interceptions_x': 'interceptions'
    })

    # Then continue with clean_name and position determination
    seasonal_aggregated['clean_name'] = seasonal_aggregated['player_name'].apply(normalize_name)
    seasonal_aggregated['position'] = seasonal_aggregated.apply(determine_position, axis=1)

    # Merge duplicates if any
    aggregated_df = seasonal_aggregated.groupby(['clean_name', 'season', 'position']).agg({
        'games': 'max',
        'fantasy_points_ppr': 'max',
        'targets': 'max',
        'carries': 'max',
        'receiving_yards': 'max',
        'rushing_yards': 'max',
        'receptions': 'max',
        'receiving_tds': 'max',
        'rushing_tds': 'max',
        'passing_yards': 'max',
        'passing_tds': 'max',
        'interceptions': 'max',
        'recent_team': 'first',
        'player_name': 'first'
    }).reset_index()

    # After aggregation, apply position determination
    aggregated_df['position'] = aggregated_df.apply(determine_position, axis=1)

    aggregated_df['player_key'] = aggregated_df['clean_name'] + '_' + aggregated_df['position'].astype(str)

    print(f"✅ Merged duplicates: Original {len(seasonal_aggregated)} -> Aggregated {len(aggregated_df)}")

    # 4. CALCULATE MULTI-YEAR FEATURES
    print("\n🔢 Calculating advanced multi-year features...")
    
    enhanced_players = []
    
    for player_key in aggregated_df['player_key'].unique():
        player_data = aggregated_df[aggregated_df['player_key'] == player_key].sort_values('season')
        
        if len(player_data) >= 1:
            # Use most recent season as base
            latest_season = player_data.iloc[-1].copy()
            
            # Multi-year calculations
            recent_3yr = player_data.tail(3)
            latest_season['avg_fantasy_points_3yr'] = recent_3yr['fantasy_points_ppr'].mean()
            latest_season['avg_targets_3yr'] = recent_3yr['targets'].mean()
            latest_season['avg_carries_3yr'] = recent_3yr['carries'].mean()
            
            # Performance trend (% change from previous year)
            if len(player_data) >= 2:
                prev_year = player_data.iloc[-2]['fantasy_points_ppr']
                curr_year = latest_season['fantasy_points_ppr']
                if prev_year > 0:
                    latest_season['performance_trend_pct'] = ((curr_year - prev_year) / prev_year) * 100
                else:
                    latest_season['performance_trend_pct'] = 0
            else:
                latest_season['performance_trend_pct'] = 0
            
            # Consistency score (coefficient of variation)
            if len(recent_3yr) > 1 and recent_3yr['fantasy_points_ppr'].std() > 0:
                latest_season['consistency_score'] = recent_3yr['fantasy_points_ppr'].mean() / recent_3yr['fantasy_points_ppr'].std()
            else:
                latest_season['consistency_score'] = 0
            
            # Experience and opportunity metrics
            latest_season['seasons_played'] = len(player_data)
            latest_season['total_games'] = player_data['games'].sum()
            latest_season['avg_opportunity'] = recent_3yr['targets'].mean() + recent_3yr['carries'].mean()
            
            # Position-specific features
            if latest_season['position'] in ['QB']:
                latest_season['passing_efficiency'] = latest_season.get('passing_tds', 0) / max(latest_season.get('passing_yards', 1), 1) * 1000
            elif latest_season['position'] in ['RB']:
                latest_season['total_touches'] = latest_season.get('carries', 0) + latest_season.get('targets', 0)
                latest_season['yards_per_touch'] = (latest_season.get('rushing_yards', 0) + latest_season.get('receiving_yards', 0)) / max(latest_season['total_touches'], 1)
            elif latest_season['position'] in ['WR', 'TE']:
                latest_season['catch_rate'] = latest_season.get('receptions', 0) / max(latest_season.get('targets', 1), 1)
                latest_season['yards_per_target'] = latest_season.get('receiving_yards', 0) / max(latest_season.get('targets', 1), 1)
            
            enhanced_players.append(latest_season)
    
    enhanced_dataset = pd.DataFrame(enhanced_players)
    print(f"✅ Enhanced dataset with multi-year features: {len(enhanced_dataset)} players")
    
    # 5. APPLY REASONABLE FILTERING
    print("\n🧹 Applying reasonable filtering...")
    
    # Filter for fantasy relevance (more lenient than before)
    enhanced_dataset = enhanced_dataset[enhanced_dataset['fantasy_points_ppr'] > 0]  # At least 0 fantasy points
    enhanced_dataset = enhanced_dataset[enhanced_dataset['games'] >= 1]  # At least 1 game
    
    # Focus on fantasy-relevant positions
    fantasy_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
    enhanced_dataset = enhanced_dataset[enhanced_dataset['position'].isin(fantasy_positions)]
    
    # Add derived features
    enhanced_dataset['points_per_game'] = enhanced_dataset['fantasy_points_ppr'] / enhanced_dataset['games']
    enhanced_dataset['opportunity_share'] = enhanced_dataset.get('targets', 0) + enhanced_dataset.get('carries', 0)
    
    # Fill missing values
    numeric_columns = enhanced_dataset.select_dtypes(include=[np.number]).columns
    enhanced_dataset[numeric_columns] = enhanced_dataset[numeric_columns].fillna(0)
    
    print(f"🎯 Final dataset: {len(enhanced_dataset)} players")
    print(f"   Positions: {dict(enhanced_dataset['position'].value_counts())}")
    print(f"   Features: {len(enhanced_dataset.columns)} columns")
    
    # 6. SAVE THE COMPREHENSIVE DATASET
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'comprehensive_training_data_{timestamp}.csv'
    
    enhanced_dataset.to_csv(output_file, index=False)
    print(f"\n💾 Saved comprehensive training data to: {output_file}")
    
    # 7. GENERATE DATASET SUMMARY
    print("\n📋 DATASET SUMMARY:")
    print("-" * 40)
    print(f"Total Players: {len(enhanced_dataset)}")
    print(f"Total Features: {len(enhanced_dataset.columns)} columns")
    print(f"Years Represented: {sorted(enhanced_dataset['season'].unique())}")
    print(f"Fantasy Points Range: {enhanced_dataset['fantasy_points_ppr'].min():.1f} - {enhanced_dataset['fantasy_points_ppr'].max():.1f}")
    
    print(f"\nPosition Breakdown:")
    for pos, count in enhanced_dataset['position'].value_counts().items():
        avg_points = enhanced_dataset[enhanced_dataset['position'] == pos]['fantasy_points_ppr'].mean()
        print(f"  {pos}: {count} players (avg: {avg_points:.1f} points)")
    
    # Show top players by position
    print(f"\n🏆 Top 3 players by position:")
    for pos in ['QB', 'RB', 'WR', 'TE']:
        if pos in enhanced_dataset['position'].values:
            top_pos = enhanced_dataset[enhanced_dataset['position'] == pos].nlargest(3, 'fantasy_points_ppr')
            print(f"\n{pos}:")
            for _, player in top_pos.iterrows():
                print(f"  {player['name']}: {player['fantasy_points_ppr']:.1f} points")
    
    return output_file, enhanced_dataset

if __name__ == "__main__":
    try:
        output_file, dataset = create_comprehensive_training_data()
        print(f"\n🎉 SUCCESS! Comprehensive training data created: {output_file}")
        print(f"Ready for model retraining with {len(dataset)} players and {len(dataset.columns)} features")
        print(f"\n💡 This dataset has:")
        print(f"   ✅ Multi-year performance trends")
        print(f"   ✅ Consistency scores") 
        print(f"   ✅ Position-specific metrics")
        print(f"   ✅ Opportunity and efficiency stats")
        print(f"   ✅ Much more comprehensive than the previous 542-line dataset!")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc() 