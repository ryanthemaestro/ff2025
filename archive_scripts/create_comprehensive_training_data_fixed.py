#!/usr/bin/env python3
"""
Fixed Comprehensive NFL Training Data Builder
Handles name matching across different formats to maximize 2024 data usage
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('/home/nar/ff2025/venv/lib/python3.12/site-packages')
from datetime import datetime
import warnings
import re
warnings.filterwarnings('ignore')

def normalize_name(name):
    """Normalize player names for better matching"""
    if pd.isna(name) or not name:
        return ""
    
    name = str(name).upper().strip()
    
    # Remove common suffixes
    name = re.sub(r'\s+(JR\.?|SR\.?|III|IV|V)$', '', name)
    
    # Handle abbreviated first names (L.JACKSON -> L JACKSON)
    name = re.sub(r'\.', ' ', name)
    
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name)
    
    return name.strip()

def create_name_variations(name):
    """Create possible name variations for matching"""
    variations = set()
    normalized = normalize_name(name)
    variations.add(normalized)
    
    if ' ' in normalized:
        parts = normalized.split()
        if len(parts) >= 2:
            first, last = parts[0], ' '.join(parts[1:])
            
            # Add abbreviated first name version
            if len(first) > 1:
                abbreviated = f"{first[0]} {last}"
                variations.add(abbreviated)
            
            # Add full name version (already normalized)
            variations.add(f"{first} {last}")
            
            # Add last name only for unique matching
            variations.add(last)
    
    return variations

def find_best_name_match(target_name, candidate_names, candidate_data):
    """Find the best name match using various strategies"""
    target_variations = create_name_variations(target_name)
    
    # Try exact matches first
    for variation in target_variations:
        for idx, candidate in enumerate(candidate_names):
            candidate_variations = create_name_variations(candidate)
            if variation in candidate_variations:
                return idx
    
    # Try partial matches (last name + position)
    target_parts = normalize_name(target_name).split()
    if len(target_parts) >= 2:
        target_last = target_parts[-1]
        
        for idx, candidate in enumerate(candidate_names):
            candidate_parts = normalize_name(candidate).split()
            if len(candidate_parts) >= 2:
                candidate_last = candidate_parts[-1]
                if target_last == candidate_last:
                    return idx
    
    return None

def create_comprehensive_training_data_fixed():
    """Create comprehensive multi-year training dataset with improved name matching"""
    
    print("🏈 BUILDING FIXED COMPREHENSIVE NFL TRAINING DATASET")
    print("=" * 60)
    
    # 1. LOAD ALL DATA SOURCES
    print("📊 Loading all data sources...")
    
    # Load historical data (2022-2023)
    try:
        hist_data = pd.read_csv('data/historical/real_nfl_historical_2022_2023.csv')
        print(f"✅ Historical data: {len(hist_data)} records from 2022-2023")
    except Exception as e:
        print(f"❌ Error loading historical data: {e}")
        return
    
    # Load 2024 comprehensive data
    try:
        comp_2024 = pd.read_csv('data/comprehensive_real_nfl_2024.csv')
        print(f"✅ 2024 data: {len(comp_2024)} records")
    except Exception as e:
        print(f"❌ Error loading 2024 data: {e}")
        return
    
    # 2. NORMALIZE ALL DATA TO COMMON STRUCTURE
    print("\n🔧 Normalizing data structures...")
    
    all_player_seasons = []
    
    # Add historical data (2022-2023) with normalized structure
    for _, player in hist_data.iterrows():
        player_record = {
            'name': player.get('name', player.get('full_name', '')),
            'position': player.get('position', ''),
            'team': player.get('team', ''),
            'season': player.get('season', 2023),
            'games': player.get('games', 0),
            'fantasy_points_ppr': player.get('fantasy_points_ppr', player.get('fantasy_points', 0)),
            'targets': player.get('targets', 0),
            'carries': player.get('carries', 0),
            'receiving_yards': player.get('receiving_yards', 0),
            'rushing_yards': player.get('rushing_yards', 0),
            'receptions': player.get('receptions', 0),
            'receiving_tds': player.get('receiving_tds', 0),
            'rushing_tds': player.get('rushing_tds', 0),
            'passing_yards': player.get('passing_yards', 0),
            'passing_tds': player.get('passing_tds', 0),
            'interceptions': player.get('interceptions', 0)
        }
        all_player_seasons.append(player_record)
    
    # Add 2024 data with normalized structure  
    for _, player in comp_2024.iterrows():
        player_record = {
            'name': player.get('player_name', player.get('name', '')),
            'position': player.get('position', ''),
            'team': player.get('team', ''),
            'season': 2024,
            'games': player.get('games', 0),
            'fantasy_points_ppr': player.get('fantasy_points_ppr', 0),
            'targets': player.get('targets', 0),
            'carries': player.get('carries', 0),
            'receiving_yards': player.get('receiving_yards', 0),
            'rushing_yards': player.get('rushing_yards', 0),
            'receptions': player.get('receptions', 0),
            'receiving_tds': player.get('receiving_tds', 0),
            'rushing_tds': player.get('rushing_tds', 0),
            'passing_yards': player.get('passing_yards', 0),
            'passing_tds': player.get('passing_tds', 0),
            'interceptions': player.get('interceptions', 0)
        }
        all_player_seasons.append(player_record)
    
    # Convert to DataFrame
    all_seasons_df = pd.DataFrame(all_player_seasons)
    
    print(f"✅ Combined player-season dataset: {len(all_seasons_df)} records")
    print(f"   Seasons: {sorted(all_seasons_df['season'].unique())}")
    
    # 3. IMPROVED NAME MATCHING
    print("\n🔗 Performing smart name matching across years...")
    
    # Group by position first to improve matching accuracy
    enhanced_players = []
    
    for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
        position_data = all_seasons_df[all_seasons_df['position'] == position].copy()
        if len(position_data) == 0:
            continue
            
        print(f"🎯 Processing {position}: {len(position_data)} records")
        
        # Create name mapping for this position
        unique_names = position_data['name'].unique()
        name_groups = {}
        
        for name in unique_names:
            if pd.isna(name) or not name:
                continue
                
            # Find if this name matches any existing group
            matched_group = None
            for group_key, group_names in name_groups.items():
                if any(normalize_name(name) in create_name_variations(existing_name) or 
                       normalize_name(existing_name) in create_name_variations(name)
                       for existing_name in group_names):
                    matched_group = group_key
                    break
            
            if matched_group:
                name_groups[matched_group].append(name)
            else:
                # Create new group
                name_groups[normalize_name(name)] = [name]
        
        print(f"   📊 Found {len(name_groups)} unique players for {position}")
        
        # Process each name group (representing one player across seasons)
        for group_key, name_variants in name_groups.items():
            # Get all data for this player (all name variants)
            player_data = position_data[position_data['name'].isin(name_variants)].copy()
            player_data = player_data.sort_values('season')
            
            if len(player_data) >= 1:
                # Use most recent season as base
                latest_season = player_data.iloc[-1].copy()
                
                # Set canonical name (prefer full names over abbreviated)
                canonical_name = max(name_variants, key=len)  # Longest name
                latest_season['name'] = canonical_name
                latest_season['canonical_name'] = canonical_name
                
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
    
    # 4. APPLY REASONABLE FILTERING
    print("\n🧹 Applying reasonable filtering...")
    
    # Filter for fantasy relevance (more lenient than before)
    enhanced_dataset = enhanced_dataset[enhanced_dataset['fantasy_points_ppr'] > 5]  # Lowered threshold
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
    print(f"   Seasons represented: {dict(enhanced_dataset['season'].value_counts())}")
    print(f"   Features: {len(enhanced_dataset.columns)} columns")
    
    # 5. SAVE THE ENHANCED DATASET
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'comprehensive_training_data_FIXED_{timestamp}.csv'
    
    # Create clean name and player key for consistency
    enhanced_dataset['clean_name'] = enhanced_dataset['canonical_name'].str.upper().str.strip()
    enhanced_dataset['player_key'] = enhanced_dataset['clean_name'] + '_' + enhanced_dataset['position'].astype(str)
    
    enhanced_dataset.to_csv(filename, index=False)
    print(f"💾 Saved enhanced dataset: {filename}")
    
    # 6. SHOW SAMPLE OF IMPROVED MATCHES
    print(f"\n🎯 SAMPLE OF IMPROVED MULTI-YEAR MATCHES:")
    print("=" * 50)
    
    # Show players with multiple seasons
    multi_season_players = enhanced_dataset[enhanced_dataset['seasons_played'] > 1].head(10)
    for _, player in multi_season_players.iterrows():
        print(f"✅ {player['canonical_name']} ({player['position']}): {player['seasons_played']} seasons")
        print(f"   Trend: {player['performance_trend_pct']:.1f}% | Avg 3yr: {player['avg_fantasy_points_3yr']:.1f}")
    
    return filename

if __name__ == "__main__":
    create_comprehensive_training_data_fixed() 