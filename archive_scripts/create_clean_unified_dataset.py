#!/usr/bin/env python3
"""
Clean Unified Dataset Builder
Starts with current player names and builds proper historical data
"""

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import warnings
import re
warnings.filterwarnings('ignore')

def normalize_name_for_matching(name):
    """Normalize names for better historical matching"""
    if pd.isna(name) or not name:
        return ""
    
    name = str(name).strip()
    
    # Remove common suffixes
    name = re.sub(r'\s+(Jr\.?|Sr\.?|III|IV|V)$', '', name, flags=re.IGNORECASE)
    
    # Handle apostrophes and special characters
    name = re.sub(r"['\-\.]", "", name)
    
    # Convert to uppercase for consistent matching
    name = name.upper()
    
    return name

def get_current_player_data():
    """Get current player data from the live system"""
    print("📊 Fetching current player data from live system...")
    
    try:
        response = requests.get("http://localhost:5000/suggest?all_available=true")
        if response.status_code == 200:
            players = response.json()
            df = pd.DataFrame(players)
            print(f"✅ Loaded {len(df)} current players from live system")
            return df
        else:
            print(f"❌ Failed to get current players: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error fetching current players: {e}")
        return None

def load_historical_data():
    """Load all available historical datasets"""
    print("📊 Loading historical datasets...")
    
    historical_data = []
    
    # Load 2022-2023 historical data
    try:
        hist_df = pd.read_csv('data/historical/real_nfl_historical_2022_2023.csv')
        print(f"✅ Loaded 2022-2023 data: {len(hist_df)} records")
        historical_data.append(hist_df)
    except Exception as e:
        print(f"⚠️ Could not load 2022-2023 data: {e}")
    
    # Load 2024 comprehensive data  
    try:
        comp_df = pd.read_csv('data/comprehensive_real_nfl_2024.csv')
        print(f"✅ Loaded 2024 data: {len(comp_df)} records") 
        historical_data.append(comp_df)
    except Exception as e:
        print(f"⚠️ Could not load 2024 data: {e}")
    
    # Load fantasy metrics
    try:
        fantasy_df = pd.read_csv('data/fantasy_metrics_2024.csv')
        print(f"✅ Loaded fantasy metrics: {len(fantasy_df)} records")
        historical_data.append(fantasy_df)
    except Exception as e:
        print(f"⚠️ Could not load fantasy metrics: {e}")
    
    return historical_data

def create_name_mapping(current_players, historical_datasets):
    """Create mapping between current player names and historical names"""
    print("🔗 Creating name mapping between current and historical data...")
    
    current_names = current_players['name'].unique()
    name_mapping = {}
    
    # Collect all historical names
    all_historical_names = set()
    for dataset in historical_datasets:
        # Try different possible name columns
        name_cols = ['name', 'player_name', 'player', 'Player']
        for col in name_cols:
            if col in dataset.columns:
                all_historical_names.update(dataset[col].dropna().unique())
                break
    
    print(f"📋 Found {len(current_names)} current players, {len(all_historical_names)} historical names")
    
    matched = 0
    for current_name in current_names:
        normalized_current = normalize_name_for_matching(current_name)
        
        # Try exact match first
        for hist_name in all_historical_names:
            normalized_hist = normalize_name_for_matching(hist_name)
            
            if normalized_current == normalized_hist:
                name_mapping[current_name] = hist_name
                matched += 1
                break
        
        # Try partial matching for common cases
        if current_name not in name_mapping:
            current_parts = normalized_current.split()
            if len(current_parts) >= 2:
                first_name = current_parts[0]
                last_name = current_parts[-1]
                
                for hist_name in all_historical_names:
                    normalized_hist = normalize_name_for_matching(hist_name)
                    
                    # Check for abbreviated first name format (J.SMITH)
                    if f"{first_name[0]}{last_name}" == normalized_hist or f"{first_name[0]}.{last_name}" == normalized_hist:
                        name_mapping[current_name] = hist_name
                        matched += 1
                        break
                    
                    # Check for last name + first initial
                    if normalized_hist.startswith(last_name) and first_name[0] in normalized_hist:
                        name_mapping[current_name] = hist_name
                        matched += 1
                        break
    
    print(f"✅ Successfully mapped {matched} players ({matched/len(current_names)*100:.1f}%)")
    return name_mapping

def build_unified_dataset():
    """Build a clean, unified dataset"""
    
    print("🏈 BUILDING CLEAN UNIFIED DATASET")
    print("=" * 50)
    
    # 1. Get current player data
    current_players = get_current_player_data()
    if current_players is None:
        print("❌ Cannot proceed without current player data")
        return None
    
    # 2. Load historical datasets
    historical_datasets = load_historical_data()
    if not historical_datasets:
        print("❌ No historical data available")
        return None
    
    # 3. Create name mapping
    name_mapping = create_name_mapping(current_players, historical_datasets)
    
    # 4. Start with current players as base
    unified_df = current_players.copy()
    
    # 5. Add historical data for matched players
    print("📊 Adding historical data for matched players...")
    
    historical_features = {}
    
    for dataset in historical_datasets:
        name_col = None
        for col in ['name', 'player_name', 'player', 'Player']:
            if col in dataset.columns:
                name_col = col
                break
        
        if name_col is None:
            continue
            
        for current_name, hist_name in name_mapping.items():
            player_data = dataset[dataset[name_col] == hist_name]
            
            if not player_data.empty:
                # Extract key historical features
                for _, row in player_data.iterrows():
                    if current_name not in historical_features:
                        historical_features[current_name] = []
                    
                    # Build historical record
                    hist_record = {
                        'fantasy_points_ppr': row.get('fantasy_points_ppr', 0),
                        'games': row.get('games', 0),
                        'season': row.get('season', 2024),
                        'targets': row.get('targets', 0),
                        'carries': row.get('carries', 0),
                        'receiving_yards': row.get('receiving_yards', 0),
                        'rushing_yards': row.get('rushing_yards', 0)
                    }
                    historical_features[current_name].append(hist_record)
    
    # 6. Calculate multi-year features
    print("📈 Calculating multi-year performance metrics...")
    
    multi_year_features = []
    
    for _, player in unified_df.iterrows():
        player_name = player['name']
        
        # Get historical data for this player
        hist_data = historical_features.get(player_name, [])
        
        if len(hist_data) >= 2:  # At least 2 seasons of data
            # Calculate averages
            fantasy_points = [d['fantasy_points_ppr'] for d in hist_data if d['fantasy_points_ppr'] > 0]
            games_played = [d['games'] for d in hist_data if d['games'] > 0]
            
            avg_fantasy_points_3yr = np.mean(fantasy_points) if fantasy_points else 0
            avg_games_3yr = np.mean(games_played) if games_played else 0
            
            # Calculate performance trend (most recent vs oldest)
            if len(fantasy_points) >= 2:
                sorted_by_season = sorted(hist_data, key=lambda x: x['season'])
                oldest_season = sorted_by_season[0]['fantasy_points_ppr']
                newest_season = sorted_by_season[-1]['fantasy_points_ppr']
                
                if oldest_season > 0:
                    performance_trend_pct = ((newest_season - oldest_season) / oldest_season) * 100
                else:
                    performance_trend_pct = 0
            else:
                performance_trend_pct = 0
            
            # Calculate consistency score (inverse of coefficient of variation)
            consistency_score = 1 / (np.std(fantasy_points) / np.mean(fantasy_points) + 1) if len(fantasy_points) > 1 and np.mean(fantasy_points) > 0 else 0
            
        else:
            # No historical data - use current projections
            avg_fantasy_points_3yr = player.get('projected_points', 0)
            avg_games_3yr = 16  # Assume full season
            performance_trend_pct = 0
            consistency_score = 0.1  # Neutral score for new players
        
        multi_year_record = {
            'name': player_name,
            'position': player['position'],
            'avg_fantasy_points_3yr': avg_fantasy_points_3yr,
            'avg_games_3yr': avg_games_3yr,
            'performance_trend_pct': performance_trend_pct,
            'consistency_score': consistency_score,
            'historical_seasons': len(hist_data),
            # Current season projections
            'projected_points': player.get('projected_points', 0),
            'fantasy_points_ppr': player.get('fantasy_points_ppr', 0),
            'adp_rank': player.get('adp_rank', 999)
        }
        
        multi_year_features.append(multi_year_record)
    
    # 7. Create final clean dataset
    clean_df = pd.DataFrame(multi_year_features)
    
    # 8. Save the clean dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"clean_unified_dataset_{timestamp}.csv"
    clean_df.to_csv(filename, index=False)
    
    print(f"✅ Created clean unified dataset: {filename}")
    print(f"📊 Total players: {len(clean_df)}")
    print(f"📊 Players with historical data: {len(clean_df[clean_df['historical_seasons'] > 0])}")
    print(f"📊 Average fantasy points range: {clean_df['avg_fantasy_points_3yr'].min():.1f} - {clean_df['avg_fantasy_points_3yr'].max():.1f}")
    
    # Show sample of players with historical data
    sample_with_history = clean_df[clean_df['historical_seasons'] > 0].head(5)
    if not sample_with_history.empty:
        print("\n📋 Sample players with historical data:")
        for _, player in sample_with_history.iterrows():
            print(f"  {player['name']}: {player['avg_fantasy_points_3yr']:.1f} avg points, {player['performance_trend_pct']:.1f}% trend")
    
    return clean_df

if __name__ == "__main__":
    build_unified_dataset() 