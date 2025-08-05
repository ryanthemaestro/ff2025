#!/usr/bin/env python3
"""
Recency-Weighted Dataset Builder
Emphasizes recent performance with exponential weighting
2024 data weighted 4x, 2023 data 2x, 2022 data 1x
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
    """Load all available historical datasets with metadata"""
    print("📊 Loading historical datasets with season info...")
    
    historical_data = []
    
    # Load 2022-2023 historical data
    try:
        hist_df = pd.read_csv('data/historical/real_nfl_historical_2022_2023.csv')
        # Add year info if not present
        if 'season' not in hist_df.columns:
            # Try to infer season from data or default to mixed
            hist_df['season'] = 2023  # Most recent in this dataset
        print(f"✅ Loaded 2022-2023 data: {len(hist_df)} records")
        historical_data.append(hist_df)
    except Exception as e:
        print(f"⚠️ Could not load 2022-2023 data: {e}")
    
    # Load 2024 comprehensive data  
    try:
        comp_df = pd.read_csv('data/comprehensive_real_nfl_2024.csv')
        # Add season if not present
        if 'season' not in comp_df.columns:
            comp_df['season'] = 2024
        print(f"✅ Loaded 2024 data: {len(comp_df)} records") 
        historical_data.append(comp_df)
    except Exception as e:
        print(f"⚠️ Could not load 2024 data: {e}")
    
    # Load fantasy metrics (assume 2024)
    try:
        fantasy_df = pd.read_csv('data/fantasy_metrics_2024.csv')
        if 'season' not in fantasy_df.columns:
            fantasy_df['season'] = 2024
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

def calculate_recency_weights():
    """Calculate exponential recency weights favoring recent seasons"""
    
    # Exponential weighting favoring recent data
    weights = {
        2024: 4.0,  # Most recent season gets 4x weight
        2023: 2.0,  # Previous season gets 2x weight  
        2022: 1.0,  # Oldest season gets base weight
        2021: 0.5,  # Even older gets reduced weight
        2020: 0.25  # Very old gets minimal weight
    }
    
    print("📈 Using exponential recency weights:")
    for year, weight in weights.items():
        print(f"   {year}: {weight}x weight")
    
    return weights

def build_recency_weighted_dataset():
    """Build a dataset that heavily emphasizes recent performance"""
    
    print("🏈 BUILDING RECENCY-WEIGHTED DATASET")
    print("=" * 55)
    
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
    
    # 3. Get recency weights
    recency_weights = calculate_recency_weights()
    
    # 4. Create name mapping
    name_mapping = create_name_mapping(current_players, historical_datasets)
    
    # 5. Start with current players as base
    unified_df = current_players.copy()
    
    # 6. Add historical data with recency weighting
    print("📊 Adding recency-weighted historical data...")
    
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
                # Extract key historical features with season weighting
                for _, row in player_data.iterrows():
                    if current_name not in historical_features:
                        historical_features[current_name] = []
                    
                    season = row.get('season', 2024)
                    weight = recency_weights.get(season, 0.1)  # Very low weight for unknown seasons
                    
                    # Build weighted historical record
                    hist_record = {
                        'fantasy_points_ppr': row.get('fantasy_points_ppr', 0) * weight,
                        'games': row.get('games', 0),
                        'season': season,
                        'weight': weight,
                        'targets': row.get('targets', 0) * weight,
                        'carries': row.get('carries', 0) * weight,
                        'receiving_yards': row.get('receiving_yards', 0) * weight,
                        'rushing_yards': row.get('rushing_yards', 0) * weight,
                        'raw_fantasy_points': row.get('fantasy_points_ppr', 0)  # Unweighted for trend calc
                    }
                    historical_features[current_name].append(hist_record)
    
    # 7. Calculate recency-weighted multi-year features
    print("📈 Calculating recency-weighted performance metrics...")
    
    recency_features = []
    
    for _, player in unified_df.iterrows():
        player_name = player['name']
        
        # Get historical data for this player
        hist_data = historical_features.get(player_name, [])
        
        if len(hist_data) >= 1:  # At least 1 season of data
            # Calculate weighted averages (heavily favoring recent seasons)
            total_weight = sum(d['weight'] for d in hist_data)
            
            if total_weight > 0:
                weighted_fantasy_points = sum(d['fantasy_points_ppr'] for d in hist_data) / total_weight
                weighted_targets = sum(d['targets'] for d in hist_data) / total_weight
                weighted_carries = sum(d['carries'] for d in hist_data) / total_weight
                weighted_rec_yards = sum(d['receiving_yards'] for d in hist_data) / total_weight
                weighted_rush_yards = sum(d['rushing_yards'] for d in hist_data) / total_weight
            else:
                weighted_fantasy_points = 0
                weighted_targets = 0
                weighted_carries = 0
                weighted_rec_yards = 0
                weighted_rush_yards = 0
            
            # Calculate recent vs historical trend (2024 vs earlier years)
            recent_data = [d for d in hist_data if d['season'] >= 2024]
            older_data = [d for d in hist_data if d['season'] < 2024]
            
            if recent_data and older_data:
                recent_avg = np.mean([d['raw_fantasy_points'] for d in recent_data])
                older_avg = np.mean([d['raw_fantasy_points'] for d in older_data])
                
                if older_avg > 0:
                    recent_trend_pct = ((recent_avg - older_avg) / older_avg) * 100
                else:
                    recent_trend_pct = 0
            else:
                recent_trend_pct = 0
            
            # Calculate momentum (2024 vs 2023 performance)
            data_2024 = [d for d in hist_data if d['season'] == 2024]
            data_2023 = [d for d in hist_data if d['season'] == 2023]
            
            momentum_score = 0
            if data_2024 and data_2023:
                pts_2024 = np.mean([d['raw_fantasy_points'] for d in data_2024])
                pts_2023 = np.mean([d['raw_fantasy_points'] for d in data_2023])
                if pts_2023 > 0:
                    momentum_score = ((pts_2024 - pts_2023) / pts_2023) * 100
            
            # Calculate consistency (lower variance = more consistent)
            all_raw_points = [d['raw_fantasy_points'] for d in hist_data if d['raw_fantasy_points'] > 0]
            if len(all_raw_points) > 1:
                consistency_score = 1 / (np.std(all_raw_points) / np.mean(all_raw_points) + 1)
            else:
                consistency_score = 0.1
                
        else:
            # No historical data - use current projections with neutral metrics
            weighted_fantasy_points = player.get('projected_points', 0)
            weighted_targets = 0
            weighted_carries = 0
            weighted_rec_yards = 0
            weighted_rush_yards = 0
            recent_trend_pct = 0
            momentum_score = 0
            consistency_score = 0.1  # Neutral score for new players
        
        # Create feature record with heavy recent emphasis
        feature_record = {
            'name': player_name,
            'position': player['position'],
            'recency_weighted_fantasy_points': weighted_fantasy_points,
            'recency_weighted_targets': weighted_targets,
            'recency_weighted_carries': weighted_carries,
            'recency_weighted_rec_yards': weighted_rec_yards,
            'recency_weighted_rush_yards': weighted_rush_yards,
            'recent_trend_pct': recent_trend_pct,
            'momentum_score_2024vs2023': momentum_score,
            'consistency_score': consistency_score,
            'historical_seasons': len(hist_data),
            'total_data_weight': sum(d['weight'] for d in hist_data) if hist_data else 0,
            # Current season projections (highest weight)
            'projected_points': player.get('projected_points', 0),
            'fantasy_points_ppr': player.get('fantasy_points_ppr', 0),
            'adp_rank': player.get('adp_rank', 999)
        }
        
        recency_features.append(feature_record)
    
    # 8. Create final recency-weighted dataset
    recency_df = pd.DataFrame(recency_features)
    
    # 9. Calculate recency scores (combination of recent performance + momentum)
    print("🚀 Calculating final recency scores...")
    
    recency_df['recency_score'] = (
        recency_df['recency_weighted_fantasy_points'] * 0.6 +  # 60% recent weighted performance
        recency_df['momentum_score_2024vs2023'] * 0.25 +      # 25% momentum 
        recency_df['consistency_score'] * 100 * 0.15          # 15% consistency
    )
    
    # 10. Save the recency-weighted dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recency_weighted_dataset_{timestamp}.csv"
    recency_df.to_csv(filename, index=False)
    
    print(f"✅ Created recency-weighted dataset: {filename}")
    print(f"📊 Total players: {len(recency_df)}")
    print(f"📊 Players with historical data: {len(recency_df[recency_df['historical_seasons'] > 0])}")
    print(f"📊 Recency score range: {recency_df['recency_score'].min():.1f} - {recency_df['recency_score'].max():.1f}")
    
    # Show sample of players with strong recent performance
    sample_recent = recency_df.nlargest(5, 'recency_score')[['name', 'position', 'recency_score', 'momentum_score_2024vs2023', 'recent_trend_pct']]
    if not sample_recent.empty:
        print("\n📋 Top 5 players by recency score:")
        for _, player in sample_recent.iterrows():
            print(f"  {player['name']} ({player['position']}): {player['recency_score']:.1f} score, {player['momentum_score_2024vs2023']:.1f}% momentum")
    
    return recency_df

if __name__ == "__main__":
    build_recency_weighted_dataset() 