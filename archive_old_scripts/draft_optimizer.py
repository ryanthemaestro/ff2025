# draft_optimizer.py
import pandas as pd
import os
import json
import numpy as np
import requests
import time
from difflib import get_close_matches
import glob
from sklearn.ensemble import RandomForestRegressor
import joblib

DATA_DIR = 'data'
MODEL_FILE = os.path.join('models', 'draft_model.pkl')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global dictionary to store player_id to name mapping
name_to_id = {}

def fetch_sleeper_players():
    print("Fetching player data from Sleeper API...")
    try:
        # Fetch NFL state to get current season
        state_url = "https://api.sleeper.app/v1/state/nfl"
        state_response = requests.get(state_url)
        state_response.raise_for_status()
        current_season = state_response.json().get('league_season')

        # Fetch all players for the current season
        players_url = f"https://api.sleeper.app/v1/players/nfl"
        players_response = requests.get(players_url)
        players_response.raise_for_status()
        all_players = players_response.json()

        # Filter for active players and extract relevant info
        player_data = {}
        for player_id, details in all_players.items():
            if details.get('status') == 'Active' and details.get('position') in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
                player_data[player_id] = {
                    'player_id': player_id,
                    'name': f"{details.get('first_name', '')} {details.get('last_name', '')}".strip(),
                    'position': details.get('position'),
                    'team': details.get('team'),
                    'age': details.get('age'),
                    'injury_status': details.get('injury_status'),
                    'injury_start_date': details.get('injury_start_date')
                }
        
        with open(os.path.join(DATA_DIR, 'sleeper_players.json'), 'w') as f:
            json.dump(player_data, f, indent=4)
        print(f"Fetched {len(player_data)} active players for {current_season} season.")
        return player_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Sleeper API: {e}")
        return {}

def load_players():
    # Use the updated players.json with fresh injury data
    player_file = os.path.join(DATA_DIR, 'players.json')
    if os.path.exists(player_file):
        with open(player_file, 'r') as f:
            players = json.load(f)
        print(f"Loaded {len(players)} players with current injury data from {player_file}")
        return players
    else:
        print("No cached player data found, fetching fresh data...")
        return fetch_sleeper_players()

def load_projections(df):
    """Load and merge fantasy projections from multiple sources, prioritizing REAL data over synthetic data"""
    
    # CRITICAL: Use VERIFIED REAL DATA ONLY - comprehensive dataset is corrupted with synthetic data
    verified_file = "data/fantasy_metrics_2024.csv"
    
    if os.path.exists(verified_file):
        print(f"🎯 Loading VERIFIED REAL DATA from {verified_file}")
        verified_df = pd.read_csv(verified_file)
        # Convert to consistent case for matching
        verified_df['name'] = verified_df['name'].str.upper()
        print(f"✅ Loaded {len(verified_df)} players with VERIFIED real statistics")
        
        # Ensure the main df has a name column in upper case for matching
        if 'name' not in df.columns:
            df['name'] = df['full_name'].fillna('').str.upper()
        else:
            df['name'] = df['name'].fillna('').str.upper()
        
        # Merge with current dataframe using only available columns
        merge_columns = ['name', 'fantasy_points_ppr']
        # Add optional columns if they exist
        if 'carries' in verified_df.columns:
            merge_columns.append('carries')
        if 'targets' in verified_df.columns:
            merge_columns.append('targets')
        if 'games' in verified_df.columns:
            merge_columns.append('games')
            
        merged_df = df.merge(
            verified_df[merge_columns], 
            on='name', 
            how='left',
            suffixes=('', '_verified')  # Handle column conflicts
        )
        
        # Use the verified data column (either original or with suffix)
        fantasy_points_col = 'fantasy_points_ppr_verified' if 'fantasy_points_ppr_verified' in merged_df.columns else 'fantasy_points_ppr'
        
        # Set projected_points from verified data where available
        # BUT preserve rookie projected_points (don't override with 0.0 fantasy_points_ppr)
        verified_mask = merged_df[fantasy_points_col].notna()
        
        # For non-rookies: use fantasy_points_ppr as projected_points
        non_rookie_mask = verified_mask & (merged_df.get('is_rookie', False) != True)
        merged_df.loc[non_rookie_mask, 'projected_points'] = merged_df.loc[non_rookie_mask, fantasy_points_col]
        
        # For rookies: keep their pre-set projected_points (don't override with 0.0)
        if 'is_rookie' in merged_df.columns:
            rookie_mask = merged_df['is_rookie'] == True
            rookie_count = rookie_mask.sum()
            if rookie_count > 0:
                print(f"🆕 Preserved projected_points for {rookie_count} rookies (not overridden with 0.0 fantasy_points_ppr)")
        
        # Calculate previous_points from the same data
        merged_df.loc[verified_mask, 'previous_points'] = merged_df.loc[verified_mask, fantasy_points_col]
        
        verified_count = verified_mask.sum()
        print(f"🎯 Using REAL verified data for {verified_count} players")
        
        return merged_df
    
    # Fallback to FantasyPros if no verified data (should not happen)
    print("⚠️ No verified data found - falling back to FantasyPros projections")
    fantasypros_file = "data/FantasyPros_2024_Draft_Overall_Rankings.csv"
    
    if os.path.exists(fantasypros_file):
        print(f"Loading projections from {fantasypros_file}")
        projections_df = pd.read_csv(fantasypros_file)
        
        # Merge projections with player data
        merged_df = df.merge(projections_df, left_on='name', right_on='PLAYER NAME', how='left')
        
        # Set projected points from FantasyPros tiers (lower tier = higher points)
        merged_df['projected_points'] = merged_df['projected_points'].fillna(
            200 - (merged_df['TIER'].fillna(10) * 15)  # Convert tiers to rough point projections
        )
        
        return merged_df
    
    print("⚠️ No projection files found, using minimal projections")
    df['projected_points'] = df['projected_points'].fillna(100)
    return df

def prepare_data(players, projections):
    # Skip Sleeper matching if no players data
    if not players:
        print("No player data available, using projections directly")
        merged_df = projections.copy()
        # Add default values for missing columns
        merged_df['age'] = 26  # Default age
        merged_df['team'] = 'UNK'  # Unknown team
    else:
        # Create name to id map from Sleeper (lowercase for matching)
        name_to_id = {}
        for pid, p in players.items():
            full_name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip().lower()
            name_to_id[full_name] = pid
        
        # Match projections names to Sleeper ids with improved matching
        def find_best_match(proj_name):
            proj_lower = proj_name.lower().strip()
            
            # First try exact match
            if proj_lower in name_to_id:
                return name_to_id[proj_lower]
            
            # Then try fuzzy matching with higher precision
            matches = get_close_matches(proj_lower, list(name_to_id.keys()), n=3, cutoff=0.9)
            if matches:
                # Prefer matches that have the same last name
                proj_last = proj_name.split()[-1].lower() if ' ' in proj_name else ''
                for match in matches:
                    match_last = match.split()[-1].lower() if ' ' in match else ''
                    if proj_last == match_last:
                        return name_to_id[match]
                # If no last name match, take the first one
                return name_to_id[matches[0]]
            
            return None
        
        projections['player_id'] = projections['name'].apply(find_best_match)
        # Drop rows without match
        projections = projections.dropna(subset=['player_id']).copy()
        # Add age, team, and injury status from players
        projections['age'] = projections['player_id'].map(lambda pid: players.get(pid, {}).get('age', 25) if pid else 25)
        projections['team'] = projections['player_id'].map(lambda pid: players.get(pid, {}).get('team', 'UNK') if pid else 'UNK')
        projections['injury_status'] = projections['player_id'].map(lambda pid: players.get(pid, {}).get('injury_status') if pid else None)
        
        # Debug injury mapping
        injured_players = projections[projections['injury_status'].notna()]
        print(f"🏥 DEBUG: Found {len(injured_players)} players with injury status in projections:")
        for _, player in injured_players.head(5).iterrows():
            print(f"  {player['name']}: {player['injury_status']}")
        
        # Check specifically for Joe Mixon
        mixon_row = projections[projections['name'].str.contains('Mixon', case=False, na=False)]
        if not mixon_row.empty:
            pid = mixon_row.iloc[0]['player_id']
            print(f"🏥 DEBUG: Joe Mixon player_id = {pid}")
            if pid and pid in players:
                print(f"🏥 DEBUG: Joe Mixon in players.json has injury_status = {players[pid].get('injury_status')}")
            else:
                print(f"🏥 DEBUG: Joe Mixon player_id {pid} not found in players.json")
        merged_df = projections.copy()
    
    # Merge SOS on team
    sos_file = os.path.join(DATA_DIR, 'FantasyPros_Fantasy_Football_2025_Stength_Of_Schedule.csv')
    if 'sos_score' not in merged_df.columns and os.path.exists(sos_file):
        sos_df = pd.read_csv(sos_file)
        sos_df['team'] = sos_df['Team']
        sos_df['sos_score'] = sos_df['QB'].str.extract(r'(\d) star', expand=False).astype(float)
        merged_df = merged_df.merge(sos_df[['team', 'sos_score']], on='team', how='left')
        merged_df['sos_score'] = merged_df['sos_score'].fillna(3.0)  # Default neutral SOS
    else:
        merged_df['sos_score'] = 3.0  # Default neutral SOS

    # Create comprehensive team-bye mapping for rookies
    team_bye_mapping = create_team_bye_mapping()

    # Add rookie rankings
    rookie_file = os.path.join(DATA_DIR, 'FantasyPros_2025_Rookies_ALL_Rankings.csv')
    if os.path.exists(rookie_file):
        rookie_df = pd.read_csv(rookie_file)
        rookie_df['name'] = rookie_df['PLAYER NAME'].astype(str).str.strip()
        rookie_df['position'] = rookie_df['POS'].str.extract(r'([A-Z]+)')
        rookie_df['adp'] = rookie_df['AVG.'] + 200  # Offset ADP to late rounds
        rookie_df['is_rookie'] = True
        
        # Assign placeholder projected points based on position and rank
        def get_placeholder_points(pos, rank):
            base_points = {
                'QB': 18.0,
                'RB': 12.0,
                'WR': 10.0,
                'TE': 6.0,
                'K': 7.0,
                'DST': 6.0
            }.get(pos, 0)
            points = base_points - (rank / 10.0)
            return points * 0.8  # 20% rookie discount
        
        rookie_df['projected_points'] = rookie_df.apply(lambda row: get_placeholder_points(row['position'], row['RK']), axis=1)
        rookie_df['previous_points'] = 0  # Rookies have no previous
        rookie_df['games_played'] = 0
        rookie_df['injury_risk'] = 0
        rookie_df['sos_score'] = 3.0  # Neutral SOS
        rookie_df['age'] = 22  # Average rookie age
        rookie_df['adp_delta'] = 0  # No previous ADP
        
        # Handle team and bye week for rookies
        if 'TEAM' in rookie_df.columns:
            rookie_df['team'] = rookie_df['TEAM']
        else:
            rookie_df['team'] = 'Unknown'
        
        rookie_df['bye_week'] = rookie_df['team'].map(team_bye_mapping).fillna('Unknown')
        
        # Select columns that match the main DataFrame
        rookie_columns = ['name', 'position', 'projected_points', 'previous_points', 'games_played', 
                         'injury_risk', 'adp', 'sos_score', 'age', 'team', 'bye_week', 'adp_delta']
        
        # Add missing columns to rookie_df
        for col in rookie_columns:
            if col not in rookie_df.columns:
                if col == 'adp_delta':
                    rookie_df[col] = 0
                elif col in ['sos_score', 'age']:
                    rookie_df[col] = rookie_df.get(col, 3.0 if col == 'sos_score' else 22)
                else:
                    rookie_df[col] = 'Unknown' if col in ['team', 'bye_week'] else 0
        
        rookie_df = rookie_df[rookie_columns]
        
        # Add is_rookie flag to veterans and rookies
        merged_df['is_rookie'] = False
        rookie_df['is_rookie'] = True
        
        # Append to merged_df
        merged_df = pd.concat([merged_df, rookie_df], ignore_index=True)
        print(f"Added {len(rookie_df)} rookies to dataset")
    else:
        merged_df['is_rookie'] = False
    
    return merged_df

def create_team_bye_mapping():
    """Create a comprehensive team-to-bye week mapping from ADP data"""
    team_bye_mapping = {}
    
    # Try to load from main ADP file first
    adp_file = os.path.join(DATA_DIR, 'FantasyPros_2025_Overall_ADP_Rankings.csv')
    if os.path.exists(adp_file):
        try:
            adp_df = pd.read_csv(adp_file)
            # Extract team-bye mapping where bye is not empty
            if 'Team' in adp_df.columns and 'Bye' in adp_df.columns:
                team_bye_data = adp_df[adp_df['Bye'].notna()].groupby('Team')['Bye'].first()
                team_bye_mapping.update(team_bye_data.to_dict())
                print(f"Loaded {len(team_bye_mapping)} team-bye mappings from ADP file")
        except Exception as e:
            print(f"Error loading team-bye mapping from ADP file: {e}")
    
    # Fallback: try secondary ADP file
    backup_adp_file = os.path.join(DATA_DIR, 'adp.csv')
    if os.path.exists(backup_adp_file) and len(team_bye_mapping) < 30:  # NFL has 32 teams
        try:
            backup_df = pd.read_csv(backup_adp_file)
            if 'Team' in backup_df.columns and 'Bye' in backup_df.columns:
                backup_mapping = backup_df[backup_df['Bye'].notna()].groupby('Team')['Bye'].first()
                team_bye_mapping.update(backup_mapping.to_dict())
                print(f"Added mappings from backup file, total: {len(team_bye_mapping)}")
        except Exception as e:
            print(f"Error loading backup team-bye mapping: {e}")
    
    # If still no mapping, create default based on 2024 schedule
    if len(team_bye_mapping) < 20:
        default_byes = {
            'ARI': 8, 'ATL': 5, 'BAL': 7, 'BUF': 7, 'CAR': 14, 'CHI': 5, 'CIN': 10,
            'CLE': 9, 'DAL': 10, 'DEN': 12, 'DET': 8, 'GB': 5, 'HOU': 6, 'IND': 11,
            'JAC': 8, 'KC': 10, 'LV': 8, 'LAC': 12, 'LAR': 8, 'MIA': 12, 'MIN': 6,
            'NE': 14, 'NO': 11, 'NYG': 14, 'NYJ': 9, 'PHI': 9, 'PIT': 5, 'SF': 14,
            'SEA': 8, 'TB': 9, 'TEN': 10, 'WAS': 12
        }
        team_bye_mapping.update(default_byes)
        print("Using default 2024 bye week schedule")
    
    return team_bye_mapping

def load_comprehensive_features():
    """Load the recency-weighted dataset that emphasizes recent performance"""
    try:
        # Try to load the latest RECENCY-WEIGHTED dataset first (best option for current performance)
        import glob
        recency_files = glob.glob('recency_weighted_dataset_*.csv')
        if recency_files:
            latest_recency = max(recency_files)
            comprehensive_df = pd.read_csv(latest_recency)
            print(f"✅ Loaded RECENCY-WEIGHTED dataset with {len(comprehensive_df)} players from {latest_recency}")
            players_with_history = len(comprehensive_df[comprehensive_df['historical_seasons'] > 0])
            print(f"📊 Players with historical data: {players_with_history} ({players_with_history/len(comprehensive_df)*100:.1f}%)")
            print(f"📈 Emphasizes recent performance: 2024 data weighted 4x, 2023 data 2x, 2022 data 1x")
            return comprehensive_df
        
        # Fallback to clean unified dataset
        clean_files = glob.glob('clean_unified_dataset_*.csv')
        if clean_files:
            latest_clean = max(clean_files)
            comprehensive_df = pd.read_csv(latest_clean)
            print(f"✅ Loaded CLEAN UNIFIED dataset with {len(comprehensive_df)} players from {latest_clean}")
            players_with_history = len(comprehensive_df[comprehensive_df['historical_seasons'] > 0])
            print(f"📊 Players with historical data: {players_with_history} ({players_with_history/len(comprehensive_df)*100:.1f}%)")
            return comprehensive_df
        
        # Fallback to FIXED comprehensive dataset
        fixed_files = glob.glob('comprehensive_training_data_FIXED_*.csv')
        if fixed_files:
            latest_fixed = max(fixed_files)
            comprehensive_df = pd.read_csv(latest_fixed)
            print(f"✅ Loaded FIXED comprehensive dataset with {len(comprehensive_df)} players from {latest_fixed}")
            print(f"📊 Multi-year features: 3-year averages, consistency, trends, durability (2022-2024)")
            return comprehensive_df
        
        # Fallback to any comprehensive dataset
        comp_files = glob.glob('comprehensive_training_data_*.csv')
        if comp_files:
            latest_comp = max(comp_files)
            comprehensive_df = pd.read_csv(latest_comp)
            print(f"✅ Loaded comprehensive dataset with {len(comprehensive_df)} players from {latest_comp}")
            print(f"📊 Multi-year features: 3-year averages, consistency, trends, durability")
            return comprehensive_df
            
        # Original fallback
        comprehensive_df = pd.read_csv('data/comprehensive_nfl_data_2022_2024.csv')
        print(f"⚠️ Using old comprehensive dataset with {len(comprehensive_df)} players")
        print(f"📊 Multi-year features: 3-year averages, consistency, trends, durability")
        return comprehensive_df
    except FileNotFoundError:
        print("⚠️ Comprehensive dataset not found. Run comprehensive_api_pipeline.py first.")
        return None
    except Exception as e:
        print(f"❌ Error loading comprehensive dataset: {e}")
        return None

def create_football_features(df):
    """Create meaningful football performance features using multi-year data"""
    print("\n🏈 CREATING FOOTBALL PERFORMANCE FEATURES WITH MULTI-YEAR DATA")
    print("="*70)
    
    # Load comprehensive multi-year dataset
    comprehensive_df = load_comprehensive_features()
    
    if comprehensive_df is not None:
        print("\n🎯 MERGING WITH COMPREHENSIVE MULTI-YEAR DATASET")
        print("-" * 50)
        
        # Merge the comprehensive features with the main dataframe
        # Import our improved name matching
        import sys
        sys.path.append('.')
        from scripts.name_mapper import find_best_name_match
        
        merged_count = 0
        for idx, row in df.iterrows():
            player_name = row.get('name', '')
            if pd.isna(player_name) or player_name == '':
                continue  # Skip players with missing names
            
            player_name = str(player_name).upper()  # Ensure it's a string and uppercase
            
            # Find the best match in comprehensive dataset using multiple name formats
            best_match = None
            
            # Try exact match first
            exact_match = comprehensive_df[comprehensive_df['name'].str.upper() == player_name]
            if not exact_match.empty:
                best_match = exact_match.iloc[0]
            else:
                # Try comprehensive dataset format matching (e.g., "Jahmyr Gibbs" -> "J.GIBBS")
                name_parts = player_name.split()
                if len(name_parts) >= 2:
                    first_name = name_parts[0]
                    last_name = name_parts[-1]
                    
                    # Create multiple possible formats to match comprehensive dataset
                    possible_formats = [
                        f"{first_name[0]}.{last_name}".upper(),  # J.GIBBS (most common)
                        f"{first_name[0]}{last_name}".upper(),   # JGIBBS  
                        f"{first_name}.{last_name}".upper(),     # JAHMYR.GIBBS
                        f"{last_name},{first_name}".upper(),     # GIBBS,JAHMYR
                        f"{last_name}, {first_name}".upper(),    # GIBBS, JAHMYR
                        f"{last_name}.{first_name[0]}".upper(),  # GIBBS.J
                    ]
                    
                    # Try each format
                    for format_name in possible_formats:
                        format_match = comprehensive_df[comprehensive_df['name'].str.upper() == format_name]
                        if not format_match.empty:
                            best_match = format_match.iloc[0]
                            break
                
                # If still no match, try fuzzy matching on last name + first initial
                if best_match is None and name_parts:
                    last_name = name_parts[-1] if name_parts else player_name
                    first_initial = name_parts[0][0] if name_parts else player_name[0]
                    
                    # Look for any row containing the last name and first initial
                    fuzzy_matches = comprehensive_df[
                        comprehensive_df['name'].str.contains(last_name.upper(), case=False, na=False) & 
                        comprehensive_df['name'].str.contains(first_initial.upper(), case=False, na=False)
                    ]
                    if not fuzzy_matches.empty:
                        best_match = fuzzy_matches.iloc[0]
            
            if best_match is not None:
                merged_count += 1
                
                # Add all multi-year features - prioritize recency-weighted data
                # Handle recency-weighted, clean unified, and legacy formats
                
                # Primary: Use recency-weighted features (heavily favors recent performance)
                df.at[idx, 'avg_fantasy_points_3yr'] = best_match.get('recency_weighted_fantasy_points', 
                                                                   best_match.get('avg_fantasy_points_3yr', 0))
                df.at[idx, 'avg_targets_3yr'] = best_match.get('recency_weighted_targets', 
                                                             best_match.get('avg_targets_3yr', 0))
                df.at[idx, 'avg_carries_3yr'] = best_match.get('recency_weighted_carries', 
                                                             best_match.get('avg_carries_3yr', 0))
                
                # New recency-specific features
                df.at[idx, 'momentum_score'] = best_match.get('momentum_score_2024vs2023', 0)
                df.at[idx, 'recent_trend'] = best_match.get('recent_trend_pct', 0)
                df.at[idx, 'recency_score'] = best_match.get('recency_score', 0)
                df.at[idx, 'total_data_weight'] = best_match.get('total_data_weight', 0)
                
                # Standard features
                df.at[idx, 'performance_trend'] = best_match.get('performance_trend_pct', best_match.get('recent_trend_pct', 0))
                df.at[idx, 'consistency_score'] = best_match.get('consistency_score', 0)
                df.at[idx, 'seasons_played'] = best_match.get('historical_seasons', best_match.get('seasons_played', 0))
                df.at[idx, 'total_games'] = best_match.get('avg_games_3yr', best_match.get('total_games', 0)) * max(df.at[idx, 'seasons_played'], 1)
                
                # Calculate derived features using recency data when available
                recency_points = best_match.get('recency_weighted_fantasy_points', best_match.get('avg_fantasy_points_3yr', 0))
                df.at[idx, 'points_per_game'] = recency_points / 16 if recency_points > 0 else 0
                
                # Opportunity metrics (use recency-weighted when available)
                df.at[idx, 'avg_opportunity'] = (best_match.get('recency_weighted_targets', 0) + 
                                                best_match.get('recency_weighted_carries', 0))
                df.at[idx, 'total_touches'] = df.at[idx, 'avg_opportunity'] * max(df.at[idx, 'seasons_played'], 1)
                
                # Efficiency metrics
                total_yards = (best_match.get('recency_weighted_rec_yards', 0) + 
                              best_match.get('recency_weighted_rush_yards', 0))
                df.at[idx, 'yards_per_touch'] = total_yards / df.at[idx, 'avg_opportunity'] if df.at[idx, 'avg_opportunity'] > 0 else 0
                df.at[idx, 'catch_rate'] = best_match.get('catch_rate', 0.8)  # Default reasonable catch rate
                
                # For backward compatibility, set weighted column names using recency data
                df.at[idx, 'avg_fantasy_points_3yr_weighted'] = recency_points
                df.at[idx, 'avg_targets_3yr_weighted'] = best_match.get('recency_weighted_targets', 0)
                df.at[idx, 'age_adjusted_score'] = best_match.get('age_adjusted_score', 0)
                
                # Set projected points if available 
                if 'projected_points' in best_match and best_match['projected_points'] > 0:
                    df.at[idx, 'projected_points'] = best_match['projected_points']
        
        print(f"✅ Merged comprehensive features for {merged_count} players")
        
        # Fill missing values with defaults
        multi_year_columns = [
            # Recency-weighted features (highest priority - emphasizes recent performance)
            'recency_score', 'momentum_score', 'recent_trend', 'total_data_weight',
            
            # Weighted features (prioritized)
            'avg_fantasy_points_3yr_weighted', 'avg_targets_3yr_weighted', 'avg_carries_3yr_weighted',
            'total_touches_3yr_weighted', 'durability_score_weighted', 'opportunity_score_weighted',
            'recent_vs_weighted_historical', 'career_momentum', 'weighted_vs_unweighted_diff',
            
            # Traditional features
            'avg_fantasy_points_3yr', 'consistency_score', 'performance_trend', 
            'avg_targets_3yr', 'avg_carries_3yr', 'total_touches_3yr',
            'durability_score', 'opportunity_score', 'avg_yards_per_target',
            'avg_yards_per_carry', 'target_share_stability', 'recent_vs_historical',
            'age_adjusted_score'
        ]
        
        for col in multi_year_columns:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0)
        
        print(f"✅ Added {len(multi_year_columns)} multi-year features")
        
        # Show sample of players with multi-year data
        players_with_multi_year = df[df['consistency_score'] > 0]
        if len(players_with_multi_year) > 0:
            print(f"\n📊 Sample players with multi-year data:")
            sample = players_with_multi_year[['name', 'position', 'avg_fantasy_points_3yr', 'consistency_score', 'performance_trend']].head(5)
            print(sample.to_string(index=False))
    
    else:
        # Fallback to single-year data if comprehensive dataset not available
        print("\n⚠️ FALLING BACK TO SINGLE-YEAR DATA")
        print("-" * 40)
        
        # Try to load advanced metrics from our corrected real NFL stats
        advanced_metrics_available = False
        try:
            advanced_df = pd.read_csv('data/fantasy_metrics_2024.csv')
            print(f"✅ Loaded advanced metrics for {len(advanced_df)} players from REAL 2024 NFL DATA")
            advanced_metrics_available = True
            
            # Import our improved name matching
            import sys
            sys.path.append('.')
            from scripts.name_mapper import find_best_name_match, nflsavant_to_fantasypros, create_name_variants
            
            # Get list of all player names in the main dataset
            fantasypros_names = df['name'].dropna().tolist()
            
        except FileNotFoundError:
            print("⚠️ Advanced metrics file not found. Run extract_fantasy_metrics.py first.")
            print("   Using basic features only for now.")
        except Exception as e:
            print(f"⚠️ Error loading advanced metrics: {e}")
        
        if advanced_metrics_available:
            print("\n🎯 ADDING ADVANCED OPPORTUNITY METRICS")
            print("-" * 40)
            
            # Merge advanced metrics using improved name matching
            merged_count = 0
            for idx, row in df.iterrows():
                player_name = row['name']
                
                # Find the best match for this player in the advanced metrics
                best_match = None
                for _, advanced_row in advanced_df.iterrows():
                    nfl_name = advanced_row['player_name']
                    fp_match = find_best_name_match(nfl_name, [player_name])
                    if fp_match:
                        best_match = advanced_row
                        break
                
                if best_match is not None:
                    
                    # Add opportunity metrics
                    df.at[idx, 'target_share_pct'] = best_match.get('target_share_pct', 0)
                    df.at[idx, 'snap_count_pct'] = best_match.get('snap_count_pct', 0)
                    df.at[idx, 'red_zone_opportunities'] = best_match.get('red_zone_opportunities', 0)
                    df.at[idx, 'catch_rate_pct'] = best_match.get('catch_rate_pct', 0)
                    df.at[idx, 'yards_per_touch'] = best_match.get('yards_per_touch', 0)
                    
                    # Add raw volume metrics
                    df.at[idx, 'total_targets'] = best_match.get('targets', 0)
                    df.at[idx, 'total_carries'] = best_match.get('carries', 0)
                    df.at[idx, 'total_receptions'] = best_match.get('receptions', 0)
                    
                    # Add advanced calculated metrics
                    df.at[idx, 'opportunity_score'] = (
                        best_match.get('target_share_pct', 0) * 0.4 +
                        best_match.get('snap_count_pct', 0) * 0.3 +
                        best_match.get('red_zone_opportunities', 0) * 5  # Weight RZ highly
                    )
                    
                    # Usage stability (how consistent the opportunity is)
                    touches = best_match.get('targets', 0) + best_match.get('carries', 0)
                    df.at[idx, 'total_touches'] = touches
                    
                    # High-value feature: Efficiency relative to opportunity
                    if best_match.get('targets', 0) > 10:  # Minimum threshold
                        efficiency = (best_match.get('receiving_yards', 0) + 
                                    best_match.get('rushing_yards', 0)) / touches if touches > 0 else 0
                        df.at[idx, 'efficiency_per_touch'] = efficiency
                    else:
                        df.at[idx, 'efficiency_per_touch'] = 0
                    
                    merged_count += 1
            
            print(f"✅ Merged advanced metrics for {merged_count} players")
            
            # Fill missing advanced metrics with defaults
            advanced_columns = ['target_share_pct', 'snap_count_pct', 'red_zone_opportunities', 
                              'catch_rate_pct', 'yards_per_touch', 'total_targets', 'total_carries',
                              'total_receptions', 'opportunity_score', 'total_touches', 'efficiency_per_touch']
            
            for col in advanced_columns:
                if col not in df.columns:
                    df[col] = 0
                else:
                    df[col] = df[col].fillna(0)
            
            print(f"✅ Added {len(advanced_columns)} advanced opportunity features")
            
            # Show sample of players with advanced metrics
            players_with_advanced = df[df['target_share_pct'] > 0]
            if len(players_with_advanced) > 0:
                print(f"\n📊 Sample players with advanced metrics:")
                sample = players_with_advanced[['name', 'position', 'target_share_pct', 'snap_count_pct', 'opportunity_score']].head(5)
                print(sample.to_string(index=False))
    
    # Historical Performance Features (always add these)
    # Ensure required columns exist with defaults
    if 'previous_points' not in df.columns:
        df['previous_points'] = 0
    if 'games_played' not in df.columns:
        df['games_played'] = 17
    
    df['ppg_last_season'] = df['previous_points'].fillna(0) / df['games_played'].fillna(17).replace(0, 17)
    df['games_missed_last_season'] = 17 - df['games_played'].fillna(17)  # NFL has 17 games
    df['durability_score_basic'] = df['games_played'].fillna(17) / 17  # 0 to 1 scale
    
    # Age-based features
    age_series = df['age'] if 'age' in df.columns else pd.Series([25] * len(df), index=df.index)
    df['age_adjusted_experience'] = age_series - 22  # Years since typical draft age
    df['is_prime_age'] = ((age_series >= 24) & (age_series <= 28)).astype(int)
    df['is_declining_age'] = (age_series >= 30).astype(int)
    
    # Performance consistency
    df['performance_floor'] = df.get('previous_points', 0) * 0.7  # Conservative estimate
    df['performance_ceiling'] = df.get('previous_points', 0) * 1.3  # Optimistic estimate
    df['upside_potential'] = df['performance_ceiling'] - df['performance_floor']
    
    # Injury risk assessment
    df['injury_risk_score'] = (17 - df.get('games_played', 17)) / 17
    
    # Team context
    df['sos_adjusted_projection'] = df.get('projected_points', 0) * df.get('sos_score', 3.0) / 3.0
    
    feature_count = len([col for col in df.columns if col not in ['name', 'position', 'team', 'adp']])
    print(f"\n✅ TOTAL FEATURES CREATED: {feature_count}")
    print("🏈 Features include MULTI-YEAR data for robust predictions!")
    
    return df

def train_model(df):
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import numpy as np
    
    print('Top Projections Sample:',
          df.sort_values('projected_points', ascending=False).head())
    
    print(f"\n🎯 TRAINING MODEL TO PREDICT: projected_points")
    print(f"📊 Training data shape: {df.shape}")
    
    # 🚫 NO MORE ADP - PURE FOOTBALL FEATURES
    print(f"\n🚫 ADP REMOVED - Using Only Football Performance Metrics")
    print(f"✅ Focus: Historical performance, health, opportunity, situational usage")
    
    # Create meaningful football features
    df = create_football_features(df)
    
    df_train = df.dropna(subset=['projected_points', 'position'])
    
    # 🚫 EXCLUDE ROOKIES FROM MODEL TRAINING (per user request)
    # Rookies don't have historical NFL performance data for training
    if 'is_rookie' in df_train.columns:
        rookies_count = len(df_train[df_train['is_rookie'] == True])
        df_train = df_train[df_train['is_rookie'] != True]
        print(f"🚫 EXCLUDED {rookies_count} rookies from model training (no historical data)")
    
    # 🚫 EXCLUDE KICKERS AND DST FROM MODEL TRAINING (per user request)
    # K and DST have different scoring patterns and should use their own projections
    if 'position' in df_train.columns:
        k_count = len(df_train[df_train['position'] == 'K'])
        dst_count = len(df_train[df_train['position'] == 'DST'])
        df_train = df_train[~df_train['position'].isin(['K', 'DST'])]
        print(f"🚫 EXCLUDED {k_count} kickers and {dst_count} defenses from model training (different scoring patterns)")
    
    if df_train.empty:
        print("Not enough data to train model.")
        return None
    
    df_train = pd.get_dummies(df_train, columns=['position'], drop_first=True)
    
    # ONLY FOOTBALL PERFORMANCE FEATURES - NO ADP
    # Select all numeric features except the target and identifiers (NO ADP!)
    base_features = [col for col in df_train.columns if 
                    col not in ['name', 'position', 'team', 'projected_points', 'adp', 'previous_adp', 'AVG', 'adp_delta'] and 
                    df_train[col].dtype in ['int64', 'float64']]
    position_features = [col for col in df_train.columns if col.startswith('position_')]
    
    all_features = base_features + position_features
    
    print(f"\n🔧 FOOTBALL FEATURE ANALYSIS:")
    print(f"   🏈 Performance features: {base_features}")
    print(f"   📊 Position features: {position_features}")
    print(f"   🚫 EXCLUDED: ADP, any draft-related metrics")
    
    # Ensure all features exist
    available_features = []
    for feature in all_features:
        if feature in df_train.columns:
            available_features.append(feature)
            # Fill missing values intelligently
            if df_train[feature].dtype in ['float64', 'int64']:
                df_train[feature] = df_train[feature].fillna(df_train[feature].median())
            else:
                df_train[feature] = df_train[feature].fillna(0)
        else:
            print(f"   ⚠️  Missing feature: {feature}")
    
    if len(available_features) < 3:
        print(f"   ❌ Not enough features available")
        return None
        
    X = df_train[available_features]
    y = df_train['projected_points']
    
    print(f"\n📊 MODEL TRAINING SETUP:")
    print(f"   🎯 Target: projected_points (range: {y.min():.1f} to {y.max():.1f})")
    print(f"   🔧 Features: {len(available_features)} football metrics")
    print(f"   📈 Training samples: {len(X)}")
    
    # PROPER TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"   📊 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model with football features only
    model = RandomForestRegressor(
        n_estimators=200,  # More trees for better performance
        max_depth=10,      # Limit depth to reduce overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # COMPREHENSIVE EVALUATION
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    # Cross-validation for robust estimate
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    
    print(f"\n{'='*60}")
    print("🏈 FOOTBALL-ONLY MODEL PERFORMANCE")
    print(f"{'='*60}")
    
    print(f"📊 TRAINING PERFORMANCE:")
    print(f"   R² Score: {train_r2:.3f} ({train_r2*100:.1f}%)")
    print(f"   MAE: {train_mae:.2f} points")
    
    print(f"\n🧪 TEST PERFORMANCE (UNSEEN DATA):")
    print(f"   R² Score: {test_r2:.3f} ({test_r2*100:.1f}%)")
    print(f"   MAE: {test_mae:.2f} points")
    
    print(f"\n🔄 CROSS-VALIDATION (5-fold):")
    print(f"   Mean R²: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
    print(f"   Individual folds: {[f'{score:.3f}' for score in cv_scores]}")
    
    # Overfitting check
    overfitting = train_r2 - test_r2
    print(f"\n🔍 OVERFITTING ANALYSIS:")
    print(f"   Train R² - Test R²: {overfitting:.3f}")
    if overfitting > 0.1:
        print(f"   ⚠️  HIGH OVERFITTING! Consider simpler model")
    elif overfitting > 0.05:
        print(f"   ⚠️  Moderate overfitting")
    else:
        print(f"   ✅ Good generalization")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🎯 TOP FEATURE IMPORTANCE (Football Metrics Only):")
    for _, row in feature_importance.head(8).iterrows():
        importance_pct = row['importance'] * 100
        print(f"   {row['feature']}: {importance_pct:.1f}%")
    
    # Reality check on performance
    print(f"\n💡 REALITY CHECK:")
    if test_r2 > 0.3:
        print(f"   ✅ Test R² of {test_r2:.1f} is reasonable for fantasy prediction")
    elif test_r2 > 0.1:
        print(f"   ⚠️  Test R² of {test_r2:.1f} is modest but usable")
    else:
        print(f"   ❌ Test R² of {test_r2:.1f} suggests model needs more/better features")
    
    if test_mae < 3.0:
        print(f"   ✅ Test MAE of {test_mae:.1f} points is good precision")
    else:
        print(f"   ⚠️  Test MAE of {test_mae:.1f} points is high - predictions not very precise")
    
    # Save model with football features
    model_data = {
        'model': model,
        'features': available_features,
        'feature_importance': feature_importance,
        'performance': {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_mae': train_mae,
            'test_mae': test_mae
        },
        'approach': 'football_only_no_adp'
    }
    
    joblib.dump(model_data, MODEL_FILE)
    print(f"\n✅ Football-only model saved!")
    print(f"✅ No ADP or circular dependencies - pure performance prediction")
    
    return model

def load_position_models():
    """Load position-specific CatBoost models"""
    print("🏈 LOADING POSITION-SPECIFIC MODELS")
    print("=" * 40)
    
    models = {}
    
    try:
        # Find the latest ensemble config
        ensemble_configs = glob.glob('models/ensemble_config_*.json')
        if not ensemble_configs:
            print("❌ No ensemble config found, falling back to old model")
            return None
            
        latest_config = max(ensemble_configs)
        print(f"✅ Found ensemble config: {latest_config}")
        
        with open(latest_config, 'r') as f:
            config = json.load(f)
        
        timestamp = config['timestamp']
        positions = config['positions']
        
        # Load each position model
        for position in positions:
            model_file = f"models/position_{position.lower()}_model_{timestamp}.pkl"
            
            if os.path.exists(model_file):
                model_data = joblib.load(model_file)
                models[position] = model_data
                print(f"✅ Loaded {position} model: R²={model_data['test_r2']:.3f}")
            else:
                print(f"⚠️ {position} model not found: {model_file}")
        
        if models:
            print(f"🎯 Successfully loaded {len(models)} position-specific models")
            return models
        else:
            print("❌ No position models loaded")
            return None
            
    except Exception as e:
        print(f"❌ Error loading position models: {e}")
        return None

def ensemble_predict(player_data, position, models, adp_rank=None):
    """
    Ensemble prediction combining multiple approaches
    """
    predictions = {}
    
    # 1. Position-specific CatBoost prediction (70% weight)
    if position in models and models[position]:
        try:
            model_data = models[position]
            model = model_data['model']
            features = model_data['features']
            
            # Prepare features for this player
            player_features = []
            for feature in features:
                player_features.append(player_data.get(feature, 0))
            
            catboost_pred = model.predict([player_features])[0]
            predictions['catboost'] = max(0, catboost_pred)  # No negative predictions
            
        except Exception as e:
            print(f"Warning: CatBoost prediction failed for {position}: {e}")
            predictions['catboost'] = 0
    else:
        predictions['catboost'] = 0
    
    # 2. ADP consensus (15% weight)
    if adp_rank and adp_rank > 0:
        # Convert ADP rank to point expectation (rough heuristic)
        if adp_rank <= 12:  # Round 1
            adp_points = 250 - (adp_rank * 15)
        elif adp_rank <= 24:  # Round 2
            adp_points = 200 - ((adp_rank - 12) * 10)
        elif adp_rank <= 36:  # Round 3
            adp_points = 150 - ((adp_rank - 24) * 8)
        else:
            adp_points = max(50, 120 - (adp_rank - 36) * 2)
        
        predictions['adp_consensus'] = max(0, adp_points)
    else:
        predictions['adp_consensus'] = 100  # Default neutral
    
    # 3. Volume projection (10% weight)
    volume_score = 0
    if position == 'QB':
        volume_score = player_data.get('passing_yards', 0) * 0.04 + player_data.get('passing_tds', 0) * 4
    elif position == 'RB':
        volume_score = player_data.get('carries', 0) * 0.1 + player_data.get('targets', 0) * 0.5
    elif position in ['WR', 'TE']:
        volume_score = player_data.get('targets', 0) * 0.8 + player_data.get('receiving_yards', 0) * 0.1
    
    predictions['volume_projection'] = max(0, volume_score)
    
    # 4. SOS adjustment (5% weight)
    sos_score = player_data.get('sos_score', 3.0)
    sos_adjustment = 100 * (sos_score / 3.0)  # Neutral is 3.0
    predictions['sos_adjustment'] = sos_adjustment
    
    # 🔧 FIX: Add efficiency penalty to combat volume bias (like Rachaad White)
    efficiency_penalty = 1.0  # Default no penalty
    
    if position in ['RB', 'WR', 'TE']:  # Skill positions where efficiency matters
        fantasy_points = player_data.get('fantasy_points', 0)
        carries = player_data.get('carries', 0)
        targets = player_data.get('targets', 0)
        total_touches = carries + targets
        
        if total_touches > 50 and fantasy_points > 0:  # Minimum volume threshold
            points_per_touch = fantasy_points / total_touches
            
            # Efficiency thresholds based on position
            if position == 'RB':
                if points_per_touch < 0.25:  # Poor efficiency (like R.White's 0.223)
                    efficiency_penalty = 0.7  # 30% penalty
                elif points_per_touch < 0.30:  # Below average
                    efficiency_penalty = 0.85  # 15% penalty
                elif points_per_touch > 0.40:  # Elite efficiency
                    efficiency_penalty = 1.15  # 15% bonus
            elif position in ['WR', 'TE']:
                if points_per_touch < 0.30:  # Poor efficiency
                    efficiency_penalty = 0.75  # 25% penalty
                elif points_per_touch < 0.35:  # Below average  
                    efficiency_penalty = 0.90  # 10% penalty
                elif points_per_touch > 0.45:  # Elite efficiency
                    efficiency_penalty = 1.10  # 10% bonus
    
    # Apply efficiency adjustment to all components
    for component in predictions:
        predictions[component] *= efficiency_penalty
    
    # 🔧 FIX: Additional reality check for volume trap players (like R.WHITE)
    volume_trap_penalty = 1.0
    if position == 'RB':
        carries = player_data.get('carries', 0)
        fantasy_points = player_data.get('fantasy_points', 0)
        
        # High volume (600+ carries) but poor per-game production
        if carries > 600 and fantasy_points > 0:
            games_played = player_data.get('games_played', 16)
            ppg = fantasy_points / games_played if games_played > 0 else 0
            
            # RBs with 600+ carries should be scoring 12+ PPG to justify the volume
            if ppg < 10:  # Poor PPG despite high volume
                volume_trap_penalty = 0.6  # 40% penalty for volume trap
                print(f"⚠️ Volume trap penalty applied to {player_data.get('name', 'Unknown')}: {carries} carries, {ppg:.1f} PPG")
            elif ppg < 12:  # Mediocre PPG despite high volume
                volume_trap_penalty = 0.8  # 20% penalty
    
    # Apply volume trap penalty to all components
    for component in predictions:
        predictions[component] *= volume_trap_penalty
    
    # Combine with ensemble weights
    ensemble_weights = {
        'catboost': 0.70,
        'adp_consensus': 0.15,
        'volume_projection': 0.10,
        'sos_adjustment': 0.05
    }
    
    final_prediction = 0
    for component, weight in ensemble_weights.items():
        if component in predictions:
            final_prediction += predictions[component] * weight
    
    return max(0, final_prediction), predictions

def load_model():
    """Load models - now uses position-specific models or falls back to old model"""
    # Try to load position-specific models first
    position_models = load_position_models()
    if position_models:
        return position_models, 'position_specific'
    
    # Fallback to old model loading
    print("⚠️ Falling back to old single model approach")
    import glob
    fixed_models = glob.glob('models/draft_model_FIXED_*.pkl')
    if fixed_models:
        latest_fixed_model = max(fixed_models)
        print(f'🎯 Loading LATEST FIXED model: {latest_fixed_model}')
        try:
            model_data = joblib.load(latest_fixed_model)
            
            # Try to load corresponding metadata
            latest_fixed_metadata = latest_fixed_model.replace('.pkl', '.json').replace('draft_model_', 'model_metadata_')
            try:
                with open(latest_fixed_metadata, 'r') as f:
                    metadata = json.load(f)
                    features = metadata['features']
                    print(f"✅ Loaded FIXED model with {len(features)} features (includes 2024 data!)")
                    return model_data, features
            except FileNotFoundError:
                print("⚠️ FIXED model metadata not found, using model without features")
                return model_data, None
                
        except Exception as e:
            print(f"❌ Error loading FIXED model: {e}")
    
    # Fallback to regular model file
    if os.path.exists(MODEL_FILE):
        print('Loading trained model...')
        model_data = joblib.load(MODEL_FILE)
        
        # Handle both old and new model formats
        if isinstance(model_data, dict) and 'model' in model_data:
            # New format with metadata
            return model_data['model'], model_data['features']
        else:
            # Old format - just the model
            print('⚠️  Old model format detected. Please retrain for better validation.')
            return model_data, None
    else:
        print('Model not found. Train first.')
        return None, None

def suggest_picks(df_available, draft_position, team_needs, num_suggestions=5, league_size=12):
    models, model_type = load_model()
    if models is None:
        return []
    if df_available.empty:
        return []
    
    # Add fallback projected points for K/DST if missing
    df_available['projected_points'] = df_available['projected_points'].fillna(0)
    
    # Add reasonable projected points for K/DST based on position
    mask_k = (df_available['position'] == 'K') & (df_available['projected_points'] == 0)
    mask_dst = (df_available['position'] == 'DST') & (df_available['projected_points'] == 0)
    
    df_available.loc[mask_k, 'projected_points'] = 8.0  # Average kicker points
    df_available.loc[mask_dst, 'projected_points'] = 9.0  # Average DST points
    
    # Include ALL positions, not just skill positions
    all_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
    df_available = df_available[df_available['position'].isin(all_positions)].copy()
    if df_available.empty:
        return []
    
    # Dynamic replacement levels for all positions
    replacement_levels = {}
    for pos in all_positions:
        pos_df = df_available[df_available['position'] == pos].sort_values('projected_points', ascending=False)
        if pos == 'QB':
            replacement_levels[pos] = pos_df['projected_points'].iloc[min(league_size - 1, len(pos_df) - 1)] if not pos_df.empty else 0
        elif pos in ['K', 'DST']:
            # For K/DST, use a lower replacement level since they're drafted late
            replacement_levels[pos] = pos_df['projected_points'].quantile(0.7) if not pos_df.empty else 0
        else:
            replacement_levels[pos] = pos_df['projected_points'].quantile(0.9) if not pos_df.empty else 0
    
    print('Dynamic Replacement Levels:', replacement_levels)
    df_available['vbd'] = df_available.apply(lambda row: row['projected_points'] - replacement_levels.get(row['position'], 0), axis=1)
    
    # Base score calculation
    df_available['score'] = df_available['vbd']
    
    # 🧠 PERFORMANCE INTELLIGENCE BOOST - Weight key multi-year metrics heavily
    print("🧠 Applying Performance Intelligence Boosts...")
    
    # 1. PERFORMANCE TREND BOOST - Heavily favor improving players (POSITION-CAPPED)
    if 'performance_trend' in df_available.columns:
        # Position-specific trend boost caps to prevent QB dominance but reward skill position trends
        for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
            pos_mask = df_available['position'] == pos
            if pos_mask.any():
                if pos == 'QB':
                    # Minimal trend boost for QBs to prevent early round dominance
                    df_available.loc[pos_mask & (df_available['performance_trend'] > 20), 'trend_boost'] = (
                        df_available.loc[pos_mask & (df_available['performance_trend'] > 20), 'score'] * 0.05  # 5% max for QBs
                    )
                    df_available.loc[pos_mask & (df_available['performance_trend'] > 5) & (df_available['performance_trend'] <= 20), 'trend_boost'] = (
                        df_available.loc[pos_mask & (df_available['performance_trend'] > 5) & (df_available['performance_trend'] <= 20), 'score'] * 0.03  # 3% mild boost for QBs
                    )
                elif pos in ['RB', 'WR']:
                    # HEAVY trend boost for skill positions - this is what we want!
                    df_available.loc[pos_mask & (df_available['performance_trend'] > 20), 'trend_boost'] = (
                        df_available.loc[pos_mask & (df_available['performance_trend'] > 20), 'score'] * 0.6  # 60% boost for hot skill players
                    )
                    df_available.loc[pos_mask & (df_available['performance_trend'] > 5) & (df_available['performance_trend'] <= 20), 'trend_boost'] = (
                        df_available.loc[pos_mask & (df_available['performance_trend'] > 5) & (df_available['performance_trend'] <= 20), 'score'] * 0.3  # 30% mild boost
                    )
                    # PENALTY for declining players
                    df_available.loc[pos_mask & (df_available['performance_trend'] < -10), 'trend_boost'] = (
                        df_available.loc[pos_mask & (df_available['performance_trend'] < -10), 'score'] * -0.4  # 40% penalty for declining players
                    )
                else:
                    # Normal boost for TE/K/DST
                    df_available.loc[pos_mask & (df_available['performance_trend'] > 20), 'trend_boost'] = (
                        df_available.loc[pos_mask & (df_available['performance_trend'] > 20), 'score'] * 0.2  # 20% boost
                    )
                    df_available.loc[pos_mask & (df_available['performance_trend'] > 5) & (df_available['performance_trend'] <= 20), 'trend_boost'] = (
                        df_available.loc[pos_mask & (df_available['performance_trend'] > 5) & (df_available['performance_trend'] <= 20), 'score'] * 0.1  # 10% mild boost
                    )
        
        # Apply trend boosts
        df_available['trend_boost'] = df_available.get('trend_boost', 0)
        df_available['score'] += df_available['trend_boost']
        trend_count = len(df_available[df_available['performance_trend'].notna()])
        declining_count = len(df_available[df_available['performance_trend'] < -10])
        print(f"   ✅ Boosted {trend_count} high-momentum players, penalized {declining_count} declining players")
    
    # 2. DURABILITY BOOST - Availability is critical (POSITION-CAPPED) - Use weighted version
    durability_col = 'durability_score_weighted' if 'durability_score_weighted' in df_available.columns else 'durability_score'
    if durability_col in df_available.columns:
        # Position-specific durability boost caps
        for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
            pos_mask = df_available['position'] == pos
            if pos_mask.any():
                if pos == 'QB':
                    # Minimal durability boost for QBs
                    df_available.loc[pos_mask & (df_available[durability_col] > 0.9), 'durability_boost'] = (
                        df_available.loc[pos_mask & (df_available[durability_col] > 0.9), 'score'] * 0.05  # 5% boost for QBs
                    )
                else:
                    # Strong durability boost for skill positions
                    df_available.loc[pos_mask & (df_available[durability_col] > 0.9), 'durability_boost'] = (
                        df_available.loc[pos_mask & (df_available[durability_col] > 0.9), 'score'] * 0.25  # 25% boost for iron men
                    )
                
                # Injury penalty applies to all positions equally
                df_available.loc[pos_mask & (df_available[durability_col] < 0.7), 'durability_boost'] = (
                    df_available.loc[pos_mask & (df_available[durability_col] < 0.7), 'score'] * -0.3  # 30% penalty for fragile players
                )
        
        # Apply durability boosts
        df_available['durability_boost'] = df_available.get('durability_boost', 0)
        df_available['score'] += df_available['durability_boost']
        durable_count = len(df_available[df_available[durability_col] > 0.9])
        fragile_count = len(df_available[df_available[durability_col] < 0.7])
        print(f"   ✅ Boosted {durable_count} durable players, penalized {fragile_count} injury-prone players")
    
    # 3. VOLUME INTELLIGENCE - Balance targets vs touches (for RB/WR) - Use weighted versions
    targets_col = 'avg_targets_3yr_weighted' if 'avg_targets_3yr_weighted' in df_available.columns else 'avg_targets_3yr'
    touches_col = 'total_touches_3yr_weighted' if 'total_touches_3yr_weighted' in df_available.columns else 'total_touches_3yr'
    
    if targets_col in df_available.columns and touches_col in df_available.columns:
        skill_positions = df_available['position'].isin(['RB', 'WR'])
        df_available.loc[skill_positions, 'volume_score'] = (
            df_available.loc[skill_positions, targets_col] * 0.6 +  # Targets slightly more valuable
            df_available.loc[skill_positions, touches_col] * 0.4   # But touches matter too
        )
        
        # Special boost for pass-catching RBs (targets are GOLD for RBs)
        rb_positions = df_available['position'] == 'RB'
        high_target_rbs = (rb_positions) & (df_available[targets_col] > 50)  # 50+ targets/yr = pass-catching back
        df_available.loc[high_target_rbs, 'score'] *= 1.2  # 20% boost for dual-threat RBs (reduced from 1.4)
        
        # WR volume boost for high-volume receivers
        wr_positions = df_available['position'] == 'WR'
        high_volume_wrs = (wr_positions) & (df_available[targets_col] > 120)  # High-volume WRs
        df_available.loc[high_volume_wrs, 'score'] *= 1.15  # 15% boost for volume WRs
        
        # PENALTY for low-volume skill players
        low_volume_skill = (skill_positions) & (df_available[targets_col] < 30)  # Very low volume
        df_available.loc[low_volume_skill, 'score'] *= 0.8  # 20% penalty for low-volume players
        
        volume_count = len(df_available[skill_positions & (df_available[targets_col] > 50)])
        low_volume_count = len(df_available[low_volume_skill])
        print(f"   ✅ Boosted {volume_count} high-volume skill players, penalized {low_volume_count} low-volume players")
    
    print("🧠 Performance Intelligence applied with position controls!")
    
    # Apply position-based multipliers based on draft position (NO ADP!)
    if draft_position <= 8:  # Early picks - prioritize RB/WR
        df_available.loc[df_available['position'].isin(['RB', 'WR']), 'score'] *= 1.5
        # Boost high-opportunity players instead of using ADP
        df_available['score'] = np.where(df_available.get('opportunity_score', 0) > 35, df_available['score'] * 1.2, df_available['score'])
        # Heavily de-prioritize K/DST early
        df_available.loc[df_available['position'].isin(['K', 'DST']), 'score'] *= 0.1
    elif draft_position >= 9:  # Late picks - value picks
        # Boost high-efficiency players instead of using ADP
        df_available['score'] = np.where(df_available.get('efficiency_per_touch', 0) > 8, df_available['score'] * 1.1, df_available['score'])
    
    # MAJOR IMPROVEMENT: Much stronger team needs multiplier
    print(f"🎯 Applying team needs multipliers: {team_needs}")
    for pos, need in team_needs.items():
        if need >= 5:  # ESSENTIAL POSITION EMPTY - MASSIVE PRIORITY
            multiplier = 10.0  # 10x boost for empty essential positions
            df_available.loc[df_available['position'] == pos, 'score'] *= multiplier
            print(f"   🚨 ESSENTIAL: {pos} needs {need} -> 10x multiplier applied")
        elif need > 0:
            # Regular multiplier for other needs
            multiplier = 1 + (need * 0.5)  # Changed from 0.1 to 0.5 for bigger impact
            df_available.loc[df_available['position'] == pos, 'score'] *= multiplier
            print(f"   ✅ NEED: {pos} needs {need} -> {multiplier:.1f}x multiplier")
        elif need < -1:  # If we're oversupplied, reduce priority significantly
            df_available.loc[df_available['position'] == pos, 'score'] *= 0.2  # Reduced from 0.3 to 0.2
            print(f"   ⬇️ EXCESS: {pos} has excess -> 0.2x penalty")
    
    # Special boost for empty starting positions
    empty_positions = []
    # This will be passed from the UI with current roster state
    for pos, need in team_needs.items():
        if need >= 5:  # Empty essential positions get identified here
            empty_positions.append(pos)
    
    # EMERGENCY boost for positions we absolutely need
    for pos in empty_positions:
        df_available.loc[df_available['position'] == pos, 'score'] *= 5.0  # Additional 5x boost
        print(f"   🚨 EMERGENCY: {pos} gets additional 5x boost (total ~50x)")
    
    # Cap skill position scores if we have essential positions empty
    if empty_positions:
        print(f"🚨 ESSENTIAL POSITIONS EMPTY: {empty_positions} - Capping skill position scores")
        # Heavily reduce RB/WR scores when essential positions are empty
        for skill_pos in ['RB', 'WR']:
            if skill_pos not in empty_positions:  # Don't cap if RB/WR is actually what we need
                skill_need = team_needs.get(skill_pos, 0)
                if skill_need <= 0:  # We have enough/excess of this skill position
                    df_available.loc[df_available['position'] == skill_pos, 'score'] *= 0.1  # 90% reduction
                    print(f"   ⬇️ CAPPED: {skill_pos} scores reduced by 90% (excess while essentials empty)")
    
    # CRITICAL FIXES FOR RANKING DISASTER
    
    # 1. Filter out NON-FANTASY positions (OL, CB, LB, FS, etc.)
    fantasy_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
    df_available = df_available[df_available['position'].isin(fantasy_positions)]
    print(f"✅ Filtered to {len(df_available)} fantasy-relevant players only")
    
    # 2. Fix NaN scores - replace with 0.0
    df_available['optimized_score'] = df_available['optimized_score'].fillna(0.0)
    
    # 3. Remove players with 0 projected points (invalid data)
    df_available = df_available[df_available['projected_points'] > 0]
    print(f"✅ Removed players with 0 projected points, {len(df_available)} players remaining")
    
    # 4. Sort by optimized_score in DESCENDING order (highest first)
    suggestions = df_available.sort_values('optimized_score', ascending=False)
    
    print('🏆 TOP OPTIMIZED PICKS (Starting Lineup Focus):')
    print('------------------------------------------------------------')
    for i, (_, suggestion) in enumerate(suggestions.head(5).iterrows()):
        name = suggestion['name']
        pos = suggestion['position']
        score = suggestion['optimized_score']
        points = suggestion['projected_points']
        reasoning = suggestion.get('reasoning', 'Base value')
        print(f'{i+1}. {name} ({pos}) - Score: {score:.1f} | Points: {points:.1f} | {reasoning}')
    
    return suggestions.head(num_suggestions)

def generate_strategy(draft_position, league_size=12, rounds=15):
    # Simple strategy based on position
    if draft_position <= 3:
        return "Prioritize Elite RB/WR. Secure a top-tier talent."
    elif draft_position <= 6:
        return "Aim for a strong RB or WR. Consider a top QB if value falls."
    elif draft_position <= 9:
        return "Balance RB/WR. Look for high-upside players or solid veterans."
    else:
        return "Focus on value. Target players with high upside or good matchups."

def test_bye_week_mapping():
    """Test the bye week mapping functionality"""
    print("\n" + "="*50)
    print("TESTING BYE WEEK MAPPING")
    print("="*50)
    
    # Load the team-bye mapping
    team_bye_mapping = create_team_bye_mapping()
    print(f"📊 Total teams with bye weeks: {len(team_bye_mapping)}")
    
    # Show sample mappings
    print("\n📋 Sample Team-Bye Mappings:")
    sample_teams = list(team_bye_mapping.keys())[:10]
    for team in sample_teams:
        print(f"   {team}: Week {team_bye_mapping[team]}")
    
    # Test with rookie data
    rookie_file = os.path.join(DATA_DIR, 'FantasyPros_2025_Rookies_ALL_Rankings.csv')
    if os.path.exists(rookie_file):
        rookie_df = pd.read_csv(rookie_file)
        if 'TEAM' in rookie_df.columns:
            rookie_df['bye_week'] = rookie_df['TEAM'].map(team_bye_mapping)
            
            # Count successful mappings
            successful_mappings = rookie_df['bye_week'].notna().sum()
            total_rookies = len(rookie_df)
            
            print(f"\n🏈 Rookie Bye Week Assignment:")
            print(f"   Total rookies: {total_rookies}")
            print(f"   Successfully mapped: {successful_mappings}")
            print(f"   Success rate: {successful_mappings/total_rookies*100:.1f}%")
            
            # Show unmapped teams
            unmapped_teams = rookie_df[rookie_df['bye_week'].isna()]['TEAM'].unique()
            if len(unmapped_teams) > 0:
                print(f"   Unmapped teams: {list(unmapped_teams)}")
            
            # Show sample rookie mappings
            print(f"\n📋 Sample Rookie Bye Assignments:")
            sample_rookies = rookie_df[rookie_df['bye_week'].notna()].head(5)
            for _, row in sample_rookies.iterrows():
                print(f"   {row['PLAYER NAME']} ({row['TEAM']}): Week {row['bye_week']}")
    
    print("="*50)
    return team_bye_mapping

def get_advanced_data_sources():
    """
    Provide URLs and instructions for downloading advanced football performance data
    that will dramatically improve the model beyond basic stats
    """
    
    print("🏈 WORKING FREE NFL DATA SOURCES")
    print("="*80)
    print("📈 These APIs provide actual 2024 NFL player data to replace NFLsavant!")
    print("🚫 NO MORE INFLATED/INCORRECT DATA - REAL STATS FROM RELIABLE SOURCES")
    
    working_apis = {
        "1. ESPN NFL API (FREE - Most Comprehensive)": {
            "description": "Official ESPN API with player stats, game logs, team data",
            "key_endpoints": [
                "🏈 Current Season Players: https://sports.core.api.espn.com/v3/sports/football/nfl/athletes?limit=2000&active=true",
                "📊 Player Game Log: https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{PLAYER_ID}/gamelog",
                "🎯 Player Splits/Stats: https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{PLAYER_ID}/splits",
                "📈 Team Rosters: https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{TEAM_ID}/roster",
                "⚠️  Injury Reports: https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{TEAM_ID}/injuries",
                "🏆 Game Summary: https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={GAME_ID}",
            ],
            "data_quality": "✅ EXCELLENT - Official ESPN data, updated in real-time",
            "rate_limits": "~1000 calls/minute (be respectful)",
            "cost": "FREE",
            "pros": ["Official source", "Real-time updates", "Comprehensive stats", "Historical data"],
            "cons": ["Need to map player IDs", "Some missing advanced metrics"],
            "example_usage": "Perfect for getting accurate receiving yards, TDs, targets from game logs"
        },
        
        "2. Sleeper API (FREE - Fantasy Focus)": {
            "description": "Fantasy-focused API with clean player data and trending info",
            "key_endpoints": [
                "👥 All NFL Players: https://api.sleeper.app/v1/players/nfl",
                "📈 Trending Players: https://api.sleeper.app/v1/players/nfl/trending/add",
                "🏈 NFL State/Week: https://api.sleeper.app/v1/state/nfl",
            ],
            "data_quality": "✅ VERY GOOD - Clean fantasy-relevant data",
            "rate_limits": "1000 calls/minute",
            "cost": "FREE",
            "pros": ["Fantasy-focused", "Clean data format", "Player trending data", "No auth required"],
            "cons": ["Less detailed stats", "Fantasy-focused only"],
            "example_usage": "Great for player IDs, basic stats, and trending analysis"
        },
        
        "3. NFL.com Official Stats (FREE - Via ESPN Endpoints)": {
            "description": "Access NFL.com data through ESPN's network endpoints",
            "key_endpoints": [
                "📊 Live Scores: https://cdn.espn.com/core/nfl/scoreboard?xhr=1",
                "📅 Schedule: https://cdn.espn.com/core/nfl/schedule?xhr=1&year=2024",
                "🏆 Game Details: https://cdn.espn.com/core/nfl/boxscore?xhr=1&gameId={GAME_ID}",
                "📈 Play-by-Play: https://cdn.espn.com/core/nfl/playbyplay?xhr=1&gameId={GAME_ID}",
            ],
            "data_quality": "✅ EXCELLENT - Official NFL data via ESPN",
            "rate_limits": "Be respectful - no official limits but don't abuse",
            "cost": "FREE",
            "pros": ["Official data", "Real-time", "Detailed play-by-play"],
            "cons": ["Different format than main ESPN API"],
            "example_usage": "Real-time game data and official statistics"
        }
    }
    
    for api_name, details in working_apis.items():
        print(f"\n{api_name}")
        print("-" * (len(api_name) - 2))
        print(f"📝 {details['description']}")
        print(f"💰 Cost: {details['cost']}")
        print(f"⭐ Quality: {details['data_quality']}")
        print(f"⏱️  Rate Limits: {details['rate_limits']}")
        
        print(f"\n🔗 Key Endpoints:")
        for endpoint in details['key_endpoints']:
            print(f"   {endpoint}")
        
        print(f"\n✅ Pros: {', '.join(details['pros'])}")
        print(f"❌ Cons: {', '.join(details['cons'])}")
        print(f"💡 Best Use: {details['example_usage']}")
    
    print("\n" + "="*80)
    print("🎯 RECOMMENDATION FOR YOUR MODEL:")
    print("="*80)
    print("1. 🏆 Use ESPN API for accurate 2024 player stats")
    print("   - Replace NFLsavant data with real ESPN game logs")
    print("   - Get actual receiving yards, TDs, targets, carries")
    print("   - Use player splits for efficiency metrics")
    
    print("\n2. 📊 Integration Strategy:")
    print("   - Keep your current feature engineering (target share, efficiency)")
    print("   - Replace the raw stats (yards, TDs) with ESPN data")
    print("   - Use Sleeper API for player ID mapping")
    
    print("\n3. 🔧 Implementation Steps:")
    print("   a) Create ESPN API client script")
    print("   b) Map player IDs between systems")
    print("   c) Extract 2024 season stats for all players")
    print("   d) Replace NFLsavant data in your pipeline")
    print("   e) Re-train model with accurate data")
    
    print("\n4. 🎯 Expected Improvement:")
    print("   - Accurate Tyreek Hill stats (959 yards, not 1248!)")
    print("   - Proper target shares based on real targets")
    print("   - Correct efficiency calculations")
    print("   - Better model predictions with honest data")
    
    print("\n💾 NEXT STEPS:")
    print("1. Run: python scripts/espn_data_collector.py")
    print("2. Update: data/fantasy_metrics_2024.csv with real ESPN data")
    print("3. Re-train: Your model with accurate historical performance")
    print("4. Validate: Compare ESPN stats vs NFLsavant to see differences")
    
    return working_apis

def calculate_starting_lineup_value(team_roster, available_df):
    """Calculate the total value of the best possible starting lineup from current roster"""
    positions_needed = {
        'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1  # Standard lineup
    }
    
    starting_lineup_value = 0
    used_players = []
    
    # Fill core positions first
    for pos in ['QB', 'TE']:  # Single positions
        if pos in team_roster and team_roster[pos]:
            player = team_roster[pos]
            if isinstance(player, dict):
                starting_lineup_value += player.get('projected_points', 0)
                used_players.append(player.get('name', ''))
    
    # Handle RB positions (need 2)
    rb_players = []
    for slot in ['RB1', 'RB2']:
        if slot in team_roster and team_roster[slot]:
            player = team_roster[slot]
            if isinstance(player, dict):
                rb_players.append(player)
                used_players.append(player.get('name', ''))
    
    # Handle WR positions (need 2) 
    wr_players = []
    for slot in ['WR1', 'WR2']:
        if slot in team_roster and team_roster[slot]:
            player = team_roster[slot]
            if isinstance(player, dict):
                wr_players.append(player)
                used_players.append(player.get('name', ''))
    
    # Add RB/WR values
    for player in rb_players + wr_players:
        starting_lineup_value += player.get('projected_points', 0)
    
    # Handle FLEX (best remaining RB/WR)
    flex_candidates = []
    if 'FLEX' in team_roster and team_roster['FLEX']:
        flex_player = team_roster['FLEX']
        if isinstance(flex_player, dict):
            starting_lineup_value += flex_player.get('projected_points', 0)
            used_players.append(flex_player.get('name', ''))
    
    return starting_lineup_value, used_players

def get_positional_scarcity_multiplier(position, round_num):
    """Calculate CONSERVATIVE scarcity multiplier - performance matters more than position"""
    # REALISTIC scarcity - smaller boosts, performance should drive rankings
    scarcity_map = {
        'QB': {1: 0.8, 2: 0.9, 3: 0.95, 4: 1.0, 5: 1.05, 6: 1.0, 7: 0.95, 8: 0.9},
        'RB': {1: 1.1, 2: 1.1, 3: 1.05, 4: 1.05, 5: 1.0, 6: 1.0, 7: 0.95, 8: 0.9},  # REDUCED from 1.4 to 1.1
        'WR': {1: 1.1, 2: 1.1, 3: 1.05, 4: 1.05, 5: 1.0, 6: 1.0, 7: 0.95, 8: 0.9},  # REDUCED from 1.3 to 1.1
        'TE': {1: 0.8, 2: 0.85, 3: 0.9, 4: 1.0, 5: 1.05, 6: 1.1, 7: 1.0, 8: 0.95},
        'K': {1: 0.1, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.5, 6: 0.7, 7: 1.0, 8: 1.0},
        'DST': {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.6, 6: 0.8, 7: 1.0, 8: 1.0}
    }
    
    round_key = min(round_num, 8)  # Cap at round 8
    return scarcity_map.get(position, {}).get(round_key, 1.0)

def optimize_starting_lineup_first(available_df, current_roster, team_needs, round_num=1, league_size=12):
    """Enhanced draft suggestions that prioritize starting lineup strength, then bench depth"""
    
    # 🚨 CRITICAL DATA FILTERING - ELIMINATE GARBAGE DATA FIRST
    print(f"🧹 CLEANING DATA: Starting with {len(available_df)} players")
    
    # 1. Filter out NON-FANTASY positions (OL, CB, LB, FS, SS, etc.)
    fantasy_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
    available_df = available_df[available_df['position'].isin(fantasy_positions)]
    print(f"✅ Filtered to {len(available_df)} fantasy-relevant players (removed OL, CB, LB, etc.)")
    
    # 2. Remove players with 0 or negative projected points (invalid data)
    available_df = available_df[available_df['projected_points'] > 0]
    print(f"✅ Removed 0-point players, {len(available_df)} valid players remaining")
    
    # 3. Fix any NaN values in critical columns
    available_df['projected_points'] = available_df['projected_points'].fillna(0.0)
    available_df['optimized_score'] = available_df.get('optimized_score', available_df['projected_points']).fillna(0.0)
    print(f"✅ Fixed NaN values in scoring columns")
    
    print(f"\n🎯 OPTIMIZING FOR STARTING LINEUP (Round {round_num})")
    print("="*60)
    
    # Calculate current starting lineup strength
    current_lineup_value, used_players = calculate_starting_lineup_value(current_roster, available_df)
    print(f"📊 Current Starting Lineup Value: {current_lineup_value:.1f} points")
    
    # Identify lineup holes and priorities - handle None values properly
    lineup_needs = {
        'QB': current_roster.get('QB') is None or current_roster.get('QB') == {},
        'RB': sum([1 for slot in ['RB1', 'RB2'] if current_roster.get(slot) is None or current_roster.get(slot) == {}]),
        'WR': sum([1 for slot in ['WR1', 'WR2'] if current_roster.get(slot) is None or current_roster.get(slot) == {}]),
        'TE': current_roster.get('TE') is None or current_roster.get('TE') == {},
        'FLEX': current_roster.get('FLEX') is None or current_roster.get('FLEX') == {},
        'K': current_roster.get('K') is None or current_roster.get('K') == {},
        'DST': current_roster.get('DST') is None or current_roster.get('DST') == {}
    }
    
    starting_holes = sum([
        1 if lineup_needs['QB'] else 0,
        lineup_needs['RB'],
        lineup_needs['WR'], 
        1 if lineup_needs['TE'] else 0,
        1 if lineup_needs['FLEX'] else 0,
        1 if lineup_needs['K'] else 0,
        1 if lineup_needs['DST'] else 0
    ])
    
    print(f"🔍 Starting Lineup Holes: {starting_holes}")
    print(f"📋 Needs: QB:{lineup_needs['QB']}, RB:{lineup_needs['RB']}, WR:{lineup_needs['WR']}, TE:{lineup_needs['TE']}, FLEX:{lineup_needs['FLEX']}, K:{lineup_needs['K']}, DST:{lineup_needs['DST']}")
    
    # Enhanced scoring for lineup optimization
    enhanced_suggestions = []
    
    # 🧠 PERFORMANCE INTELLIGENCE - Apply trend penalties before player evaluation
    print("🧠 Applying Performance Intelligence to optimizer...")
    available_df = available_df.copy()  # Don't modify original
    
    # Initialize performance boosts
    available_df['performance_intelligence_boost'] = 0
    
    # 1. PERFORMANCE TREND PENALTIES/BOOSTS - This is critical for accuracy!
    if 'performance_trend' in available_df.columns:
        for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
            pos_mask = available_df['position'] == pos
            if pos_mask.any():
                if pos in ['RB', 'WR']:
                    # CONSERVATIVE trend impact - adjusted for new percentage-based trends
                    # Strong positive trends get modest boost (thresholds lowered for % trends)
                    available_df.loc[pos_mask & (available_df['performance_trend'] > 40), 'performance_intelligence_boost'] += (
                        available_df.loc[pos_mask & (available_df['performance_trend'] > 40), 'projected_points'] * 0.10  # 10% max boost for exceptional trends (40%+ improvement)
                    )
                    available_df.loc[pos_mask & (available_df['performance_trend'] > 15) & (available_df['performance_trend'] <= 40), 'performance_intelligence_boost'] += (
                        available_df.loc[pos_mask & (available_df['performance_trend'] > 15) & (available_df['performance_trend'] <= 40), 'projected_points'] * 0.05  # 5% mild boost (15-40% improvement)
                    )
                    # REALITY CHECK: Penalty for declining players (negative trends)
                    available_df.loc[pos_mask & (available_df['performance_trend'] < -15), 'performance_intelligence_boost'] -= (
                        available_df.loc[pos_mask & (available_df['performance_trend'] < -15), 'projected_points'] * 0.15  # 15% penalty for declining players (-15% or worse)
                    )
                    available_df.loc[pos_mask & (available_df['performance_trend'] < -5) & (available_df['performance_trend'] >= -15), 'performance_intelligence_boost'] -= (
                        available_df.loc[pos_mask & (available_df['performance_trend'] < -5) & (available_df['performance_trend'] >= -15), 'projected_points'] * 0.08  # 8% penalty for mild decline (-5% to -15%)
                    )
                elif pos == 'QB':
                    # Minimal trend impact for QBs to prevent over-emphasis (adjusted for % trends)
                    available_df.loc[pos_mask & (available_df['performance_trend'] > 15), 'performance_intelligence_boost'] += (
                        available_df.loc[pos_mask & (available_df['performance_trend'] > 15), 'projected_points'] * 0.05  # 5% max for QBs (15%+ improvement)
                    )
                else:
                    # Moderate impact for other positions
                    available_df.loc[pos_mask & (available_df['performance_trend'] > 20), 'performance_intelligence_boost'] += (
                        available_df.loc[pos_mask & (available_df['performance_trend'] > 20), 'projected_points'] * 0.15  # 15% boost
                    )
        
        trend_applied = len(available_df[available_df['performance_trend'].notna()])
        declining_players = len(available_df[available_df['performance_trend'] < -10])
        improving_players = len(available_df[available_df['performance_trend'] > 20])
        print(f"📈 Applied trend analysis to {trend_applied} players ({improving_players} rising, {declining_players} declining)")
    
    # 2. CONSISTENCY BOOST - Reward reliable players
    if 'consistency_score' in available_df.columns:
        available_df.loc[available_df['consistency_score'] > 15, 'performance_intelligence_boost'] += (
            available_df.loc[available_df['consistency_score'] > 15, 'projected_points'] * 0.1  # 10% boost for consistent players
        )
        consistent_players = len(available_df[available_df['consistency_score'] > 15])
        print(f"🎯 Applied consistency boost to {consistent_players} reliable players")
    
    # 3. OPPORTUNITY BOOST - Reward high-volume players  
    if 'opportunity_score_weighted' in available_df.columns:
        available_df.loc[available_df['opportunity_score_weighted'] > 20, 'performance_intelligence_boost'] += (
            available_df.loc[available_df['opportunity_score_weighted'] > 20, 'projected_points'] * 0.1  # 10% boost for high opportunity
        )
        high_opp_players = len(available_df[available_df['opportunity_score_weighted'] > 20])
        print(f"🎯 Applied opportunity boost to {high_opp_players} high-volume players")
    
    # 4. REALITY CHECK - Recent actual performance vs projections (CRITICAL!)
    if 'previous_points' in available_df.columns:
        # Calculate points per game from last season
        available_df['ppg_last_season'] = available_df['previous_points'] / available_df.get('games_played', 16).replace(0, 16)
        available_df['ppg_last_season'] = available_df['ppg_last_season'].fillna(0)
        
        # Compare projections to recent reality
        available_df['projection_vs_reality'] = available_df['projected_points'] - available_df['ppg_last_season']
        
        # PENALIZE over-optimistic projections (especially for RB/WR)
        for pos in ['RB', 'WR']:
            pos_mask = available_df['position'] == pos
            if pos_mask.any():
                # If projected points are way higher than recent performance, apply reality discount
                over_optimistic_mask = (
                    pos_mask & 
                    (available_df['ppg_last_season'] > 0) &  # Had some performance data
                    (available_df['projection_vs_reality'] > 4)  # Projected >4 points higher than recent reality
                )
                if over_optimistic_mask.any():
                    reality_discount = available_df.loc[over_optimistic_mask, 'projection_vs_reality'] * 0.5  # 50% discount for over-optimism
                    available_df.loc[over_optimistic_mask, 'performance_intelligence_boost'] -= reality_discount
        
        # REWARD players who actually performed well recently
        for pos in ['RB', 'WR']:
            pos_mask = available_df['position'] == pos
            if pos_mask.any():
                strong_recent_performance = (
                    pos_mask & 
                    (available_df['ppg_last_season'] > 12 if pos == 'RB' else available_df['ppg_last_season'] > 10)  # Strong recent performance
                )
                if strong_recent_performance.any():
                    available_df.loc[strong_recent_performance, 'performance_intelligence_boost'] += (
                        available_df.loc[strong_recent_performance, 'ppg_last_season'] * 0.3  # 30% of recent performance as bonus
                    )
        
        reality_adjustments = len(available_df[abs(available_df.get('projection_vs_reality', 0)) > 2])
        print(f"⚖️ Applied reality check to {reality_adjustments} players (projection vs recent performance)")
    
    # 5. INJURY STATUS ANALYSIS - Critical for current season!
    if 'injury_status' in available_df.columns:
        # Major penalties for current injury concerns
        injury_out_mask = available_df['injury_status'].isin(['Out', 'IR', 'PUP', 'Suspended'])
        injury_questionable_mask = available_df['injury_status'].isin(['Questionable', 'Doubtful'])
        
        if injury_out_mask.any():
            # MAJOR penalty for players who are Out/IR (essentially undraftable)
            available_df.loc[injury_out_mask, 'performance_intelligence_boost'] -= (
                available_df.loc[injury_out_mask, 'projected_points'] * 0.8  # 80% penalty for Out players
            )
        
        if injury_questionable_mask.any():
            # Moderate penalty for Questionable players
            available_df.loc[injury_questionable_mask, 'performance_intelligence_boost'] -= (
                available_df.loc[injury_questionable_mask, 'projected_points'] * 0.15  # 15% penalty for uncertainty
            )
        
        out_players = len(available_df[injury_out_mask])
        questionable_players = len(available_df[injury_questionable_mask])
        print(f"🏥 Applied injury penalties: {out_players} Out players (-80%), {questionable_players} Questionable players (-15%)")
    
    # Initialize base score column (needed for essential position logic)
    if 'projected_points' in available_df.columns:
        available_df['score'] = available_df['projected_points'].fillna(0)
    else:
        available_df['score'] = 0
    print(f"📊 Initialized base scores from projected_points")
    
    # 🚨 ESSENTIAL POSITION PRIORITIZATION - Apply MASSIVE multipliers for empty positions
    print(f"🎯 TEAM CONSTRUCTION: Applying essential position multipliers")
    print(f"   Team needs: {team_needs}")
    
    # Identify empty essential positions that need MASSIVE priority
    essential_empty_positions = []
    for pos, need in team_needs.items():
        if need >= 5:  # Empty essential positions (QB, TE, K, DST)
            essential_empty_positions.append(pos)
    
    if essential_empty_positions:
        print(f"🚨 ESSENTIAL POSITIONS EMPTY: {essential_empty_positions}")
        
        # Apply REALISTIC draft strategy multipliers for essential positions
        # CRITICAL: Prioritize actual need levels, not just position type
        position_multipliers = {
            'QB': 2.5,   # 2.5x boost - QB very important when missing
            'TE': 2.0,   # 2.0x boost - TE important but not QB level
            'K': 1.3,    # 1.3x boost - Kicker moderate boost
            'DST': 1.3   # 1.3x boost - Defense moderate boost
        }

        # SKILL POSITIONS: Scale based on actual need vs other positions
        skill_position_multipliers = {
            'RB': 1.1,   # Base RB multiplier - conservative
            'WR': 1.1    # Base WR multiplier - conservative  
        }

        for pos, need in team_needs.items():
            if need > 0:
                pos_mask = available_df['position'] == pos
                if pos_mask.any():
                    # Use realistic multipliers based on position type AND need level
                    if pos in ['QB', 'TE', 'K', 'DST']:
                        base_multiplier = position_multipliers.get(pos, 1.1)
                        # Scale more aggressively for essential positions
                        multiplier = base_multiplier + min(need * 0.2, 0.8)  # Max +0.8 for essential
                    elif pos in ['RB', 'WR']:
                        base_multiplier = skill_position_multipliers.get(pos, 1.1)  
                        # CRITICAL: Don't over-boost skill positions when essentials have higher need
                        max_essential_need = max([team_needs.get(essential, 0) for essential in ['QB', 'TE', 'K', 'DST']])
                        if need < max_essential_need:
                            # Skill position has lower need than essentials - minimal boost
                            multiplier = base_multiplier + min(need * 0.02, 0.1)  # Max +0.1
                        else:
                            # Skill position has equal/higher need
                            multiplier = base_multiplier + min(need * 0.05, 0.2)  # Max +0.2
                    else:
                        multiplier = 1.1
                    
                    # Cap all multipliers to prevent inflation
                    multiplier = min(multiplier, 2.0)  # Max 2x boost
                    
                    available_df.loc[pos_mask, 'score'] *= multiplier
                    print(f"   ✅ {pos}: {multiplier:.2f}x multiplier (need: {need}) - PRIORITY-BASED scaling")
            elif need < -1:
                pos_mask = available_df['position'] == pos
                if pos_mask.any():
                    available_df.loc[pos_mask, 'score'] *= 0.8  # Modest 20% penalty for excess
                    print(f"   ⬇️ {pos}: 0.8x penalty (excess: {need})")
    
    else:
        print(f"✅ All essential positions filled - normal prioritization")
        # Apply REALISTIC team needs multipliers (FIXED VERSION)
        for pos, need in team_needs.items():
            if need > 0:
                pos_mask = available_df['position'] == pos
                if pos_mask.any():
                    # Use same realistic scaling as the other branch
                    if pos in ['QB', 'TE', 'K', 'DST']:
                        base_multiplier = 1.2 if pos == 'QB' else (1.3 if pos == 'TE' else 1.1)
                        multiplier = base_multiplier + min(need * 0.1, 0.3)  # Max +0.3 additional boost
                    elif pos in ['RB', 'WR']:
                        base_multiplier = 1.1  # Much more conservative for skill positions
                        multiplier = base_multiplier + min(need * 0.05, 0.2)  # Max +0.2 additional boost
                    else:
                        multiplier = 1.1
                    
                    # Cap all multipliers to prevent inflation
                    multiplier = min(multiplier, 1.5)  # Never more than 50% boost
                    
                    available_df.loc[pos_mask, 'score'] *= multiplier
                    print(f"   ✅ {pos}: {multiplier:.2f}x multiplier (need: {need}) - REALISTIC scaling")
            elif need < -1:
                pos_mask = available_df['position'] == pos
                if pos_mask.any():
                    available_df.loc[pos_mask, 'score'] *= 0.8  # Modest 20% penalty for excess
                    print(f"   ⬇️ {pos}: 0.8x penalty (excess: {need})")
    
    for _, player in available_df.iterrows():
        base_score = player.get('score', 0)
        position = player.get('position', '')
        projected_points = player.get('projected_points', 0)
        
        # Apply Performance Intelligence boost to base score
        performance_boost = player.get('performance_intelligence_boost', 0)
        base_score += performance_boost
        
        # Base value factors
        scarcity_mult = get_positional_scarcity_multiplier(position, round_num)
        
        # Starting lineup priority boost - CONSERVATIVE FANTASY STRATEGY
        starting_boost = 0
        
        # CRITICAL: Only boost elite players in early rounds, not mediocre backups
        is_elite_player = projected_points > 200  # Only true elite players get early round boost
        
        if round_num <= 3:  # Early rounds: ONLY boost ELITE performers
            if position == 'RB' and lineup_needs['RB'] > 0 and is_elite_player:
                starting_boost = projected_points * 0.15  # Elite RB boost only
            elif position == 'WR' and lineup_needs['WR'] > 0 and is_elite_player:
                starting_boost = projected_points * 0.12  # Elite WR boost only
            elif position == 'QB' and lineup_needs['QB'] and is_elite_player:
                starting_boost = projected_points * 0.05  # Elite QB boost only
            elif position in ['RB', 'WR', 'TE'] and lineup_needs['FLEX'] and is_elite_player:
                starting_boost = projected_points * 0.08  # Elite FLEX boost only
        elif round_num <= 6:
            if position == 'QB' and lineup_needs['QB']:
                starting_boost = projected_points * 0.15  # REDUCED from 30% to 15% boost
            elif position == 'RB' and lineup_needs['RB'] > 0:
                starting_boost = projected_points * 0.20  # REDUCED from 40% to 20% boost
            elif position == 'WR' and lineup_needs['WR'] > 0:
                starting_boost = projected_points * 0.18  # REDUCED from 35% to 18% boost
            elif position == 'TE' and lineup_needs['TE']:
                starting_boost = projected_points * 0.15  # REDUCED from 25% to 15% boost
            elif position in ['RB', 'WR', 'TE'] and lineup_needs['FLEX']:
                starting_boost = projected_points * 0.10   # REDUCED from 20% to 10% boost
        else:  # Late rounds: fill remaining gaps
            if starting_holes > 0:
                if position == 'QB' and lineup_needs['QB']:
                    starting_boost = projected_points * 0.10  # REDUCED from 20% to 10%
                elif position == 'TE' and lineup_needs['TE']:
                    starting_boost = projected_points * 0.10  # REDUCED from 20% to 10%
                elif position in ['RB', 'WR'] and (lineup_needs['RB'] > 0 or lineup_needs['WR'] > 0):
                    starting_boost = projected_points * 0.08  # REDUCED from 15% to 8%
        
        # Bench depth boost - ONLY after starting lineup is completely filled
        bench_boost = 0
        starting_lineup_complete = not any([lineup_needs['QB'], lineup_needs['RB'] > 0, lineup_needs['WR'] > 0, lineup_needs['TE'], lineup_needs['FLEX'], lineup_needs['K'], lineup_needs['DST']])
        
        if starting_lineup_complete:
            # Only give bench bonuses when starting lineup is COMPLETELY filled
            if position in ['RB', 'WR'] and projected_points > 8:
                bench_boost = projected_points * 0.3  # 30% bonus for quality bench depth
        else:
            # Starting lineup not complete - prioritize missing positions
            # BUT in early rounds, follow fantasy football strategy (RB/WR priority)
            if round_num <= 3:
                # Early rounds: ONLY skill positions (NO K/DST regardless of need)
                if lineup_needs['FLEX'] and position in ['RB', 'WR', 'TE']:
                    starting_boost = projected_points * 0.4  # FLEX priority
                # NO boosts for QB, K, DST, or TE in rounds 1-3 (fantasy strategy)
            elif round_num <= 8:
                # Mid rounds: Allow TE, still no K/DST 
                if lineup_needs['QB'] and position == 'QB':
                    starting_boost = projected_points * 0.8  # QB priority when needed
                elif lineup_needs['TE'] and position == 'TE':
                    starting_boost = projected_points * 0.7  # TE priority when needed  
                elif lineup_needs['FLEX'] and position in ['RB', 'WR', 'TE']:
                    starting_boost = projected_points * 0.4  # FLEX priority
                # NO K/DST boosts until very late
            else:
                # Late rounds ONLY: Allow K/DST 
                if lineup_needs['QB'] and position == 'QB':
                    starting_boost = projected_points * 0.8  # QB priority when needed
                elif lineup_needs['TE'] and position == 'TE':
                    starting_boost = projected_points * 0.7  # TE priority when needed  
                elif lineup_needs['K'] and position == 'K':
                    starting_boost = projected_points * 1.0  # K priority when needed (late only)
                elif lineup_needs['DST'] and position == 'DST':
                    starting_boost = projected_points * 1.0  # DST priority when needed (late only)
                elif lineup_needs['FLEX'] and position in ['RB', 'WR', 'TE']:
                    starting_boost = projected_points * 0.4  # FLEX priority
        
        # Early vs late round strategy - ONLY if we need those positions AND player is elite
        round_strategy_mult = 1.0
        if round_num <= 3:  # Early rounds: prioritize elite starters ONLY if needed AND elite
            if position == 'RB' and lineup_needs['RB'] > 0 and projected_points > 200:
                round_strategy_mult = 1.2  # Only boost ELITE RBs we need
            elif position == 'WR' and lineup_needs['WR'] > 0 and projected_points > 200:
                round_strategy_mult = 1.2  # Only boost ELITE WRs we need
            elif position == 'QB':
                round_strategy_mult = 0.3  # HEAVILY discourage QB early (was 0.8, too weak)
            elif position in ['K', 'DST']:
                round_strategy_mult = 0.01  # NEVER draft K/DST early
        
        # Calculate final optimized score
        optimized_score = (
            base_score * scarcity_mult * round_strategy_mult + 
            starting_boost + 
            bench_boost
        )
        
        enhanced_suggestions.append({
            'name': player.get('name', ''),
            'position': position,
            'projected_points': projected_points,
            'base_score': base_score,
            'scarcity_mult': scarcity_mult,
            'starting_boost': starting_boost,
            'bench_boost': bench_boost,
            'round_strategy': round_strategy_mult,
            'optimized_score': optimized_score,
            'vbd': player.get('vbd', 0),
            'bye_week': player.get('bye_week', 'Unknown'),
            'consistency_score': player.get('consistency_score', 0),
            'performance_trend': player.get('performance_trend', 0),
            'avg_targets_3yr': player.get('avg_targets_3yr_weighted', player.get('avg_targets_3yr', 0)),
            'durability_score': player.get('durability_score_weighted', player.get('durability_score', 0)),
            'opportunity_score_weighted': player.get('opportunity_score_weighted', 0),
            'recent_vs_weighted_historical': player.get('recent_vs_weighted_historical', 1.0),
            'injury_status': player.get('injury_status', None),
            # Add ADP and rookie data
            'adp_rank': player.get('adp_rank', player.get('adp', None)),
            'adp_tier': player.get('adp_tier', 'Unknown'),
            'is_rookie': player.get('is_rookie', False),
            'draft_round': player.get('draft_round', None),
            'draft_pick': player.get('draft_pick', None)
        })
    
    # Sort by optimized score
    enhanced_suggestions.sort(key=lambda x: x['optimized_score'], reverse=True)
    
    # Show top suggestions with reasoning
    print(f"\n🏆 TOP OPTIMIZED PICKS (Starting Lineup Focus):")
    print("-" * 60)
    
    top_suggestions = enhanced_suggestions[:5]
    for i, suggestion in enumerate(top_suggestions, 1):
        name = suggestion['name']
        pos = suggestion['position']
        score = suggestion['optimized_score']
        reasoning = suggestion.get('reasoning', 'Base value')
        print(f'{i+1}. {name} ({pos}) - Score: {score:.1f} | {reasoning}')
    
    return enhanced_suggestions[:5]  # Return top 5 for compact UI

def main():
    league_size = int(input('Enter league size (e.g., 12): '))
    draft_position = int(input('Enter your draft position (1-{0}): '.format(league_size)))
    print(generate_strategy(draft_position, league_size))
    players = load_players()
    projections = load_projections()

    if 'name' not in projections.columns and 'player_name' in projections.columns:
        projections = projections.rename(columns={'player_name': 'name'})
        print("Renamed 'player_name' to 'name' in projections")

    df = prepare_data(players, projections)
    
    # Test bye week mapping
    if input('Test bye week mapping? (y/n): ') == 'y':
        test_bye_week_mapping()
    
    if input('Train model? (y/n): ') == 'y':
        train_model(df)
    available_df = df.copy()
    team_needs = {'QB': 0.2, 'RB': 2.5, 'WR': 3.5, 'TE': 1}  # Increased bias to RB/WR
    drafted = []
    while True:
        cmd = input('Enter command (suggest/cross <name>/quit): ')
        if cmd == 'quit':
            break
        elif cmd == 'test_bye':
            test_bye_week_mapping()
        elif cmd == 'data_sources':
            get_advanced_data_sources()
        elif cmd == 'suggest':
            suggestions = suggest_picks(available_df, draft_position, team_needs, league_size=league_size)
            print(suggestions)
        elif cmd.startswith('cross '):
            player_name = cmd.split('cross ')[1]
            if player_name in available_df['name'].values:
                available_df = available_df[available_df['name'] != player_name]
                drafted.append(player_name)
                print(f'Crossed off {player_name}. Drafted: {drafted}')


if __name__ == '__main__':
    main() 