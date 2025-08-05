#!/usr/bin/env python3
"""
Integrate Sleeper Real Data Into Fantasy System
Replace synthetic data with REAL NFL players from Sleeper API

REAL NFL STATISTICS ONLY
"""

import json
import pandas as pd
import shutil
from datetime import datetime
from pathlib import Path

def backup_existing_data():
    """Backup existing player data"""
    backup_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    files_to_backup = [
        'data/players.json',
        'data/fantasy_metrics_2024.csv'
    ]
    
    print("💾 BACKING UP EXISTING DATA...")
    for file_path in files_to_backup:
        if Path(file_path).exists():
            backup_path = f"{file_path}.backup_{backup_time}"
            shutil.copy2(file_path, backup_path)
            print(f"   ✅ Backed up {file_path} → {backup_path}")
    print()

def load_sleeper_data():
    """Load the collected Sleeper data"""
    print("📥 LOADING SLEEPER DATA...")
    
    try:
        with open('sleeper_nfl_data.json', 'r') as f:
            data = json.load(f)
        
        print(f"   ✅ Loaded {data['total_players']} real NFL players")
        print(f"   📅 Collection time: {data['collection_time']}")
        print(f"   🏈 NFL Season: {data.get('nfl_state', {}).get('season', 'Unknown')}")
        print()
        
        return data
    except Exception as e:
        print(f"   ❌ Error loading Sleeper data: {e}")
        return None

def convert_to_fantasy_metrics(sleeper_data):
    """Convert Sleeper data to fantasy metrics format"""
    print("🔄 CONVERTING TO FANTASY METRICS FORMAT...")
    
    players = sleeper_data['players']
    fantasy_metrics = []
    
    for player in players:
        # Focus on fantasy-relevant positions
        if player['position'] not in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
            continue
        
        # Create fantasy metrics entry
        fantasy_metric = {
            'name': player['name'],
            'position': player['position'],
            'team': player['team'] if player['team'] else 'FA',
            'age': player['age'],
            'experience': player['years_exp'],
            'injury_status': player['injury_status'],
            'player_id': player['player_id'],
            
            # Fantasy scoring - will be calculated based on projections
            'projected_points': player.get('projected_points', 0.0),
            'avg_points': player.get('avg_points', 0.0),
            'fantasy_points_2024': player.get('fantasy_points_2024', 0.0),
            'games_played': player.get('games_played', 0),
            
            # Player details
            'height': player['height'],
            'weight': player['weight'],
            'college': player['college'],
            'draft_position': player.get('search_rank', 999),
            
            # Data source tracking
            'data_source': 'sleeper_api_real',
            'is_real_player': True,
            'last_updated': player['last_updated'],
            'sleeper_player_id': player['player_id']
        }
        
        fantasy_metrics.append(fantasy_metric)
    
    print(f"   ✅ Created {len(fantasy_metrics)} fantasy metrics entries")
    print()
    
    return fantasy_metrics

def create_players_json(sleeper_data):
    """Create new players.json with real data"""
    print("📝 CREATING NEW PLAYERS.JSON...")
    
    players_dict = {}
    
    for player in sleeper_data['players']:
        player_key = player['player_id']
        
        # Create comprehensive player entry
        players_dict[player_key] = {
            'player_id': player['player_id'],
            'name': player['name'],
            'first_name': player['first_name'],
            'last_name': player['last_name'],
            'position': player['position'],
            'team': player['team'] if player['team'] else 'FA',
            'jersey_number': player['number'],
            'age': player['age'],
            'height': player['height'],
            'weight': player['weight'],
            'college': player['college'],
            'years_exp': player['years_exp'],
            'status': player['status'],
            'injury_status': player['injury_status'],
            'fantasy_positions': player['fantasy_positions'],
            'search_rank': player['search_rank'],
            'depth_chart_position': player['depth_chart_position'],
            'depth_chart_order': player['depth_chart_order'],
            
            # Fantasy data
            'projected_points': player.get('projected_points', 0.0),
            'avg_points': player.get('avg_points', 0.0),
            'fantasy_points_2024': player.get('fantasy_points_2024', 0.0),
            
            # Metadata
            'data_source': 'sleeper_api',
            'is_real': True,
            'last_updated': player['last_updated']
        }
    
    # Save to file
    with open('data/players.json', 'w') as f:
        json.dump(players_dict, f, indent=2)
    
    print(f"   ✅ Created data/players.json with {len(players_dict)} real players")
    print()

def create_fantasy_metrics_csv(fantasy_metrics):
    """Create new fantasy_metrics_2024.csv with real data"""
    print("📊 CREATING NEW FANTASY_METRICS_2024.CSV...")
    
    df = pd.DataFrame(fantasy_metrics)
    
    # Save to file
    csv_path = 'data/fantasy_metrics_2024.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"   ✅ Created {csv_path} with {len(df)} real players")
    
    # Show position breakdown
    position_counts = df['position'].value_counts()
    print(f"   📈 Position breakdown:")
    for pos, count in position_counts.items():
        print(f"      {pos}: {count} players")
    print()

def update_system_stats(sleeper_data, fantasy_metrics):
    """Display system update statistics"""
    print("📊 SYSTEM UPDATE STATISTICS")
    print("=" * 50)
    
    nfl_state = sleeper_data.get('nfl_state', {})
    
    print(f"🏈 NFL Season: {nfl_state.get('season', 'Unknown')}")
    print(f"📅 NFL Week: {nfl_state.get('week', 'Unknown')}")
    print(f"🎯 Season Type: {nfl_state.get('season_type', 'Unknown')}")
    print()
    
    print(f"👥 Total Real Players: {sleeper_data['total_players']}")
    print(f"🏆 Fantasy Relevant: {len(fantasy_metrics)}")
    
    # Show top players by position
    df = pd.DataFrame(fantasy_metrics)
    
    for position in ['QB', 'RB', 'WR', 'TE']:
        pos_players = df[df['position'] == position].head(3)
        if not pos_players.empty:
            print(f"\n🏈 Top {position}s:")
            for _, player in pos_players.iterrows():
                team = player['team'] if player['team'] != 'FA' else 'Free Agent'
                print(f"   • {player['name']} ({team})")

def main():
    """Main integration function"""
    print("🚀 INTEGRATING SLEEPER REAL DATA INTO FANTASY SYSTEM")
    print("=" * 60)
    print()
    
    # Step 1: Backup existing data
    backup_existing_data()
    
    # Step 2: Load Sleeper data
    sleeper_data = load_sleeper_data()
    if not sleeper_data:
        print("❌ Failed to load Sleeper data. Aborting integration.")
        return
    
    # Step 3: Convert to fantasy metrics
    fantasy_metrics = convert_to_fantasy_metrics(sleeper_data)
    
    # Step 4: Create new data files
    create_players_json(sleeper_data)
    create_fantasy_metrics_csv(fantasy_metrics)
    
    # Step 5: Show statistics
    update_system_stats(sleeper_data, fantasy_metrics)
    
    print("\n🎉 INTEGRATION COMPLETE!")
    print("✅ System now uses 100% REAL NFL data from Sleeper API")
    print("✅ No more synthetic players!")
    print("🏈 Ready for fantasy drafting with real players!")

if __name__ == "__main__":
    main() 