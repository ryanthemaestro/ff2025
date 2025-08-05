#!/usr/bin/env python3
"""
Integrate nflverse Real Data Into Fantasy System
Replace synthetic data with REAL NFL statistics from nflverse

REAL NFL STATISTICS ONLY - NO SYNTHETIC DATA
"""

import pandas as pd
import shutil
from datetime import datetime
from pathlib import Path

def backup_existing_data():
    """Backup existing synthetic data"""
    backup_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    files_to_backup = [
        'data/fantasy_metrics_2024.csv',
        'data/players.json'
    ]
    
    print("💾 BACKING UP EXISTING SYNTHETIC DATA...")
    for file_path in files_to_backup:
        if Path(file_path).exists():
            backup_path = f"{file_path}.synthetic_backup_{backup_time}"
            shutil.copy2(file_path, backup_path)
            print(f"   ✅ Backed up {file_path} → {backup_path}")

def fix_column_names():
    """Fix the column name issue in draft_optimizer.py"""
    print("🔧 FIXING COLUMN NAME ISSUES...")
    
    # Read the draft optimizer file
    with open('scripts/draft_optimizer.py', 'r') as f:
        content = f.read()
    
    # Replace player_name with name
    if "verified_df['name'] = verified_df['player_name'].str.upper()" in content:
        content = content.replace(
            "verified_df['name'] = verified_df['player_name'].str.upper()",
            "verified_df['name'] = verified_df['name'].str.upper()"
        )
        print("   ✅ Fixed player_name → name column reference")
        
        # Write back the file
        with open('scripts/draft_optimizer.py', 'w') as f:
            f.write(content)
    else:
        print("   ℹ️  Column name fix not needed or already applied")

def integrate_nflverse_data():
    """Replace synthetic data with real nflverse data"""
    print("🏈 INTEGRATING REAL NFLVERSE DATA...")
    
    # Find the most recent nflverse data file
    nflverse_files = list(Path('.').glob('nflverse_fantasy_data_*.csv'))
    if not nflverse_files:
        print("❌ No nflverse data files found!")
        return False
    
    latest_file = max(nflverse_files, key=lambda p: p.stat().st_mtime)
    print(f"📁 Using latest file: {latest_file}")
    
    # Load nflverse data
    nflverse_data = pd.read_csv(latest_file)
    print(f"✅ Loaded {len(nflverse_data)} real NFL players")
    
    # Create players.json format for injury status
    players_data = []
    for _, player in nflverse_data.iterrows():
        player_dict = {
            'name': player['name'],
            'position': player['position'],
            'team': player['team'],
            'injury_status': player['injury_status']
        }
        players_data.append(player_dict)
    
    # Save as players.json
    import json
    with open('data/players.json', 'w') as f:
        json.dump(players_data, f, indent=2)
    print(f"✅ Created data/players.json with {len(players_data)} real players")
    
    # Create fantasy_metrics_2024.csv format
    fantasy_metrics = nflverse_data[[
        'name', 'position', 'team', 'projected_points', 'fantasy_points_2024', 
        'avg_points', 'passing_yards', 'passing_tds', 'rushing_yards', 'rushing_tds',
        'receiving_yards', 'receiving_tds', 'receptions', 'targets', 'games_played',
        'consistency_score', 'age', 'years_exp', 'injury_status', 'adp'
    ]].copy()
    
    # Add any missing columns our system expects
    if 'player_name' not in fantasy_metrics.columns:
        fantasy_metrics['player_name'] = fantasy_metrics['name']  # Backward compatibility
    
    fantasy_metrics.to_csv('data/fantasy_metrics_2024.csv', index=False)
    print(f"✅ Created data/fantasy_metrics_2024.csv with {len(fantasy_metrics)} real players")
    
    return True

def main():
    """Main integration process"""
    print("🚀 NFLVERSE DATA INTEGRATION")
    print("=" * 50)
    
    # Step 1: Backup existing synthetic data
    backup_existing_data()
    
    # Step 2: Fix column name issues in code
    fix_column_names()
    
    # Step 3: Integrate real nflverse data
    success = integrate_nflverse_data()
    
    if success:
        print("\n🎉 INTEGRATION COMPLETE!")
        print("✅ Replaced synthetic data with REAL NFL statistics")
        print("✅ Fixed column name issues")
        print("✅ System ready with nflverse data")
        print("\n🏈 Top 5 Players in New Dataset:")
        
        # Show top players
        df = pd.read_csv('data/fantasy_metrics_2024.csv')
        top_5 = df[['name', 'position', 'team', 'projected_points']].head()
        print(top_5.to_string(index=False))
        
        print(f"\n📊 Total Real NFL Players: {len(df)}")
        print(f"📋 Position Breakdown:")
        pos_counts = df['position'].value_counts()
        print(pos_counts.to_string())
        
        return True
    else:
        print("\n❌ Integration failed!")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready to restart draft UI with REAL NFL data!")
    else:
        print("\n💥 Integration issues need to be resolved") 